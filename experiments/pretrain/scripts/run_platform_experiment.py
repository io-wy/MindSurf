import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments" / "pretrain" / "configs" / "pretrain_platform_experiments.json"


TRAIN_ARG_MAP = {
    "data_path": "--data_path",
    "data_mix_json": "--data_mix_json",
    "tokenizer_path": "--tokenizer_path",
    "init_weight": "--init_weight",
    "epochs": "--epochs",
    "max_steps": "--max_steps",
    "batch_size": "--batch_size",
    "accumulation_steps": "--accumulation_steps",
    "learning_rate": "--learning_rate",
    "weight_decay": "--weight_decay",
    "adam_beta1": "--adam_beta1",
    "adam_beta2": "--adam_beta2",
    "adam_eps": "--adam_eps",
    "warmup_steps": "--warmup_steps",
    "lr_schedule": "--lr_schedule",
    "min_lr_ratio": "--min_lr_ratio",
    "lr_stable_ratio": "--lr_stable_ratio",
    "hidden_size": "--hidden_size",
    "num_hidden_layers": "--num_hidden_layers",
    "num_attention_heads": "--num_attention_heads",
    "num_key_value_heads": "--num_key_value_heads",
    "intermediate_size": "--intermediate_size",
    "max_seq_len": "--max_seq_len",
    "num_workers": "--num_workers",
    "grad_clip": "--grad_clip",
    "dtype": "--dtype",
    "device": "--device",
    "seed": "--seed",
    "log_interval": "--log_interval",
    "use_moe": "--use_moe",
    "shuffle_buffer": "--shuffle_buffer",
    "skip_blocks": "--skip_blocks",
}

TRAIN_BOOL_ARGS = {
    "fused_adamw": "--fused_adamw",
    "packed": "--packed",
    "stream_packed": "--stream_packed",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stringify_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def merge_train(defaults: dict, stage: dict) -> dict:
    merged = {}
    merged.update(defaults.get("common_train", {}))
    merged["tokenizer_path"] = defaults.get("tokenizer_path", "model")
    if "data_path" not in stage and "data_mix_json" not in stage:
        merged["data_path"] = defaults.get("data_path")
    merged.update(stage)
    return merged


def command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def build_train_command(python_exe: str, train_script: Path, params: dict, save_dir: Path, save_weight: str) -> list[str]:
    command = [
        python_exe,
        str(train_script),
        "--save_dir",
        str(save_dir),
        "--save_weight",
        save_weight,
    ]
    for key, flag in TRAIN_ARG_MAP.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if key in {"data_path", "data_mix_json", "tokenizer_path", "init_weight"}:
            value = stringify_path(value)
        command.extend([flag, str(value)])
    for key, flag in TRAIN_BOOL_ARGS.items():
        if params.get(key):
            command.append(flag)
    return command


def build_eval_command(
    python_exe: str,
    eval_script: Path,
    data_path: Path,
    weight_path: Path,
    output_path: Path,
    stage_params: dict,
    max_batches: int,
) -> list[str]:
    command = [
        python_exe,
        str(eval_script),
        "--data_path",
        str(data_path),
        "--weight_path",
        str(weight_path),
        "--output",
        str(output_path),
        "--hidden_size",
        str(stage_params.get("hidden_size", 768)),
        "--num_hidden_layers",
        str(stage_params.get("num_hidden_layers", 8)),
        "--num_attention_heads",
        str(stage_params.get("num_attention_heads", 8)),
        "--num_key_value_heads",
        str(stage_params.get("num_key_value_heads", 4)),
        "--max_seq_len",
        str(stage_params.get("max_seq_len", 512)),
        "--batch_size",
        "8",
        "--max_batches",
        str(max_batches),
        "--dtype",
        str(stage_params.get("dtype", "bfloat16")),
    ]
    if stage_params.get("intermediate_size") is not None:
        command.extend(["--intermediate_size", str(stage_params["intermediate_size"])])
    return command


def run_command(command: list[str], log_path: Path, timeout: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + command_text(command) + "\n\n")
        log.flush()
        try:
            return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT\n")
            return -9


def find_experiment(config: dict, name: str) -> dict:
    for item in config.get("experiments", []):
        if item.get("name") == name:
            return item
    names = ", ".join(item.get("name", "") for item in config.get("experiments", []))
    raise SystemExit(f"Unknown experiment '{name}'. Available: {names}")


def list_experiments(config: dict) -> None:
    for item in config.get("experiments", []):
        print(f"{item['name']}: {item.get('purpose', '')}")


def run_experiment(config: dict, experiment: dict, dry_run: bool, run_eval: bool, skip_train: bool, output_root_override: str | None) -> None:
    defaults = config["defaults"]
    python_exe = defaults.get("python", sys.executable)
    output_root = resolve_path(output_root_override or defaults["output_root"])
    train_script = resolve_path(defaults["train_script"])
    eval_script = resolve_path(defaults["eval_script"])
    exp_dir = output_root / experiment["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    previous_weight = None
    final_weight = None
    final_stage_params = None
    run_manifest = {
        "experiment": experiment["name"],
        "purpose": experiment.get("purpose", ""),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "stages": [],
    }

    for index, raw_stage in enumerate(experiment.get("stages", []), start=1):
        stage = dict(raw_stage)
        tag = stage.get("tag", f"stage{index}")
        stage_dir = exp_dir / f"{index:02d}_{tag}"
        save_weight = stage.get("save_weight", f"{experiment['name']}_{tag}")
        if stage.get("init_from_previous"):
            if previous_weight is None:
                raise SystemExit(f"Stage {tag} requested init_from_previous but no previous weight exists.")
            stage["init_weight"] = str(previous_weight)
        params = merge_train(defaults, stage)
        command = build_train_command(python_exe, train_script, params, stage_dir, save_weight)
        hidden_size = int(params.get("hidden_size", 768))
        weight_path = stage_dir / f"{save_weight}_{hidden_size}.pth"
        summary_path = stage_dir / f"{save_weight}_{hidden_size}_summary.json"
        stage_manifest = {
            "tag": tag,
            "save_dir": str(stage_dir),
            "save_weight": save_weight,
            "weight_path": str(weight_path),
            "summary_path": str(summary_path),
            "command": command,
        }
        run_manifest["stages"].append(stage_manifest)
        (stage_dir / "train_command.txt").parent.mkdir(parents=True, exist_ok=True)
        (stage_dir / "train_command.txt").write_text(command_text(command) + "\n", encoding="utf-8")
        print(f"train stage {index}/{len(experiment['stages'])}: {tag}")
        print(command_text(command))
        if not dry_run and not skip_train:
            if summary_path.exists() and weight_path.exists():
                print(f"skip existing stage: {summary_path}")
            else:
                code = run_command(command, stage_dir / "train.log", timeout=None)
                if code != 0:
                    write_json(exp_dir / "run_manifest.json", run_manifest)
                    raise SystemExit(f"stage {tag} failed with code {code}")
        previous_weight = weight_path
        final_weight = weight_path
        final_stage_params = params

    if run_eval and final_weight and final_stage_params:
        eval_results = []
        for eval_item in defaults.get("eval_sets", []):
            data_path = resolve_path(eval_item["data_path"])
            output_path = exp_dir / f"eval_{eval_item['name']}.json"
            command = build_eval_command(
                python_exe,
                eval_script,
                data_path,
                final_weight,
                output_path,
                final_stage_params,
                int(eval_item.get("max_batches", 250)),
            )
            (exp_dir / f"eval_{eval_item['name']}_command.txt").write_text(command_text(command) + "\n", encoding="utf-8")
            print(f"eval {eval_item['name']}")
            print(command_text(command))
            result = {"name": eval_item["name"], "data_path": str(data_path), "output": str(output_path), "command": command}
            if not dry_run:
                if not data_path.exists():
                    result["status"] = "missing_data"
                elif not final_weight.exists():
                    result["status"] = "missing_weight"
                elif output_path.exists():
                    result["status"] = "existing"
                else:
                    code = run_command(command, exp_dir / f"eval_{eval_item['name']}.log", timeout=None)
                    result["status"] = "ok" if code == 0 and output_path.exists() else f"eval_failed_{code}"
            eval_results.append(result)
        run_manifest["evals"] = eval_results

    run_manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(exp_dir / "run_manifest.json", run_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run platform-ready MiniMind pretraining experiments from JSON config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", help="Experiment name from the config.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifests without running training.")
    parser.add_argument("--run-eval", action="store_true", help="Evaluate the final checkpoint on configured eval sets.")
    parser.add_argument("--skip-train", action="store_true", help="Only run evals for an existing final checkpoint.")
    parser.add_argument("--output-root", default=None, help="Override defaults.output_root.")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    if args.list:
        list_experiments(config)
        return
    if not args.experiment:
        raise SystemExit("--experiment is required unless --list is used.")
    experiment = find_experiment(config, args.experiment)
    run_experiment(config, experiment, args.dry_run, args.run_eval, args.skip_train, args.output_root)


if __name__ == "__main__":
    main()
