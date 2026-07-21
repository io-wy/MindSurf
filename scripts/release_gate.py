"""Enforce the publication rules in PROJECT_RULES.md sections 7 and 11.

Those sections already define what may not reach the shared repository: agent
process records, personal usernames, internal addresses, absolute paths, SSH
details, credentials, and model or dataset blobs. They were written as a manual
checklist, which is why a blunt local exclude of an entire directory was
standing in for them: a directory rule cannot tell a compliant experiment
report from a leaked absolute path, so it discarded documentation while the
real traces went through untouched.

Run this against the staged set before committing, or against a commit range
before pushing to the shared branch::

    python scripts/release_gate.py
    python scripts/release_gate.py --range origin/pretrain...HEAD

Read the exit code, and do not pipe it into ``tail`` or ``head`` when chaining
with ``&&``: a pipeline reports the last command's status, so a failing gate
becomes a success and the commit proceeds. That has already happened once.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# PROJECT_RULES.md documents the first two as the standard's own text. The gate
# and its tests state the forbidden patterns by necessity: a test that pins
# "this credential shape is caught" has to contain that shape.
EXEMPT = {
    "PROJECT_RULES.md",
    "AGENTS.md",
    "scripts/release_gate.py",
    "tests/test_release_gate.py",
}

# A dotted quad in a generated dependency manifest is a package version, not a
# host: nvidia-curand-cu12 10.3.5.147 parses as an RFC1918 address. Nobody
# writes a server address into a lock file, so exempt that one rule there.
RULE_EXEMPT_PATHS: dict[str, re.Pattern[str]] = {
    "internal_address": re.compile(r"(^|/)(uv|poetry|Cargo)\.lock$|-lock\.json$"),
}

CONTENT_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "agent_process_record",
        r"Codex|Claude|ChatGPT|Superpowers",
        "section 7: agent prompts, plans and execution records",
    ),
    (
        "personal_path",
        r"/home/[A-Za-z0-9_-]+|/Users/[A-Za-z0-9_-]+|[A-Z]:\\\\|[A-Z]:/(?:Users|UserData)",
        "section 7: personal usernames and absolute paths",
    ),
    (
        "internal_address",
        r"\b(?:192\.168\.\d{1,3}\.\d{1,3}"
        r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b",
        "section 7: internal addresses",
    ),
    (
        "credential",
        r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|DPAPI",
        "section 11: credentials",
    ),
)

PATH_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "agent_state_directory",
        r"(^|/)\.(codex|agents|codebase-memory|superpowers)/|task-brief|review-package",
        "section 7: local agent state and hand-off packages",
    ),
    (
        "large_asset",
        r"\.(pth|pt|ckpt|safetensors|env|key|pem)$",
        "section 8: weights, datasets and credentials belong outside git",
    ),
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout


def _staged_files() -> list[str]:
    return [line for line in _git("diff", "--cached", "--diff-filter=ACMR", "--name-only").splitlines() if line]


def _range_files(commit_range: str) -> list[str]:
    return [
        line
        for line in _git("diff", "--diff-filter=ACMR", "--name-only", commit_range).splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range",
        dest="commit_range",
        help="Commit range to audit, e.g. origin/pretrain...HEAD; default is the staged set",
    )
    args = parser.parse_args()

    files = _range_files(args.commit_range) if args.commit_range else _staged_files()
    findings: list[str] = []

    for name in files:
        if name in EXEMPT:
            continue
        for rule, pattern, reason in PATH_RULES:
            if re.search(pattern, name, re.IGNORECASE):
                findings.append(f"{name}: {rule} — {reason}")

        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; the path rules already covered blobs
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule, pattern, reason in CONTENT_RULES:
                exempt = RULE_EXEMPT_PATHS.get(rule)
                if exempt and exempt.search(name):
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(f"{name}:{line_number}: {rule} — {reason}")

    scope = args.commit_range or "staged"
    if findings:
        print(f"release gate FAILED for {scope}: {len(findings)} finding(s)")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)
    print(f"release gate passed for {scope}: {len(files)} file(s) checked")


if __name__ == "__main__":
    main()
