import argparse
import asyncio
import json
import re
import os
import sys
from dataclasses import dataclass
from pathlib import Path

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Lock, Thread
from queue import Empty, Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()
ROOT = Path(__file__).resolve().parents[1]
runtime_max_seq_len = 2048
dynamic_batching_enabled = False
dynamic_batch_max_size = 4
dynamic_batch_wait_seconds = 0.008
batch_request_queue = Queue()
tokenizer_lock = Lock()
service_config = {}
batch_stats_lock = Lock()
batch_stats = {
    "batches": 0,
    "jobs": 0,
    "max_batch_size": 0,
    "last_batch_size": 0,
    "last_input_token_counts": [],
}


@app.get("/healthz")
async def healthz():
    with batch_stats_lock:
        stats = dict(batch_stats)
        stats["last_input_token_counts"] = list(batch_stats["last_input_token_counts"])
    return {
        "status": "ok",
        "model_loaded": "model" in globals(),
        "device": str(globals().get("device", "")),
        "dynamic_batching": dynamic_batching_enabled,
        "batch_max_size": dynamic_batch_max_size,
        "batch_wait_ms": dynamic_batch_wait_seconds * 1000,
        "max_seq_len": runtime_max_seq_len,
        "service_config": service_config,
        "batch_stats": stats,
    }


def resolve_project_path(raw_path):
    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def init_model(args):
    load_from = resolve_project_path(args.load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = resolve_project_path(args.weight_path) if args.weight_path else resolve_project_path(
            f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        )
        config_kwargs = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "max_position_embeddings": max(args.max_seq_len, 2048),
            "use_moe": bool(args.use_moe),
            "inference_rope_scaling": args.inference_rope_scaling,
        }
        if args.intermediate_size is not None:
            config_kwargs["intermediate_size"] = args.intermediate_size
        model = MiniMindForCausalLM(MiniMindConfig(**config_kwargs))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, resolve_project_path(f'{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'))
    else:
        model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.half().eval().to(device), tokenizer


@dataclass
class BatchJob:
    request: "ChatRequest"
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    prompt: str
    input_token_count: int


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: dict = None

    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def build_chat_prompt(messages, tools=None, open_thinking=False):
    with tokenizer_lock:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            open_thinking=open_thinking,
        )


def tokenize_prompts(prompts, max_new_tokens=512):
    max_input_tokens = max(1, runtime_max_seq_len - max(1, max_new_tokens))
    with tokenizer_lock:
        return tokenizer(prompts, return_tensors="pt", truncation=True, max_length=max_input_tokens, padding=True).to(device)


def count_prompt_tokens(prompt: str, max_new_tokens=512) -> int:
    max_input_tokens = max(1, runtime_max_seq_len - max(1, max_new_tokens))
    with tokenizer_lock:
        encoded = tokenizer(prompt, truncation=True, max_length=max_input_tokens)
    return len(encoded["input_ids"])


def build_chat_inputs(messages, tools=None, open_thinking=False, max_new_tokens=512):
    prompt = build_chat_prompt(messages, tools=tools, open_thinking=open_thinking)
    return tokenize_prompts(prompt, max_new_tokens=max_new_tokens)


def build_generate_kwargs(input_ids, attention_mask, max_tokens, temperature, top_p):
    do_sample = temperature > 1e-10
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


def build_completion_response(request: ChatRequest, answer: str) -> dict:
    content, reasoning_content, tool_calls = parse_response(answer)
    message = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }
        ]
    }


def is_dynamic_batchable(request: ChatRequest) -> bool:
    return (
        dynamic_batching_enabled
        and not request.stream
        and not request.tools
        and not request.get_open_thinking()
    )


def batch_key(job: BatchJob):
    request = job.request
    return (request.temperature, request.top_p, request.max_tokens)


def complete_batch_job(job: BatchJob, result=None, error: Exception = None):
    if job.future.done():
        return
    if error is not None:
        job.loop.call_soon_threadsafe(job.future.set_exception, error)
    else:
        job.loop.call_soon_threadsafe(job.future.set_result, result)


def run_batch_jobs(jobs: list[BatchJob]) -> None:
    request = jobs[0].request
    try:
        with batch_stats_lock:
            batch_stats["batches"] += 1
            batch_stats["jobs"] += len(jobs)
            batch_stats["max_batch_size"] = max(batch_stats["max_batch_size"], len(jobs))
            batch_stats["last_batch_size"] = len(jobs)
            batch_stats["last_input_token_counts"] = [job.input_token_count for job in jobs]
        prompts = [job.prompt for job in jobs]
        inputs = tokenize_prompts(prompts, max_new_tokens=request.max_tokens)
        with torch.no_grad():
            generated_ids = model.generate(
                **build_generate_kwargs(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                )
            )
        prompt_width = inputs["input_ids"].shape[1]
        for index, job in enumerate(jobs):
            with tokenizer_lock:
                answer = tokenizer.decode(generated_ids[index][prompt_width:], skip_special_tokens=True)
            complete_batch_job(job, build_completion_response(job.request, answer))
    except Exception as exc:
        for job in jobs:
            complete_batch_job(job, error=exc)


def dynamic_batch_worker() -> None:
    while True:
        first_job = batch_request_queue.get()
        jobs = [first_job]
        key = batch_key(first_job)
        deadline = time.monotonic() + dynamic_batch_wait_seconds
        while len(jobs) < dynamic_batch_max_size:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0:
                break
            try:
                job = batch_request_queue.get(timeout=timeout)
            except Empty:
                break
            if batch_key(job) == key:
                jobs.append(job)
            else:
                batch_request_queue.put(job)
                break
        run_batch_jobs(jobs)


async def enqueue_dynamic_batch(request: ChatRequest) -> dict:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    prompt = build_chat_prompt(request.messages, tools=None, open_thinking=False)
    input_token_count = count_prompt_tokens(prompt, max_new_tokens=request.max_tokens)
    batch_request_queue.put(BatchJob(request=request, future=future, loop=loop, prompt=prompt, input_token_count=input_token_count))
    return await future


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        inputs = build_chat_inputs(
            messages=messages,
            tools=tools,
            open_thinking=open_thinking,
            max_new_tokens=max_tokens,
        )

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                **build_generate_kwargs(
                    inputs.input_ids,
                    inputs.attention_mask,
                    max_tokens,
                    temperature,
                    top_p,
                ),
                streamer=streamer
            )

        Thread(target=_generate).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            full_text += text

            if not thinking_ended:
                pos = full_text.find('</think>')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            if is_dynamic_batchable(request):
                return await enqueue_dynamic_batch(request)
            inputs = build_chat_inputs(
                messages=request.messages,
                tools=request.tools,
                open_thinking=request.get_open_thinking(),
                max_new_tokens=request.max_tokens,
            )
            with torch.no_grad():
                generated_ids = model.generate(
                    **build_generate_kwargs(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        request.max_tokens,
                        request.temperature,
                        request.top_p,
                    )
                )
                with tokenizer_lock:
                    answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return build_completion_response(request, answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--weight_path', default='', type=str, help="直接指定 PyTorch 权重路径；设置后优先于 save_dir/weight")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数量")
    parser.add_argument('--num_key_value_heads', default=4, type=int, help="KV 头数量")
    parser.add_argument('--intermediate_size', default=None, type=int, help="FFN 中间层宽度；默认使用 MiniMindConfig 默认值")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--host', default='0.0.0.0', type=str, help="服务监听地址")
    parser.add_argument('--port', default=8998, type=int, help="服务监听端口")
    parser.add_argument('--dynamic_batching', action='store_true', help="对非 streaming 普通 chat 请求启用最小动态 batching")
    parser.add_argument('--batch_max_size', default=4, type=int, help="动态 batching 单批最大请求数")
    parser.add_argument('--batch_wait_ms', default=8.0, type=float, help="动态 batching 等待窗口，毫秒")
    args = parser.parse_args()
    device = args.device
    runtime_max_seq_len = args.max_seq_len
    dynamic_batching_enabled = args.dynamic_batching
    dynamic_batch_max_size = max(1, args.batch_max_size)
    dynamic_batch_wait_seconds = max(0.0, args.batch_wait_ms / 1000.0)
    service_config = {
        "load_from": args.load_from,
        "weight_path": args.weight_path,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "intermediate_size": args.intermediate_size,
    }
    model, tokenizer = init_model(args)
    if dynamic_batching_enabled:
        Thread(target=dynamic_batch_worker, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port)
