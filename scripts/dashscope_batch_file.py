#!/usr/bin/env python3
"""Submit, inspect, and download DashScope OpenAI-compatible Batch File jobs."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def api_key(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise SystemExit(f"environment variable {env_name} is not set")
    return value


def request(url: str, key: str, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    merged = {"Authorization": f"Bearer {key}"}
    merged.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=merged, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc


def multipart_file(path: Path) -> tuple[bytes, str]:
    boundary = "----LexChain" + secrets.token_hex(12)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), boundary


def submit(args: argparse.Namespace) -> int:
    key = api_key(args.api_key_env)
    body, boundary = multipart_file(args.input)
    uploaded = json.loads(request(f"{args.base_url}/files", key, "POST", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    payload = json.dumps({
        "input_file_id": uploaded["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"ds_name": args.name, "ds_description": args.description},
    }).encode()
    batch = json.loads(request(f"{args.base_url}/batches", key, "POST", payload, {"Content-Type": "application/json"}))
    state = {"input_path": str(args.input.resolve()), "input_file_id": uploaded["id"], "batch": batch}
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def status(args: argparse.Namespace) -> int:
    key = api_key(args.api_key_env)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    batch_id = state["batch"]["id"]
    batch = json.loads(request(f"{args.base_url}/batches/{batch_id}", key))
    state["batch"] = batch
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    return 0


def download(args: argparse.Namespace) -> int:
    key = api_key(args.api_key_env)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    batch = state["batch"]
    if batch.get("status") != "completed" or not batch.get("output_file_id"):
        raise SystemExit(f"batch is not completed: {batch.get('status')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(request(f"{args.base_url}/files/{batch['output_file_id']}/content", key))
    if batch.get("error_file_id") and args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        args.errors.write_bytes(request(f"{args.base_url}/files/{batch['error_file_id']}/content", key))
    print(json.dumps({"output": str(args.output.resolve()), "bytes": args.output.stat().st_size}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--base-url", default=BASE_URL)
    root.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("submit")
    create.add_argument("--input", type=Path, required=True)
    create.add_argument("--state", type=Path, required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--description", default="")
    create.set_defaults(handler=submit)
    show = commands.add_parser("status")
    show.add_argument("--state", type=Path, required=True)
    show.set_defaults(handler=status)
    fetch = commands.add_parser("download")
    fetch.add_argument("--state", type=Path, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--errors", type=Path)
    fetch.set_defaults(handler=download)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(args.handler(args))
