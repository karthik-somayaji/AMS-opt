#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in your environment or in a .env file as:\n"
            "OPENAI_API_KEY=sk-..."
        )
    return key


def call_model(client: OpenAI, model: str, prompt_text: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt_text,
    )
    return resp.output_text


def extract_json_text(output_text: str) -> str:
    # Prefer fenced JSON blocks if present.
    fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = fence_re.search(output_text)
    if match:
        return match.group(1).strip()
    return output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="netlists/diff_amps",
        help="Base directory containing circuit subfolders",
    )
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model name")
    parser.add_argument(
        "--prompt-filename",
        default="prune_prompt.txt",
        help="Prompt filename inside each circuit directory",
    )
    args = parser.parse_args()

    load_api_key()
    client = OpenAI()

    base = Path(args.base_dir)
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")

    for circuit_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        circuit_id = circuit_dir.name
        prompt_path = circuit_dir / args.prompt_filename
        out_path = circuit_dir / f"{circuit_id}_prune.json"

        if not prompt_path.is_file():
            print(f"Skipping {circuit_id}: prompt not found at {prompt_path}")
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")

        print(f"Calling model for {circuit_id} ({prompt_path}) ...")
        try:
            output_text = call_model(client, args.model, prompt_text)
        except Exception as e:
            print(f"ERROR {circuit_id}: API call failed: {e}")
            continue

        try:
            json_text = extract_json_text(output_text)
            obj = json.loads(json_text)
            out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Wrote {out_path}")
        except json.JSONDecodeError:
            raw_path = circuit_dir / f"{circuit_id}_prune.raw.txt"
            raw_path.write_text(output_text, encoding="utf-8")
            print(f"WARNING {circuit_id}: Output was not valid JSON. Wrote raw output to {raw_path}")


if __name__ == "__main__":
    main()
