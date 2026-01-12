#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_api_key():
    # Load .env from current working directory (and parents if you want to extend it)
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in your environment or in a .env file as:\n"
            "OPENAI_API_KEY=sk-..."
        )
    return key


def call_model(client: OpenAI, model: str, prompt_text: str) -> str:
    # Responses API: simplest text-in/text-out usage.
    # The model is instructed by your prompt to output JSON.
    resp = client.responses.create(
        model=model,
        input=prompt_text,
    )
    return resp.output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="netlists/diff_amps", help="Base directory containing circuit subfolders")
    parser.add_argument("--start", type=int, default=921, help="Start ID (inclusive)")
    parser.add_argument("--end", type=int, default=932, help="End ID (inclusive)")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model name")
    parser.add_argument(
        "--prompt-filename",
        default="graph_query_prompt.txt",
        help="Prompt filename inside each circuit directory",
    )
    parser.add_argument(
        "--out-filename",
        default="fun_graph.json",
        help="Output JSON filename inside each circuit directory",
    )
    args = parser.parse_args()

    load_api_key()  # ensures env is loaded / key exists
    client = OpenAI()

    base = Path(args.base_dir)

    for i in range(args.start, args.end + 1):
        circuit_dir = base / str(i)
        prompt_path = circuit_dir / args.prompt_filename
        out_path = circuit_dir / args.out_filename

        if not prompt_path.is_file():
            print(f"Skipping {i}: prompt not found at {prompt_path}")
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")

        print(f"Calling model for {i} ({prompt_path}) ...")
        try:
            output_text = call_model(client, args.model, prompt_text)
        except Exception as e:
            print(f"ERROR {i}: API call failed: {e}")
            continue

        # Try to parse JSON and save pretty-formatted JSON if valid.
        try:
            obj = json.loads(output_text)
            out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Wrote {out_path}")
        except json.JSONDecodeError:
            # If the model returned extra text, keep it for debugging.
            raw_path = out_path.with_suffix(out_path.suffix + ".raw.txt")
            raw_path.write_text(output_text, encoding="utf-8")
            print(f"WARNING {i}: Output was not valid JSON. Wrote raw output to {raw_path}")


if __name__ == "__main__":
    main()
