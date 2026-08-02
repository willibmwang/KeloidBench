#!/usr/bin/env python3
"""Presenter-friendly launcher for the KeloidBench Streamlit demonstration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from demo_utils import CURATED_EXAMPLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch KeloidBench on a curated demo-day example."
    )
    parser.add_argument(
        "--example",
        choices=list(CURATED_EXAMPLES),
        default="confident_keloid",
        help="Opening story shown when the website loads.",
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not ask Streamlit to open a local browser.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="Print the curated routes and exit.",
    )
    return parser.parse_args()


def print_examples(selected: str | None = None) -> None:
    print("\nKeloidBench · guided demo routes")
    print("=" * 38)
    for slug, item in CURATED_EXAMPLES.items():
        marker = "→" if slug == selected else " "
        print(f"{marker} {slug:21} {item['title']}")
        print(f"  {'':21} {item['presenter_note']}")
    print()


def main() -> None:
    args = parse_args()
    print_examples(args.example if not args.list_examples else None)
    if args.list_examples:
        return

    item = CURATED_EXAMPLES[args.example]
    print(f"Opening story : {item['title']}")
    print(f"Public sample : {item['sample_id']}")
    print(f"Local URL     : http://{args.address}:{args.port}/?example={args.example}")
    print("Stop server   : Ctrl-C\n")
    sys.stdout.flush()

    environment = os.environ.copy()
    environment["KELOIDBENCH_DEMO_EXAMPLE"] = args.example
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DEMO_DIR / "app.py"),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true" if args.headless else "false",
    ]
    os.chdir(PROJECT_ROOT)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
