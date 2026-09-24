#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence


MANAGED_OPTIONS = ("--lrs", "--cuda_device", "--output_dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent GRASP learning rates across multiple GPUs")
    parser.add_argument("--lrs", type=float, nargs="+", required=True)
    parser.add_argument("--cuda_devices", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to grasp-tool train-moco; place them after --",
    )
    return parser


def build_train_command(
    *,
    learning_rate: float,
    cuda_device: int,
    output_dir: Path,
    train_args: Sequence[str],
) -> list[str]:
    forwarded_args = list(train_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]

    for argument in forwarded_args:
        if any(argument == option or argument.startswith(f"{option}=") for option in MANAGED_OPTIONS):
            raise ValueError(f"{argument} is managed by this launcher and cannot be forwarded")

    return [
        sys.executable,
        "-m",
        "grasp_tool",
        "train-moco",
        *forwarded_args,
        "--lrs",
        str(learning_rate),
        "--cuda_device",
        str(cuda_device),
        "--output_dir",
        str(output_dir),
    ]


def run_device_queue(
    cuda_device: int,
    learning_rates: Sequence[float],
    output_root: Path,
    train_args: Sequence[str],
) -> list[int]:
    return_codes = []
    for learning_rate in learning_rates:
        rate_name = f"{learning_rate:.12g}".replace("-", "m").replace(".", "p")
        command = build_train_command(
            learning_rate=learning_rate,
            cuda_device=cuda_device,
            output_dir=output_root / f"lr_{rate_name}",
            train_args=train_args,
        )
        print(
            f"[GPU {cuda_device}] starting learning rate {learning_rate}",
            flush=True,
        )
        return_codes.append(subprocess.run(command, check=False).returncode)
    return return_codes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.cuda_devices)) != len(args.cuda_devices):
        parser.error("--cuda_devices must not contain duplicates")
    if len(set(args.lrs)) != len(args.lrs):
        parser.error("--lrs must not contain duplicates")
    assignments = [args.lrs[index :: len(args.cuda_devices)] for index in range(len(args.cuda_devices))]
    active_queues = [
        (device, learning_rates) for device, learning_rates in zip(args.cuda_devices, assignments) if learning_rates
    ]

    print(
        f"Launching {len(args.lrs)} independent runs on "
        f"{len(active_queues)} GPU(s). Each process loads the training PKL separately.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(active_queues)) as executor:
        futures = [
            executor.submit(
                run_device_queue,
                device,
                learning_rates,
                args.output_dir,
                args.train_args,
            )
            for device, learning_rates in active_queues
        ]
        return_codes = [return_code for future in futures for return_code in future.result()]

    return 0 if all(return_code == 0 for return_code in return_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
