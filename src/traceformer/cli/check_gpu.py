"""Check that the selected PyTorch build can execute CUDA kernels."""

from __future__ import annotations

import argparse
import json

from traceformer.training.runtime import prepare_device


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="traceformer-check-gpu",
        description="Verify the PyTorch/CUDA/GPU runtime before TraceFormer training",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="device to verify (default: cuda)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="PyTorch CPU thread count used alongside the selected device",
    )
    arguments = parser.parse_args()
    try:
        runtime = prepare_device(arguments.device, num_threads=arguments.num_threads)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(runtime.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
