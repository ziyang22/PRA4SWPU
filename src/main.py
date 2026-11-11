"""Command-line workflows; run `python src/main.py --help`."""

import argparse
import importlib
import json
from pathlib import Path

from benchmark import benchmark
from data import inspect_data
from engine import evaluate_checkpoint, predict, train
from utils import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seismic U-Net baseline and operator benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate data and compute SHA256 summaries")
    inspect.add_argument("--data-dir", default="data")
    inspect.add_argument("--output", default="outputs/data-inspection.json")
    training = commands.add_parser("train", help="Train native PyTorch U-Net")
    training.add_argument("--config", default="configs/baseline.json")
    training.add_argument("--data-dir")
    training.add_argument("--output-dir")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    for command in ("evaluate", "predict", "benchmark"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--data-dir")
        sub.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
        sub.add_argument("--backend", default="example" if command == "benchmark" else "torch")
        sub.add_argument(
            "--backend-module", help="Import a trusted Python module registering a backend"
        )
        sub.add_argument(
            "--output", default=f"outputs/{command}" + (".json" if command == "benchmark" else "")
        )
        if command in ("evaluate", "predict"):
            sub.add_argument("--batch-size", type=int)
        if command == "predict":
            sub.add_argument("--normalize-test", action="store_true")
            sub.add_argument("--threshold", type=float, default=0.3)
            sub.add_argument("--csv", action="store_true", help="Also export index,value CSV")
        if command == "benchmark":
            sub.add_argument("--warmup", type=int, default=5)
            sub.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", None) is not None and args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if getattr(args, "backend_module", None):
        importlib.import_module(args.backend_module)
    if args.command == "inspect":
        filenames = ["data.npy", "label.npy"]
        if (Path(args.data_dir) / "new_test_data.npy").is_file():
            filenames.append("new_test_data.npy")
        values = inspect_data(args.data_dir, filenames)
        write_json(args.output, values)
        print(json.dumps(values, ensure_ascii=False, indent=2))
    elif args.command == "train":
        with open(args.config) as stream:
            config = json.load(stream)
        for key in ("data_dir", "output_dir", "epochs", "batch_size", "device"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        train(config)
    elif args.command == "evaluate":
        evaluate_checkpoint(
            args.checkpoint, args.backend, args.device, args.output, args.data_dir, args.batch_size
        )
    elif args.command == "predict":
        predict(
            args.checkpoint,
            args.backend,
            args.device,
            args.output,
            args.data_dir,
            args.batch_size,
            args.normalize_test,
            args.threshold,
            args.csv,
        )
    else:
        benchmark(
            args.checkpoint,
            args.backend,
            args.device,
            args.output,
            args.warmup,
            args.repeats,
            args.data_dir,
        )


if __name__ == "__main__":
    main()
