"""Check the real pybind11/HIP extension on an allocated DCU, not the login node."""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from data import normalize_image
from ops import ReplaceableConv2d, hip_conv2d, load_hip_extension
from utils import load_checkpoint, metadata, seed_everything, write_json


def compare(actual, expected):
    error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    return error


@torch.inference_mode()
def check(checkpoint, data_dir):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This check requires an allocated DCU and HIP PyTorch")
    device = torch.device("cuda:0")
    seed_everything(12345, 4)
    result = {"status": "running", "environment": metadata(device), "hip": torch.version.hip}
    result["allocation"] = {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    }
    start = time.perf_counter()
    extension = load_hip_extension()
    result["extension_build_load_seconds"] = time.perf_counter() - start
    result["extension"] = extension.__file__
    result["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((Path(__file__).resolve().parents[1] / "src/csrc").iterdir())
        if p.is_file()
    }
    cases = []
    for kernel in (1, 3):
        for padding in ((0, 0), (1, 1), (0, 1)):
            for bias_enabled in (False, True):
                x = torch.randn(2, 3, 11, 13, device=device)
                weight = torch.randn(5, 3, kernel, kernel, device=device)
                bias = torch.randn(5, device=device) if bias_enabled else None
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    actual = hip_conv2d(x, weight, bias, (1, 1), padding, (1, 1), 1)
                    expected = F.conv2d(x, weight, bias, padding=padding)
                stream.synchronize()
                cases.append(
                    {
                        "kernel": kernel,
                        "padding": padding,
                        "bias": bias_enabled,
                        "max_abs_error": compare(actual, expected),
                    }
                )
    result["operator_cases"] = cases
    x = torch.randn(1, 3, 11, 13, device=device)
    w = torch.randn(5, 3, 3, 3, device=device)
    b = torch.randn(5, device=device)
    errors = {
        "noncontiguous": (x.transpose(2, 3), w, b, (1, 1), (1, 1), (1, 1), 1),
        "stride": (x, w, b, (2, 2), (1, 1), (1, 1), 1),
        "dilation": (x, w, b, (1, 1), (1, 1), (2, 2), 1),
        "groups": (x, w, b, (1, 1), (1, 1), (1, 1), 3),
        "dtype": (x.half(), w.half(), b.half(), (1, 1), (1, 1), (1, 1), 1),
        "bias_shape": (x, w, b[:2], (1, 1), (1, 1), (1, 1), 1),
        "device": (x, w.cpu(), b, (1, 1), (1, 1), (1, 1), 1),
        "padding": (x, w, b, (1, 1), (2, 2), (1, 1), 1),
        "channels": (x[:, :2].contiguous(), w, b, (1, 1), (1, 1), (1, 1), 1),
    }
    result["rejected_inputs"] = {}
    for name, args in errors.items():
        try:
            hip_conv2d(*args)
        except RuntimeError as error:
            result["rejected_inputs"][name] = str(error).splitlines()[0]
        else:
            raise AssertionError(f"Invalid input accepted: {name}")
    reference, saved = load_checkpoint(checkpoint, "torch", device)
    custom, _ = load_checkpoint(checkpoint, "hip", device)
    result["parameters"] = sum(p.numel() for p in reference.parameters())
    # Exercise the native training path without changing the supplied checkpoint.
    from model import UNet

    with torch.inference_mode(False), torch.enable_grad():
        training = UNet().to(device).train()
        optimizer = torch.optim.Adam(training.parameters(), lr=1e-4, eps=1e-7)
        sample = torch.randn(2, 1, 32, 32, device=device)
        loss = F.binary_cross_entropy_with_logits(training(sample), torch.zeros_like(sample))
        loss.backward()
        if not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in training.parameters()
        ):
            raise AssertionError("Native training produced missing/nonfinite gradients")
        optimizer.step()
        result["native_training_step_bce"] = loss.item()
    images = np.load(Path(data_dir) / "data.npy", mmap_mode="r", allow_pickle=False)
    image = normalize_image(images[saved["config"]["split_index"]])
    real = torch.from_numpy(image[None, None]).to(device)
    result["models"] = []
    for x in (real, torch.randn(1, 1, 256, 640, device=device)):
        layers = []
        handles = []
        for name, layer in reference.named_modules():
            if isinstance(layer, ReplaceableConv2d):

                def capture(module, args, expected, name=name):
                    actual = hip_conv2d(
                        args[0],
                        module.weight,
                        module.bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                    layers.append(
                        {
                            "name": name,
                            "shape": list(args[0].shape),
                            "weight": list(module.weight.shape),
                            "max_abs_error": compare(actual, expected),
                        }
                    )

                handles.append(layer.register_forward_hook(capture))
        try:
            expected = reference(x)
        finally:
            for handle in handles:
                handle.remove()
        actual = custom(x)
        result["models"].append(
            {
                "shape": list(x.shape),
                "layers": layers,
                "logits_max_abs_error": compare(actual, expected),
                "probability_max_abs_error": compare(actual.sigmoid(), expected.sigmoid()),
                "mask_disagreements": int(
                    ((actual.sigmoid() > 0.5) != (expected.sigmoid() > 0.5)).sum()
                ),
            }
        )
    result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights/baseline.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/hip-check.json")
    args = parser.parse_args()
    try:
        report = check(args.checkpoint, args.data_dir)
    except Exception as error:
        write_json(args.output, {"status": "failed", "error": repr(error)})
        raise
    write_json(args.output, report)
    print(f"HIP_CHECK=passed; report={args.output}")
