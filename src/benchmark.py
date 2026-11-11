"""Correctness-gated, synchronized operator/model/warm-data pipeline benchmarks."""

import statistics
import time
from pathlib import Path

import torch

from data import SeismicDataset, inspect_data
from ops import BACKENDS, ReplaceableConv2d
from utils import (
    load_checkpoint,
    metadata,
    resolve_device,
    seed_everything,
    synchronize,
    write_json,
)


def timed(function, device, warmup, repeats, batch_size):
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    median = statistics.median(samples)
    return dict(
        median_ms=median,
        min_ms=min(samples),
        max_ms=max(samples),
        images_per_second=batch_size * 1000 / median,
        samples_ms=samples,
    )


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    return (actual - expected).abs().max().item()


@torch.inference_mode()
def benchmark(checkpoint_path, backend, device_name, output, warmup=5, repeats=20, data_dir=None):
    if warmup < 0 or repeats < 1:
        raise ValueError("repeats must be positive and warmup nonnegative")
    if backend == "torch" or backend not in BACKENDS:
        raise ValueError("Select a registered custom backend for comparison")
    device = resolve_device(device_name)
    reference, checkpoint = load_checkpoint(checkpoint_path, "torch", device)
    custom, _ = load_checkpoint(checkpoint_path, backend, device)
    config = dict(checkpoint["config"])
    if data_dir:
        config["data_dir"] = data_dir
    seed_everything(config["seed"], config["threads"])
    val = SeismicDataset(config["data_dir"], "val", config["split_index"])
    datasets = [
        ("validation-single", val, 1),
        ("validation-batch", val, config["batch_size"]),
    ]
    filenames = ["data.npy", "label.npy"]
    if (Path(config["data_dir"]) / "new_test_data.npy").is_file():
        datasets.append(("test-single", SeismicDataset(config["data_dir"], "test"), 1))
        filenames.append("new_test_data.npy")
    cases = []
    for label, dataset, batch_size in datasets:
        batch_size = min(batch_size, len(dataset))

        def prepare():
            samples = [dataset[i] for i in range(batch_size)]
            return torch.stack([s[0] if isinstance(s, tuple) else s for s in samples]).to(device)

        x = prepare()
        captured = []
        handles = []
        for name, module in reference.named_modules():
            if isinstance(module, ReplaceableConv2d):

                def capture(layer, args, result, name=name):
                    captured.append((name, layer, args[0]))

                handles.append(module.register_forward_hook(capture))
        try:
            expected = reference(x)
        finally:
            for handle in handles:
                handle.remove()
        actual = custom(x)
        correctness = dict(
            logits_max_abs_error=compare(actual, expected),
            probability_max_abs_error=compare(actual.sigmoid(), expected.sigmoid()),
            mask_disagreements_at_04=int(
                ((actual.sigmoid() > 0.4) != (expected.sigmoid() > 0.4)).sum()
            ),
            atol=1e-5,
            rtol=1e-4,
        )
        operators = []
        for name, layer, input in captured:
            args = (
                input,
                layer.weight,
                layer.bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )

            def ref_fn(args=args):
                return BACKENDS["torch"](*args)

            def custom_fn(args=args):
                return BACKENDS[backend](*args)

            error = compare(custom_fn(), ref_fn())
            ref_time = timed(ref_fn, device, warmup, repeats, batch_size)
            custom_time = timed(custom_fn, device, warmup, repeats, batch_size)
            operators.append(
                dict(
                    name=name,
                    input_shape=list(input.shape),
                    weight_shape=list(layer.weight.shape),
                    stride=list(layer.stride),
                    padding=list(layer.padding),
                    dilation=list(layer.dilation),
                    groups=layer.groups,
                    bias=layer.bias is not None,
                    max_abs_error=error,
                    reference=ref_time,
                    custom=custom_time,
                    speedup=ref_time["median_ms"] / custom_time["median_ms"],
                )
            )
        # Release captured activations before whole-model timing.
        captured.clear()
        ref_time = timed(lambda: reference(x), device, warmup, repeats, batch_size)
        custom_time = timed(lambda: custom(x), device, warmup, repeats, batch_size)

        def pipeline(model):
            # Includes warm mmap access, validation, normalization, stacking, transfers,
            # inference, sigmoid, thresholding and host materialization; excludes cold I/O.
            return (model(prepare()).sigmoid() > 0.3).cpu().numpy()

        ref_pipeline = timed(lambda: pipeline(reference), device, warmup, repeats, batch_size)
        custom_pipeline = timed(lambda: pipeline(custom), device, warmup, repeats, batch_size)
        cases.append(
            dict(
                name=label,
                input_shape=list(x.shape),
                normalize=dataset.normalize,
                sample_indices=list(range(dataset.start, dataset.start + batch_size)),
                correctness=correctness,
                operators=operators,
                model=dict(
                    reference=ref_time,
                    custom=custom_time,
                    speedup=ref_time["median_ms"] / custom_time["median_ms"],
                ),
                pipeline=dict(
                    reference=ref_pipeline,
                    custom=custom_pipeline,
                    speedup=ref_pipeline["median_ms"] / custom_pipeline["median_ms"],
                ),
            )
        )
        print(
            f"Benchmark {label} {list(x.shape)}: torch={ref_time['median_ms']:.3f}ms "
            f"{backend}={custom_time['median_ms']:.3f}ms",
            flush=True,
        )
    report = dict(
        environment=metadata(device),
        checkpoint=str(checkpoint_path),
        epoch=checkpoint["epoch"],
        backend=backend,
        warmup=warmup,
        repeats=repeats,
        data=dict(
            data_dir=str(Path(config["data_dir"]).resolve()),
            split_index=config["split_index"],
            files=inspect_data(config["data_dir"], filenames),
        ),
        correctness_scope="Only listed sample_indices; full-validation metrics require evaluate",
        timing="host wall-clock with device synchronization; inference_mode",
        pipeline="warm mmap + preprocessing + transfers + model + sigmoid/mask + host output; "
        "excludes loader creation, cold disk I/O and file export",
        cases=cases,
    )
    write_json(output, report)
    return report
