"""Training, validation, prediction and plots without notebook dependencies."""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SeismicDataset, data_provenance, inspect_data
from model import UNet
from utils import (
    BinaryMetrics,
    load_checkpoint,
    metadata,
    plot_history,
    plot_predictions,
    resolve_device,
    seed_everything,
    synchronize,
    write_json,
)


def loader(config, split, batch_size=None, normalize_test=False):
    dataset = SeismicDataset(config["data_dir"], split, config["split_index"], normalize_test)
    return DataLoader(
        dataset,
        batch_size=batch_size or config["batch_size"],
        shuffle=split == "train",
        num_workers=0,
    )


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    metrics = BinaryMetrics()
    total_loss, pixels = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
        pixels += y.numel()
        metrics.update(logits.sigmoid(), y)
    return dict(bce=total_loss / pixels, metrics=metrics.compute())


def train(config):
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["threads"] < 1:
        raise ValueError("epochs, batch_size and threads must be positive")
    if config["learning_rate"] <= 0 or config["adam_epsilon"] <= 0:
        raise ValueError("Optimizer parameters must be positive")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists():
        raise FileExistsError(f"{output} already contains a run; choose another output_dir")
    seed_everything(config["seed"], config["threads"])
    device = resolve_device(config["device"])
    environment = metadata(device)
    data_summary = inspect_data(config["data_dir"], filenames=("data.npy", "label.npy"))
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment)
    write_json(output / "data.json", data_summary)
    training, validation = loader(config, "train"), loader(config, "val")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=config["adam_epsilon"],
        weight_decay=0,
    )
    history, best = [], float("inf")
    print(
        f"Training on {device}: {len(training.dataset)} train / "
        f"{len(validation.dataset)} val; {config['epochs']} epochs",
        flush=True,
    )
    for epoch in range(1, config["epochs"] + 1):
        synchronize(device)
        started = time.perf_counter()
        model.train()
        total_loss, correct, pixels = 0.0, 0, 0
        for x, y in training:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            correct += ((logits.detach() > 0) == y.bool()).sum().item()
            pixels += y.numel()
        validation_result = evaluate(model, validation, device)
        synchronize(device)
        row = dict(
            epoch=epoch,
            train_bce=total_loss / pixels,
            train_accuracy=correct / pixels,
            validation=validation_result,
            seconds=time.perf_counter() - started,
        )
        history.append(row)
        checkpoint = dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            config=config,
            validation=validation_result,
            environment=environment,
        )
        torch.save(checkpoint, output / "last.pt")
        if validation_result["bce"] < best:
            best = validation_result["bce"]
            torch.save(checkpoint, output / "best.pt")
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch:02d}/{config['epochs']}: train BCE={row['train_bce']:.6f} "
            f"val BCE={validation_result['bce']:.6f} "
            f"acc={validation_result['metrics']['0.5']['accuracy']:.6f} "
            f"Dice@0.4={validation_result['metrics']['0.4']['dice']:.6f} "
            f"{row['seconds']:.2f}s",
            flush=True,
        )
    plot_history(history, output / "training-curves.png")
    plot_predictions(model, validation.dataset, device, output / "validation-examples.png")
    write_json(
        output / "summary.json",
        dict(
            config=config,
            environment=environment,
            parameters=sum(p.numel() for p in model.parameters()),
            epoch=config["epochs"],
            validation=history[-1]["validation"],
            total_epoch_seconds=sum(row["seconds"] for row in history),
            best_val_bce=best,
        ),
    )
    return history


def evaluate_checkpoint(
    checkpoint_path, backend, device_name, output, data_dir=None, batch_size=None
):
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, backend, device)
    config = dict(checkpoint["config"])
    if data_dir:
        config["data_dir"] = data_dir
    seed_everything(config["seed"], config["threads"])
    batches = loader(config, "val", batch_size)
    result = evaluate(model, batches, device)
    result.update(
        checkpoint=str(checkpoint_path),
        epoch=checkpoint["epoch"],
        backend=backend,
        environment=metadata(device),
        data=data_provenance(batches.dataset, batches.batch_size),
    )
    output = Path(output)
    write_json(output / "metrics.json", result)
    plot_predictions(model, batches.dataset, device, output / "validation-examples.png")
    print(result, flush=True)
    return result


@torch.inference_mode()
def predict(
    checkpoint_path,
    backend,
    device_name,
    output,
    data_dir=None,
    batch_size=None,
    normalize_test=False,
    threshold=0.3,
    export_csv=False,
):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, backend, device)
    config = dict(checkpoint["config"])
    if data_dir:
        config["data_dir"] = data_dir
    seed_everything(config["seed"], config["threads"])
    batches = loader(config, "test", batch_size, normalize_test)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    masks = np.lib.format.open_memmap(
        output / "masks.npy", mode="w+", dtype=np.uint8, shape=batches.dataset.data.shape
    )
    position = 0
    for x in batches:
        result = (model(x.to(device)).sigmoid() > threshold).cpu().numpy()[:, 0]
        masks[position : position + len(x)] = result
        position += len(x)
    masks.flush()
    if export_csv:
        with (output / "submission.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "value"])
            writer.writerows(enumerate(masks.reshape(-1)))
    write_json(
        output / "prediction.json",
        dict(
            checkpoint=str(checkpoint_path),
            epoch=checkpoint["epoch"],
            backend=backend,
            threshold=threshold,
            normalize_test=normalize_test,
            shape=list(masks.shape),
            samples=position,
            environment=metadata(device),
            data=data_provenance(batches.dataset, batches.batch_size),
            accuracy=None,
            reason="No test labels provided",
        ),
    )
    plot_predictions(model, batches.dataset, device, output / "test-examples.png", threshold)
    print(f"Saved {position} test masks to {output}; no test accuracy available", flush=True)
