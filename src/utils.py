"""Metrics, eager plotting imports, device selection and checkpoint helpers."""

import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Invalid device: {name}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metadata(device):
    hardware = platform.processor()
    if platform.system() == "Darwin":
        hardware = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    return dict(
        timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        platform=platform.platform(),
        hardware=hardware,
        device=str(device),
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        threads=torch.get_num_threads(),
        dtype="float32",
        layout="NCHW",
        cross_device_bitwise_reproducibility=False,
    )


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_checkpoint(path, backend, device):
    from model import UNet

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(backend=backend)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


class BinaryMetrics:
    def __init__(self, thresholds=(0.5, 0.4)):
        self.counts = {str(t): [0, 0, 0, 0] for t in thresholds}

    def update(self, probabilities, labels):
        target = labels.bool()
        for threshold, counts in self.counts.items():
            pred = probabilities > float(threshold)
            values = (
                torch.stack(
                    (
                        (pred & target).sum(),
                        (pred & ~target).sum(),
                        (~pred & target).sum(),
                        (~pred & ~target).sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            for i, value in enumerate(values):
                counts[i] += value

    def compute(self):
        results = {}
        for threshold, (tp, fp, fn, tn) in self.counts.items():
            if tp + fp + fn + tn == 0:
                raise ValueError("Cannot compute metrics without samples")
            results[threshold] = dict(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                accuracy=(tp + tn) / (tp + fp + fn + tn),
                dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                foreground_fraction=(tp + fn) / (tp + fp + fn + tn),
            )
        return results


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected binary masks (0/1), not probabilities")
    return mask.astype(bool)


def prediction_error_rgb(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    """White=TP, black=TN, red=FP or FN. Never modifies the input masks."""
    prediction, label = _binary_mask(prediction), _binary_mask(label)
    if prediction.ndim != 2 or prediction.shape != label.shape:
        raise ValueError("Prediction and label must have the same HW shape")
    rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    rgb[prediction & label] = (255, 255, 255)
    rgb[prediction ^ label] = (255, 0, 0)
    return rgb


def plot_prediction_grid(
    images: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    indices: list[int] | None = None,
) -> None:
    """Plot selected NHW input/label/prediction-error triples, with exact pixel counts."""
    images, predictions, labels = map(np.asarray, (images, predictions, labels))
    if images.ndim != 3 or images.shape != predictions.shape or images.shape != labels.shape:
        raise ValueError("Images, predictions and labels must have the same NHW shape")
    if not len(images) or not np.isfinite(images).all():
        raise ValueError("Nonempty finite images required")
    if indices is None:
        indices = list(range(len(images)))
    if len(indices) != len(images):
        raise ValueError("One display index required per image")
    predictions, labels = _binary_mask(predictions), _binary_mask(labels)
    fig, axes = plt.subplots(len(images), 3, figsize=(10, 3 * len(images)), squeeze=False)
    try:
        for row, (image, prediction, label, index) in enumerate(
            zip(images, predictions, labels, indices, strict=True)
        ):
            tp = int(np.count_nonzero(prediction & label))
            fp = int(np.count_nonzero(prediction & ~label))
            fn = int(np.count_nonzero(~prediction & label))
            denominator = 2 * tp + fp + fn
            dice = 2 * tp / denominator if denominator else 1.0
            axes[row, 0].imshow(image, cmap="gray", interpolation="nearest")
            axes[row, 1].imshow(label, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[row, 2].imshow(prediction_error_rgb(prediction, label), interpolation="nearest")
            axes[row, 0].set_title(f"Input #{index}")
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].set_title(f"Prediction: Dice={dice:.3f}\nFP={fp}, FN={fn}")
            for axis in axes[row]:
                axis.axis("off")
        fig.suptitle("White: correct foreground | Red: false positive OR missed fault", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


@torch.inference_mode()
def plot_predictions(model, dataset, device, path, threshold=0.4):
    indices = sorted(set((0, len(dataset) // 2, len(dataset) - 1)))
    samples = [dataset[index] for index in indices]
    if all(isinstance(sample, tuple) for sample in samples):
        images, predictions, labels = [], [], []
        for x, y in samples:
            pred = model(x[None].to(device)).sigmoid().cpu().numpy()[0, 0]
            images.append(x.numpy()[0])
            predictions.append(pred > threshold)
            labels.append(y.numpy()[0])
        plot_prediction_grid(
            np.stack(images), np.stack(predictions), np.stack(labels), path, indices
        )
        return
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        sample = dataset[index]
        x, y = sample if isinstance(sample, tuple) else (sample, None)
        pred = model(x[None].to(device)).sigmoid().cpu().numpy()[0, 0]
        images = [x.numpy()[0], pred > threshold, y.numpy()[0] if y is not None else pred]
        titles = [
            f"Input #{index}",
            f"Mask (>{threshold})",
            "Ground truth" if y is not None else "Probability",
        ]
        for ax, image, title in zip(axes[row], images, titles, strict=True):
            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_bce"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation"]["bce"] for row in history], label="val")
    axes[0].set_title("BCE")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation"]["metrics"]["0.5"]["accuracy"] for row in history])
    axes[1].set_title("Validation accuracy (>0.5)")
    for metric in ("dice", "iou", "recall"):
        axes[2].plot(
            epochs, [row["validation"]["metrics"]["0.4"][metric] for row in history], label=metric
        )
    axes[2].set_title("Validation foreground (>0.4)")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
