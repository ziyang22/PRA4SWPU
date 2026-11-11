"""Fixed participant entry. Replace model and inference while retaining this signature."""

from pathlib import Path

BATCH_SIZE = 16
SEED = 12345


def main(input_path: str, output_path: str, device: str) -> None:
    import random

    import numpy as np
    import torch

    from data import normalize_image
    from utils import load_checkpoint

    # Read all input only inside the timed process; labels are never supplied.
    images = np.load(input_path, allow_pickle=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model, _ = load_checkpoint(Path(__file__).parent / "weights/model.pt", "torch", device)
    masks = np.empty(images.shape, dtype=np.uint8)
    with torch.inference_mode():
        for offset in range(0, len(images), BATCH_SIZE):
            batch = images[offset : offset + BATCH_SIZE]
            x = np.stack([normalize_image(image) for image in batch])[:, None]
            probabilities = model(torch.from_numpy(x).to(device)).sigmoid()
            masks[offset : offset + len(batch)] = (
                (probabilities[:, 0] > 0.5).to("cpu", dtype=torch.uint8).numpy()
            )
    np.save(output_path, masks, allow_pickle=False)
