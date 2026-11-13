"""Validate uploaded arrays before read-only mmap; preserve float64 normalization."""

import hashlib
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PUBLIC_ARRAYS = {
    "data.npy": (
        209715328,
        "acc169d7bbea351262eeb3d5d804a1b65aac2072d6425e72fb7318450a3b8233",
    ),
    "label.npy": (
        209715328,
        "0622b117dfa7176c98cd2d1bb600646dd4a10d67f40ba38112aca96160e2ab34",
    ),
}


def array_path(path: Path) -> Path:
    """Recover only a complete official upload, never overwrite an existing array."""
    temporary = path.with_name(path.name + ".efilePart")
    if not path.exists() and temporary.exists():
        expected = PUBLIC_ARRAYS.get(path.name)
        if temporary.is_symlink() or not temporary.is_file() or expected is None:
            raise ValueError(f"无法校验临时文件 {temporary}；请等待上传完成后确认文件名")
        size, sha256 = expected
        if temporary.stat().st_size != size:
            raise ValueError(f"上传文件校验失败：{temporary} 大小不符，请完成或重新上传")
        # Snapshot the upload: the uploader may still modify its temporary file.
        # Keep that source untouched; publish only our independently verified copy.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".array-recovery-") as snapshot:
            digest, copied = hashlib.sha256(), 0
            with temporary.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    copied += len(chunk)
                    if copied > size:
                        raise ValueError(f"上传文件校验失败：{temporary} 仍在变化，请完成上传")
                    snapshot.write(chunk)
                    digest.update(chunk)
            snapshot.flush()
            if copied != size or digest.hexdigest() != sha256:
                raise ValueError(f"上传文件校验失败：{temporary} 内容不符，请重新上传")
            # Hard-link creation is atomic and never overwrites an existing target.
            try:
                path.hardlink_to(snapshot.name)
            except FileExistsError:
                pass
            except OSError as error:
                raise OSError(f"无法恢复 {path}；请确认上传完成后手动校验并改名") from error
            else:
                warnings.warn(f"已校验并恢复上传文件：{path}", RuntimeWarning, stacklevel=2)
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件 {path}；请上传完整代码包并解压")
    with path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{path} 是LFS指针；请下载发布页的完整代码包并上传解压")
    if not header.startswith(b"\x93NUMPY"):
        raise ValueError(f"{path} 不是有效的NumPy数组；请检查上传或重新解压完整代码包")
    return path


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if not np.isfinite(image).all():
        raise ValueError("Input must contain finite values")
    std = image.std()
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    return np.asarray((image - image.mean()) / std, dtype=np.float32)


class SeismicDataset(Dataset):
    def __init__(self, data_dir, split, split_index=1600, normalize_test=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.normalize = split != "test" or normalize_test
        path = Path(data_dir)
        self.data = np.load(
            array_path(path / ("new_test_data.npy" if split == "test" else "data.npy")),
            mmap_mode="r",
            allow_pickle=False,
        )
        if self.data.ndim != 3 or any(s < 8 or s % 8 for s in self.data.shape[-2:]):
            raise ValueError("Images must be NHW with H/W divisible by 8")
        self.labels = None
        if split == "test":
            self.start, self.stop = 0, len(self.data)
        else:
            self.labels = np.load(array_path(path / "label.npy"), mmap_mode="r", allow_pickle=False)
            if self.labels.shape != self.data.shape:
                raise ValueError("Label and image shapes must match")
            if not 0 < split_index < len(self.data):
                raise ValueError("split_index must leave nonempty train and validation sets")
            self.start, self.stop = (
                (0, split_index) if split == "train" else (split_index, len(self.data))
            )

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        index += self.start
        image = self.data[index]
        if self.normalize:
            image = normalize_image(image)
        else:
            if not np.isfinite(image).all():
                raise ValueError("Input must contain finite values")
            image = np.array(image, dtype=np.float32, copy=True)
        x = torch.from_numpy(image[None])
        if self.labels is None:
            return x
        label = np.array(self.labels[index], dtype=np.float32, copy=True)
        if not np.isin(label, (0, 1)).all():
            raise ValueError("Labels must be binary 0/1")
        return x, torch.from_numpy(label[None])


def inspect_data(data_dir, filenames=("data.npy", "label.npy", "new_test_data.npy")):
    result = {}
    for name in filenames:
        path = array_path(Path(data_dir) / name)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lo, hi, total, foreground = float("inf"), -float("inf"), 0.0, 0
        for chunk in values:
            if not np.isfinite(chunk).all():
                raise ValueError(f"{name} contains nonfinite values")
            lo, hi = min(lo, float(chunk.min())), max(hi, float(chunk.max()))
            total += float(chunk.sum(dtype=np.float64))
            if name == "label.npy":
                if not np.isin(chunk, (0, 1)).all():
                    raise ValueError("Labels must be binary 0/1")
                foreground += int(np.count_nonzero(chunk))
        result[name] = dict(
            shape=list(values.shape),
            dtype=str(values.dtype),
            min=lo,
            max=hi,
            mean=total / values.size,
            sha256=digest.hexdigest(),
        )
        if name == "label.npy":
            result[name].update(
                foreground_pixels=foreground, foreground_fraction=foreground / values.size
            )
    return result


def data_provenance(dataset, batch_size):
    """Identify the effective input files, sample range and preprocessing for a result."""
    return dict(
        data_dir=str(Path(dataset.data.filename).resolve().parent),
        split=dataset.split,
        sample_range=[dataset.start, dataset.stop],
        batch_size=batch_size,
        normalization="per-image-zscore" if dataset.normalize else "none",
        files=inspect_data(
            Path(dataset.data.filename).parent,
            ("new_test_data.npy",) if dataset.split == "test" else ("data.npy", "label.npy"),
        ),
    )
