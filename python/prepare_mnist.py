#!/usr/bin/env python3
"""
Baixa o MNIST original (formato IDX) e converte para um binario raw
simples que o loader em C++ (mnist.cpp) le com um unico fread.

Formato de saida (train.bin / test.bin):
    uint32 n_samples
    uint32 rows
    uint32 cols
    float32[n_samples * rows * cols]   imagens, normalizadas em [0,1]
    uint8[n_samples]                   labels
"""
import array
import gzip
import struct
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master"
DATA_DIR = Path(__file__).parent.parent / "data"

SPLITS = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def download(name: str) -> Path:
    gz_path = DATA_DIR / f"{name}.gz"
    if not gz_path.exists():
        print(f"baixando {name}.gz...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.gz", gz_path)
    return gz_path


def read_idx_images(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"magic number invalido em {gz_path}")
        data = f.read(n * rows * cols)
        return data, n, rows, cols


def read_idx_labels(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"magic number invalido em {gz_path}")
        return f.read(n), n


def main():
    DATA_DIR.mkdir(exist_ok=True)

    for split, (images_name, labels_name) in SPLITS.items():
        images_gz = download(images_name)
        labels_gz = download(labels_name)

        pixels, n_img, rows, cols = read_idx_images(images_gz)
        labels, n_lab = read_idx_labels(labels_gz)
        if n_img != n_lab:
            raise ValueError(f"{split}: numero de imagens e labels difere")

        out_path = DATA_DIR / f"{split}.bin"
        with open(out_path, "wb") as out:
            out.write(struct.pack("<III", n_img, rows, cols))
            # normaliza uint8 [0,255] -> float32 [0,1]
            floats = array.array("f", (px / 255.0 for px in pixels))
            out.write(floats.tobytes())
            out.write(labels)

        print(f"{split}: {n_img} amostras {rows}x{cols} -> {out_path}")

    print("MNIST pronto em ./data (train.bin, test.bin)")


if __name__ == "__main__":
    main()
