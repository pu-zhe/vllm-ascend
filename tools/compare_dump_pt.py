import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Compare dumped .pt files or dump directories.")
    parser.add_argument("left", help="left .pt file or directory")
    parser.add_argument("right", help="right .pt file or directory")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    left = Path(args.left)
    right = Path(args.right)

    if left.is_file() and right.is_file():
        compare_file(left, right, args.rtol, args.atol)
        return

    if left.is_dir() and right.is_dir():
        compare_dir(left, right, args.rtol, args.atol)
        return

    raise SystemExit("left/right must both be files or both be directories")


def compare_dir(left_dir: Path, right_dir: Path, rtol: float, atol: float):
    left_files = {p.relative_to(left_dir).as_posix(): p for p in left_dir.rglob("*.pt")}
    right_files = {p.relative_to(right_dir).as_posix(): p for p in right_dir.rglob("*.pt")}

    only_left = sorted(set(left_files) - set(right_files))
    only_right = sorted(set(right_files) - set(left_files))
    common = sorted(set(left_files) & set(right_files))

    if only_left:
        print("only in left:")
        for name in only_left:
            print(name)
    if only_right:
        print("only in right:")
        for name in only_right:
            print(name)

    for name in common:
        print(f"\n=== {name} ===")
        compare_file(left_files[name], right_files[name], rtol, atol)


def compare_file(left_file: Path, right_file: Path, rtol: float, atol: float):
    left_obj = torch.load(left_file, map_location="cpu")
    right_obj = torch.load(right_file, map_location="cpu")
    compare_obj(left_obj, right_obj, "root", rtol, atol)


def compare_obj(left_obj, right_obj, name: str, rtol: float, atol: float):
    if torch.is_tensor(left_obj) and torch.is_tensor(right_obj):
        compare_tensor(left_obj, right_obj, name, rtol, atol)
        return

    if isinstance(left_obj, dict) and isinstance(right_obj, dict):
        left_keys = set(left_obj)
        right_keys = set(right_obj)
        for key in sorted(left_keys - right_keys):
            print(f"{name}.{key}: only in left")
        for key in sorted(right_keys - left_keys):
            print(f"{name}.{key}: only in right")
        for key in sorted(left_keys & right_keys):
            compare_obj(left_obj[key], right_obj[key], f"{name}.{key}", rtol, atol)
        return

    if isinstance(left_obj, (list, tuple)) and isinstance(right_obj, (list, tuple)):
        if len(left_obj) != len(right_obj):
            print(f"{name}: length mismatch left={len(left_obj)} right={len(right_obj)}")
            return
        for idx, (left_item, right_item) in enumerate(zip(left_obj, right_obj)):
            compare_obj(left_item, right_item, f"{name}[{idx}]", rtol, atol)
        return

    if left_obj != right_obj:
        print(f"{name}: value mismatch left={left_obj} right={right_obj}")
    else:
        print(f"{name}: equal")


def compare_tensor(left_tensor: torch.Tensor, right_tensor: torch.Tensor, name: str, rtol: float, atol: float):
    if left_tensor.shape != right_tensor.shape:
        print(f"{name}: shape mismatch left={tuple(left_tensor.shape)} right={tuple(right_tensor.shape)}")
        return
    if left_tensor.dtype != right_tensor.dtype:
        print(f"{name}: dtype mismatch left={left_tensor.dtype} right={right_tensor.dtype}")

    left_tensor = left_tensor.float()
    right_tensor = right_tensor.float()
    diff = (left_tensor - right_tensor).abs()
    max_abs = diff.max().item() if diff.numel() > 0 else 0.0
    mean_abs = diff.mean().item() if diff.numel() > 0 else 0.0
    same = torch.allclose(left_tensor, right_tensor, rtol=rtol, atol=atol)
    cosine = (
        F.cosine_similarity(left_tensor.flatten().unsqueeze(0), right_tensor.flatten().unsqueeze(0)).item()
        if diff.numel() > 0
        else 1.0
    )
    print(
        f"{name}: shape={tuple(left_tensor.shape)} "
        f"allclose={same} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} cosine={cosine:.6g}"
    )


if __name__ == "__main__":
    main()
