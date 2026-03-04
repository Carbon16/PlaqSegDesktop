#!/usr/bin/env python3
"""
Export PlaqSeg PyTorch models to ONNX format for use with the Rust ort crate.

Requirements:
    pip install ultralytics

Usage:
    python scripts/export_models.py
    python scripts/export_models.py --models-dir path/to/models
"""
import argparse
from pathlib import Path


def export_model(pt_path: Path, out_path: Path) -> None:
    from ultralytics import YOLO
    print(f"  Exporting {pt_path.name} → {out_path.name} ...")
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        nms=True,       # bake NMS into graph — produces (1, 300, 38) output
        # dynamic=False, # fixed batch size = 1
    )
    # Ultralytics saves next to the .pt file; move to desired location
    generated = pt_path.with_suffix(".onnx")
    if generated.exists() and generated != out_path:
        generated.rename(out_path)
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  ✅ {out_path.name}  ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Export failed — {out_path} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pt models to .onnx")
    parser.add_argument(
        "--models-dir", "-d",
        default=str(Path(__file__).parent.parent / "models"),
        help="Directory containing .pt model files (default: models/)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    print(f"🔬 PlaqSeg model export")
    print(f"   Models dir: {models_dir}")
    print()

    pt_files = sorted(models_dir.glob("*.pt"))
    if not pt_files:
        print("  ⚠ No .pt files found — nothing to export")
        return

    for pt in pt_files:
        out = models_dir / (pt.stem + ".onnx")
        if out.exists():
            size_mb = out.stat().st_size / 1_048_576
            print(f"  ⏭  Skipping {pt.name} — {out.name} already exists ({size_mb:.1f} MB)")
            continue
        export_model(pt, out)

    print()
    print("✅ Done. Copy the generated .onnx files into your models folder.")


if __name__ == "__main__":
    main()
