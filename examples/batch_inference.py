"""Run inference over a folder of images and write predictions to CSV.

python examples/batch_inference.py path/to/images/ predictions.csv
"""

import sys
from pathlib import Path

from trinet.inference.predict import predict_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main(folder: str, out_csv: str = "predictions.csv") -> None:
    images = [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT]
    rows = ["image,diagnosis,confidence,mpox_label,mpox_prob"]
    for p in images:
        r = predict_image(p)
        m = r["mpox_screening"]
        rows.append(f"{p},{r['diagnosis']},{r['confidence']},{m['label']},{m['mpox_probability']}")
        print(f"{p.name:<40} -> {r['diagnosis']} ({100 * r['confidence']:.1f}%)")
    Path(out_csv).write_text("\n".join(rows))
    print(f"\nWrote {len(images)} predictions to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "predictions.csv")
