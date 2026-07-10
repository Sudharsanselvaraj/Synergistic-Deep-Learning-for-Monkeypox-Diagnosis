"""Predict the disease class for a single lesion image.

python examples/predict_single.py path/to/lesion.jpg
"""

import sys

from trinet.inference.predict import predict_image

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "lesion.jpg"
    result = predict_image(image)
    print(f"Diagnosis      : {result['diagnosis']} ({100 * result['confidence']:.1f}%)")
    print(
        f"Mpox screening : {result['mpox_screening']['label']} "
        f"(p={result['mpox_screening']['mpox_probability']:.3f})"
    )
    for name, prob in result["top_k"]:
        print(f"  {name:<24} {100 * prob:5.1f}%")
