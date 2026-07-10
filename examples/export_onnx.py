"""Export the Tri-Net v2 champion to SavedModel / ONNX for deployment.

pip install tf2onnx
python examples/export_onnx.py
"""

from trinet.inference.export import export

if __name__ == "__main__":
    out = export(fmt="onnx")
    print(f"Exported to {out}")
