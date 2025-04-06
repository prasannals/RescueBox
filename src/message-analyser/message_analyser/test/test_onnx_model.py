import onnx

# Load the ONNX model
onnx_model_path="models/gemma-2b.onnx"

# Verify the model
onnx.checker.check_model(onnx_model_path)

print("ONNX model is valid!")