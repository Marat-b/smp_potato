import onnx
from onnxsim import simplify

onnx_model = onnx.load(r'..\weights\model_20220701_3class_noteval.onnx')
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, r'..\weights\model_20220701_3class_noteval_s.onnx')