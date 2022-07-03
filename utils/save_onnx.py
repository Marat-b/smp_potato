import torch
import onnx

batch_size = 10
# set the model to inference mode
torch_model = torch.load('../weights/best_model_20220701_deeplabv3_resnet_1x_3class.pth')
torch_model.eval()
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=13,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # variable length axes
        'output': {0: 'batch_size'}
    }
    )

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)
