import torch

model = torch.load('../weights/best_model_20220701_deeplabv3_resnet_1x_3class.pth', map_location=torch.device('cpu'))
model.eval()

model_scripted = torch.jit.script(model)
torch.jit.save(model_scripted, '../weights/best_model_20220701_deeplabv3_resnet_1x_3class_scripted.pth')
