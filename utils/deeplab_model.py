import torch
from PIL import Image
from torchvision import transforms

model = torch.load('../weights/best_model_20220701_deeplabv3_resnet_1x_3class.pth',  map_location=torch.device('cpu'))
input_image = Image.open('../images/image_256.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
with torch.no_grad():
    # print(model)
    # output = model(input_batch)['out'][0]
    output = model(input_batch)[0]


print(input_batch.shape)
print(output.shape)