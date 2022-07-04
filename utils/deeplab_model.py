import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

model = torch.load('../weights/best_model_20220704_deeplabv3_resnet50_1x_3class.pth',  map_location=torch.device('cpu'))
input_image = Image.open('../images/image_256.jpg')
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
# with torch.no_grad():
#     # print(model)
#     # output = model(input_batch)['out'][0]
#     output = model(input_batch)[0]
# print(input_batch.shape)
# print(output.shape)

prediction = model(input_batch).squeeze(0).softmax(0)
print(f'prediction={prediction.shape}')
class_id = prediction.argmax().item()
print(f'prediction.argmax()={prediction.argmax()}')
print(f'class_id={class_id}')
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")