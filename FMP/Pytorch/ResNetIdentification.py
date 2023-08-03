from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.495, 0.456, 0.406],
        std=[0.232, 0.224, 0.225]
    )

])

zhouye = Image.open("/Users/zcq30/Documents/XJLTU Learning/XJTLU/FMP/Pytorch/Coco_umbrella.jpeg")
zhouye_tensor = preprocess(zhouye)
# batch_Coconut_t = preprocess(zhouye)
batch_Coconut_t = torch.unsqueeze(zhouye_tensor, 0)

resnet = models.resnet101(pretrained=True)
resnet.eval()
result = resnet(batch_Coconut_t)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100
num_index = result.argmax()

print("将椰子预测的编号为： ", result.argmax())
print("将椰子预测的东西为： ", labels[num_index])
print("预测的正确率为： ", percentage[num_index].item())

print("正确率前五的东西为：")
_, indices = torch.sort(result, descending=True)
for idx in indices[0][:5]:
    print(labels[idx])
# print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]],'\n')
