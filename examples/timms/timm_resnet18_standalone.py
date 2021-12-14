import timm
import torch


model = timm.create_model('resnet18', pretrained=True)
x = torch.randn(1, 3, 224, 224)

