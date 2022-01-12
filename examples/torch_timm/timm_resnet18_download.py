import timm
import torch

# load pretrained model and save
model = timm.create_model('resnet18', pretrained=True)
torch.save(model, 'resnet18.pt')
