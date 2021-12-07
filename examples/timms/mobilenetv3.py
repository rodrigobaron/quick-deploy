import timm
import torch


model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.save("mobilenetv3_large_100.pt")
