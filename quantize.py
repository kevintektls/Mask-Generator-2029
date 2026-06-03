import torch
import os
from train import UNet

model = UNet()
model.load_state_dict(torch.load("model/unet.pth", map_location="cpu"))
model.eval()

model_scripted = torch.jit.script(model)
model_scripted.save("model/unet_scripted.pt")

original = os.path.getsize("model/unet.pth") / 1e6
scripted = os.path.getsize("model/unet_scripted.pt") / 1e6
print(f"Original : {original:.1f} MB | Scripted : {scripted:.1f} MB")
