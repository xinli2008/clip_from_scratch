import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MNIST
from clip import Clip

# load model and dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Clip()
model.load_state_dict(torch.load("/mnt/clip_from_scratch/assets/model_step_2000.pt"))
model = model.to(device)
dataset = MNIST()

# eval
model.eval()

# inference
image, label = dataset[1]
print("正确分类:", label)

targets = torch.arange(0, 10)  # 10种分类
logits = model(image.unsqueeze(0).to(device), targets.to(device))  # 1张图片 vs 10种分类
print("CLIP分类:", logits.argmax(-1).item())
