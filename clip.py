import torch 
import torch.nn as nn
import torch.nn.functional as F
from text_encoder import TextEncoder
from image_encoder import ImageEncoder

class Clip(nn.Module):
    def __init__(self):
        super(Clip, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, image, text):
        image = self.image_encoder(image)  # [b, 8]
        text  = self.text_encoder(text)    # [b, 8]
        logits = image@text.T              # [b, b]
        return logits


if __name__ == "__main__":
    clip = Clip()
    image = torch.rand(5, 1, 64, 64)    # [b, 1, 64, 64]
    text = torch.randint(0, 10, (5,))   # [b]
    logits = clip(image, text)
    print(logits.shape)