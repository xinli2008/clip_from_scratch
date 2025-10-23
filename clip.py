import torch 
import torch.nn as nn
import torch.nn.functional as F
from text_encoder import TextEncoder
from image_encoder import ImageEncoder

class Clip(nn.Module):
    def __init__(self, temperature=0.07):
        super(Clip, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        # 可学习的温度参数，初始值设为0.07（CLIP论文中的经验值）
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

    def forward(self, image, text):
        image_features = self.image_encoder(image)  # [b, 8]
        text_features = self.text_encoder(text)     # [b, 8]
        
        r"""
        在clip中, 会使用L2归一化将图像和文本的特征向量都归一化到单位长度.
        这样做的目的是为了计算余弦相似度，因为归一化后的点积等价于余弦相似度.
        """
        image_features = F.normalize(image_features, p=2, dim=1)  # [b, 8]
        text_features = F.normalize(text_features, p=2, dim=1)    # [b, 8]
        
        # 计算相似度矩阵并应用温度缩放
        # 归一化后的点积等价于余弦相似度
        logits = (image_features @ text_features.T) / self.temperature  # [b, b]
        return logits


if __name__ == "__main__":
    clip = Clip()
    image = torch.rand(5, 1, 64, 64)    # [b, 1, 64, 64]
    text = torch.randint(0, 10, (5,))   # [b]
    logits = clip(image, text)
    print(logits.shape)