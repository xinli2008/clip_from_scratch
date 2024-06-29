import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0, stride = stride)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(x1))
        x2 = self.conv3(x)
        return F.relu(x1 + x2)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.res_block1 = ResidualBlock(1, 16, 2)
        self.res_block2 = ResidualBlock(16, 4, 2)
        self.res_block3 = ResidualBlock(4, 1, 2)
        self.linear = nn.Linear(64, 8)
        self.norm = nn.LayerNorm(8)
        
    def forward(self, x):
        r"""
        Perform ImageEncoder forward process
        Args:
            x: [b, c, h, w]
        Return:
            torch.Tensor, []
        """
        x = self.res_block1(x)  # [b, 16, h//2, w//2]
        x = self.res_block2(x)  # [b, 4,  h//4, w//4]
        x = self.res_block3(x)  # [b, 1,  h//8, w//8]

        x = self.norm(self.linear(x.view(x.shape[0], -1)))

        return x

if __name__ == "__main__":
    image_encoder = ImageEncoder()
    out = image_encoder(torch.randn(3,1,64,64))
    print(out.shape)