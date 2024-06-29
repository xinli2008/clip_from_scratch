import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings = 10, embedding_dim = 16)
        self.dense1 = nn.Linear(16, 64)
        self.dense2 = nn.Linear(64, 16)
        self.linear = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        r"""
        Perform TextEncoder forward process
        Args:
            x: torch.Tensor, [b]
        Return:
            torch.Tensor, [b, 8]
        """
        x = self.emb(x)   # [10, 16]
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.linear(x)
        x = self.norm(x)
        return x

if __name__ == "__main__":
    text_encoder = TextEncoder()
    x = torch.tensor([1,2,3,4,5,6,7,8,9,0])  # [10]
    y = text_encoder(x)
    print(y.shape)