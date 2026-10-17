"""
PyTorch examples: MLP, CNN, RNN (LSTM), and Transformer.
Each model is paired with a tiny synthetic task so the file runs end to end.

    pip install torch
    python pytorch_models.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- MLP
class MLP(nn.Module):
    """Fully connected network for flat feature vectors."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):                       # x: (B, in_dim)
        return self.net(x)


# ---------------------------------------------------------------- CNN
class CNN(nn.Module):
    """Small conv net for 1x28x28 images (MNIST-shaped)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 7x7
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):                       # x: (B, 1, 28, 28)
        return self.classifier(self.features(x).flatten(1))


# ---------------------------------------------------------------- RNN
class LSTMClassifier(nn.Module):
    """Embedding -> LSTM -> linear head, classifying a whole sequence."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, tokens):                  # tokens: (B, T)
        out, _ = self.lstm(self.embed(tokens))  # (B, T, 2H)
        return self.head(out.mean(dim=1))       # mean-pool over time


# ---------------------------------------------------------------- Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # x: (B, T, D)
        return x + self.pe[: x.size(1)]


class TransformerClassifier(nn.Module):
    """Encoder-only transformer using PyTorch's built-in layers."""

    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, tokens, pad_mask=None):   # pad_mask: (B, T), True = padding
        x = self.pos(self.embed(tokens) * math.sqrt(self.d_model))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x.mean(dim=1))


class SelfAttention(nn.Module):
    """From-scratch multi-head self-attention, for reference."""

    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead, self.dk = nhead, d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):         # x: (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nhead, self.dk).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # (B, H, T, dk)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


# ---------------------------------------------------------------- Training utils
def train(model, loader, epochs=5, lr=1e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f"  epoch {epoch + 1}: loss {loss_sum / total:.3f}  acc {correct / total:.2%}")


def loader_from(x, y, bs=64):
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True)


# ---------------------------------------------------------------- Demo
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 1024

    print("MLP: classify whether the sum of 20 features is positive")
    x = torch.randn(N, 20)
    y = (x.sum(1) > 0).long()
    train(MLP(20, 64, 2), loader_from(x, y))

    print("CNN: classify whether the left or right half of an image is brighter")
    x = torch.randn(N, 1, 28, 28)
    y = (x[..., :14].mean((1, 2, 3)) > x[..., 14:].mean((1, 2, 3))).long()
    x[..., :14] += y.view(-1, 1, 1, 1).float() * 0.5       # make it learnable
    train(CNN(2), loader_from(x, y))

    print("LSTM: does token 1 appear more often than token 2?")
    x = torch.randint(0, 10, (N, 30))
    y = ((x == 1).sum(1) > (x == 2).sum(1)).long()
    train(LSTMClassifier(10, 16, 32, 2), loader_from(x, y), epochs=8)

    print("Transformer: same task as the LSTM")
    train(TransformerClassifier(10, d_model=32, nhead=4, num_layers=2),
          loader_from(x, y), epochs=8)

    # shape check of the from-scratch attention block
    print("SelfAttention output:", SelfAttention(32, 4)(torch.randn(2, 10, 32), causal=True).shape)