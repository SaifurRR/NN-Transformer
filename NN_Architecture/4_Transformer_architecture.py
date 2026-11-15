"""
The full encoder-decoder Transformer ("Attention Is All You Need", Vaswani et al. 2017),
built from scratch in PyTorch. Every block is written out by hand: no nn.Transformer*.

    Input tokens ──► Embedding + Positional Encoding
                         │
                 ┌───────▼────────┐  x N
                 │ Encoder layer  │  self-attention -> feed-forward
                 └───────┬────────┘
                         │ memory
    Target tokens ──► Embedding + PE      │
                 ┌───────▼────────┐       │
                 │ Decoder layer  │◄──────┘  x N
                 │ masked self-attn -> cross-attn -> feed-forward
                 └───────┬────────┘
                         ▼
                  Linear -> softmax over vocab

Demo task: reverse a sequence of digits of variable length (exercises padding + causal masks).

    pip install torch
    python 2_Transformer_architecture.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

PAD, BOS, EOS = 0, 1, 2                         # special tokens; digits use ids 3..12

# ---------------------------------------------------------------- Building blocks
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding: PE(pos, 2i) = sin(pos / 10000^(2i/D)), cos for odd dims."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # x: (B, T, D)
        return self.drop(x + self.pe[: x.size(1)])


class MultiHeadAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, run over h heads in parallel.

    Used three ways: encoder self-attention, decoder masked self-attention,
    and decoder cross-attention (queries from decoder, keys/values from encoder).
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead, self.dk = nhead, d_model // nhead
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def split(self, x):                         # (B, T, D) -> (B, H, T, dk)
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.dk).transpose(1, 2)

    def forward(self, q, k, v, mask=None):      # mask broadcasts to (B, H, Tq, Tk), True = keep
        q, k, v = self.split(self.wq(q)), self.split(self.wk(k)), self.split(self.wv(v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)       # (B, H, Tq, Tk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.drop(scores.softmax(dim=-1))
        y = attn @ v                                                # (B, H, Tq, dk)
        B, H, T, dk = y.shape
        return self.wo(y.transpose(1, 2).reshape(B, T, H * dk))     # concat heads


class FeedForward(nn.Module):
    """Position-wise FFN: applied to every token independently."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------- Encoder / Decoder layers
# Pre-LN (x + Sublayer(LayerNorm(x))) trains more stably without LR warmup.
# The original paper used Post-LN: LayerNorm(x + Sublayer(x)).
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, src_mask))
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, tgt_mask))             # can't see the future
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, src_mask))  # look at the source
        return x + self.drop(self.ff(self.norm3(x)))

# ---------------------------------------------------------------- Full model
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, nhead=4, num_layers=3,
                 d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.enc_layers = nn.ModuleList(EncoderLayer(d_model, nhead, d_ff, dropout)
                                        for _ in range(num_layers))
        self.dec_layers = nn.ModuleList(DecoderLayer(d_model, nhead, d_ff, dropout)
                                        for _ in range(num_layers))
        self.enc_norm, self.dec_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # masks are boolean, True = position may be attended to
    @staticmethod
    def src_mask(src):                          # (B, S) -> (B, 1, 1, S)
        return (src != PAD)[:, None, None, :]

    @staticmethod
    def tgt_mask(tgt):                          # (B, T) -> (B, 1, T, T): padding AND causal
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
        return (tgt != PAD)[:, None, None, :] & causal

    def encode(self, src, src_mask):
        x = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.generator(self.dec_norm(x))  # logits: (B, T, tgt_vocab)

    def forward(self, src, tgt):
        src_mask = self.src_mask(src)
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, self.tgt_mask(tgt))

    @torch.no_grad()
    def greedy_decode(self, src, max_len):
        """Autoregressive generation: feed predictions back in one token at a time."""
        src_mask = self.src_mask(src)
        memory = self.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_mask, self.tgt_mask(ys))[:, -1]
            nxt = logits.argmax(-1).masked_fill(done, PAD)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if done.all():
                break
        return ys

# ---------------------------------------------------------------- Data
def make_batch(n, vocab, min_len=4, max_len=10):
    """src = random digits (padded); tgt = BOS + reversed(src) + EOS (padded)."""
    src = torch.full((n, max_len), PAD, dtype=torch.long)
    tgt = torch.full((n, max_len + 2), PAD, dtype=torch.long)
    for i in range(n):
        L = torch.randint(min_len, max_len + 1, ()).item()
        seq = torch.randint(3, vocab, (L,))
        src[i, :L] = seq
        tgt[i, : L + 2] = torch.cat([torch.tensor([BOS]), seq.flip(0), torch.tensor([EOS])])
    return src.to(device), tgt.to(device)

# ---------------------------------------------------------------- Train / evaluate
def train(model, vocab, steps=2000, bs=64, lr=5e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    for step in range(1, steps + 1):
        model.train()
        src, tgt = make_batch(bs, vocab)
        dec_in, labels = tgt[:, :-1], tgt[:, 1:]                 # teacher forcing, shift by one
        logits = model(src, dec_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=PAD, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"  step {step}: loss {loss.item():.3f}  exact-match {evaluate(model, vocab):.2%}")


def evaluate(model, vocab, n=256):
    model.eval()
    src, tgt = make_batch(n, vocab)
    pred = model.greedy_decode(src, max_len=tgt.size(1) - 1)
    pred = F.pad(pred, (0, tgt.size(1) - pred.size(1)), value=PAD)  # stopped early -> pad
    # a row is correct if every non-pad target token matches the prediction
    keep = tgt != PAD
    return ((pred == tgt) | ~keep).all(1).float().mean().item()

# ---------------------------------------------------------------- Demo
if __name__ == "__main__":
    torch.manual_seed(0)
    VOCAB = 13                                  # PAD, BOS, EOS + 10 digits

    model = Transformer(VOCAB, VOCAB, d_model=64, nhead=4, num_layers=2, d_ff=256)
    print(f"Transformer params: {sum(p.numel() for p in model.parameters()):,}")
    print("Task: reverse a digit sequence")
    train(model, VOCAB)

    src, _ = make_batch(3, VOCAB)
    out = model.greedy_decode(src, max_len=12)
    for s, o in zip(src.tolist(), out.tolist()):
        digits = lambda xs: [x - 3 for x in xs if x > EOS]
        print(f"  in  {digits(s)}\n  out {digits(o[1:o.index(EOS)] if EOS in o else o[1:])}")
