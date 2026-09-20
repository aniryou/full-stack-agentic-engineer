"""
Lesson 3 - Train a tiny GPT and watch it learn (PyTorch, CPU, ~30 seconds).

The task is "copy": each sequence is K random digits, a separator, then the same K digits.
    3 1 4 1 5 9 | 3 1 4 1 5 9
Reading left to right, the first half is unpredictable and the second half is fully determined.
Watching the loss on each half separately shows exactly what "learning to predict the next token" means.
Companion: docs/transformer-primer.md, Sections 5-7.

    python lessons/03_tiny_gpt.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=140)

# --- data -----------------------------------------------------------------------
V, K = 10, 6                      # digits 0-9; copy 6 of them
SEP = V                           # token id 10 is the separator, so vocab size is 11
T = 2 * K + 1                     # sequence length: 6 digits + separator + 6 digits = 13


def make_batch(B):
    digits = torch.randint(0, V, (B, K))
    sep = torch.full((B, 1), SEP)
    return torch.cat([digits, sep, digits], dim=1)      # (B, T)


# --- model: the primer's 50 lines, plus GPT-2 initialisation ----------------------
class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.last_weights = None                        # kept so we can look at what it learned

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        self.last_weights = w.detach()
        return self.out((w @ v).transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        for p in self.parameters():                     # GPT-2 init: small weights -> initial loss ~ ln(vocab)
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))                  # (B, T, vocab) logits


def split_losses(model, x):
    """Per-position next-token loss, averaged separately over the unpredictable and the predictable columns."""
    logits = model(x)[:, :-1]                            # position t predicts token t+1
    targets = x[:, 1:]
    per_token = F.cross_entropy(logits.reshape(-1, V + 1), targets.reshape(-1), reduction="none").view(x.shape[0], -1)
    random_part = per_token[:, : K - 1].mean()           # predicting digits 2..K: nothing to go on
    copy_part = per_token[:, K:].mean()                  # predicting the second half: fully determined by the first
    return per_token.mean(), random_part, copy_part


# --- train ----------------------------------------------------------------------
model = GPT(vocab=V + 1, d=32, n_layers=2, n_heads=4, max_len=T)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"loss at init: {split_losses(model, make_batch(256))[0]:.2f}   (ln(11) = {math.log(11):.2f}: a uniform guess)")
print(f"floor for the random half: ln(10) = {math.log(10):.2f}   floor for the copy half: 0\n")
print(f"{'step':>5} {'total':>7} {'random':>7} {'copy':>7}")
for step in range(1501):
    x = make_batch(64)
    total, rand_loss, copy_loss = split_losses(model, x)
    opt.zero_grad()
    total.backward()
    opt.step()
    if step % 150 == 0:
        print(f"{step:>5} {total.item():>7.2f} {rand_loss.item():>7.2f} {copy_loss.item():>7.2f}")
print("\nThe random half never improves: the model can only learn what is predictable.")
print("The copy half goes to zero: it found a rule nobody wrote down.\n")


# --- generate: autoregressive decoding, one token at a time ---------------------
@torch.no_grad()
def generate(model, prefix, n_new):
    x = prefix.clone()
    for _ in range(n_new):
        logits = model(x)[:, -1]                          # only the last position's prediction is needed
        nxt = logits.argmax(-1, keepdim=True)             # greedy: take the most likely token
        x = torch.cat([x, nxt], dim=1)                    # append and go again (a KV cache would avoid recomputing)
    return x


print("=== generation ===")
for prefix in (torch.tensor([[3, 1, 4, 1, 5, 9, SEP]]), torch.tensor([[7, 7, 0, 2, 7, 0, SEP]])):
    out = generate(model, prefix, K)
    print("prompt", prefix[0, :K].tolist(), "|  ->  model continues", out[0, K + 1:].tolist())
print()

# --- look inside: which head does the copying? ---------------------------------
print("=== the learned attention pattern ===")
x = make_batch(1)
model(x)
print(f"sequence: {x[0].tolist()}\n")
print(f"share of attention on 'the token {K} positions back', averaged over the copy positions:")
best = None
for li, blk in enumerate(model.blocks):
    w = blk.attn.last_weights[0]                          # (heads, T, T)
    stripes = [torch.stack([w[hi, t, t - K] for t in range(K, T - 1)]).mean().item() for hi in range(w.shape[0])]
    print(f"  layer {li}: " + "   ".join(f"head {hi}: {s:.0%}" for hi, s in enumerate(stripes)))
    for hi, s in enumerate(stripes):
        if best is None or s > best[0]:
            best = (s, li, hi)
stripe, li, hi = best
print(f"\nlayer {li}, head {hi} (rows = where each position looks):")
print(model.blocks[li].attn.last_weights[0, hi])
print(f"""
Rows {K}..{T - 2} are the positions that must produce the copy. Most of them put nearly all their
weight on the digit exactly {K} steps back: the one they need to output next. Where this head
doesn't, another head covers that position; the work is shared. The last row predicts nothing
(there is no next token), so it was never trained and is just noise.

Nobody wrote a copy rule. Attention learned to move the right digit across positions.""")
