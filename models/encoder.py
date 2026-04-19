import math
import torch
from torch import nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al., 2021.

    Encodes *relative* distances between tokens by rotating query/key vectors,
    so the model learns "frame X is 5 steps after frame Y" rather than
    "frame X is at absolute position 37."
    """
    def __init__(self, dim, max_T=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_T)

    def _build_cache(self, max_T):
        t = torch.arange(max_T, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                  # (max_T, dim/2)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)  # (max_T, dim/2)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, seq_len):
        """Returns (cos, sin) each of shape (seq_len, dim/2)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x, cos, sin):
    """Apply RoPE rotation to tensor x of shape (B, nhead, T, head_dim).

    Splits head_dim into pairs, rotates each pair by the position angle.
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    # cos/sin are (T, d2) -> broadcast to (1, 1, T, d2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings.

    Drop-in replacement for nn.MultiheadAttention (batch_first=True).
    CLS token at position 0 gets position 0 rotation (no special treatment needed).
    """
    def __init__(self, d_model, nhead, dropout=0.1, max_T=512):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_T)

    def forward(self, x, src_key_padding_mask=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, nhead, T, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE to queries and keys
        cos, sin = self.rope(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, nhead, T, T)

        if src_key_padding_mask is not None:
            # Mask shape: (B, T) -> (B, 1, 1, T) for broadcasting
            attn = attn.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer using RoPE attention instead of absolute positions."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, max_T=512):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout, max_T)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Pre-norm architecture (more stable training)
        x = x + self.dropout(self.self_attn(self.norm1(x), src_key_padding_mask))
        x = x + self.ffn(self.norm2(x))
        return x


class SignLanguageEncoder(nn.Module):
    def __init__(self, num_classes, body_dim=69, left_hand_dim=63, right_hand_dim=63,
                 emb_dim=64, nhead=8, num_layers=4, proj_dim=128, max_T=512,
                 dropout=0.1, use_rope=False, use_triplet=True):
        super().__init__()
        self.use_rope = use_rope
        self.use_triplet = use_triplet

        # d_model must be the same regardless of tokenization so the
        # transformer, heads, and checkpoints are comparable.
        d_model = emb_dim * 3  # 192
        self.d_model = d_model

        if use_triplet:
            # Pose-Triplet: independent projections per body part
            self.body_proj = nn.Linear(body_dim, emb_dim)
            self.left_proj = nn.Linear(left_hand_dim, emb_dim)
            self.right_proj = nn.Linear(right_hand_dim, emb_dim)
        else:
            # Flat baseline: single projection from concatenated keypoints
            flat_dim = body_dim + left_hand_dim + right_hand_dim  # 195
            self.flat_proj = nn.Linear(flat_dim, d_model)

        # Learnable [CLS] token (BERT-style init, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        if use_rope:
            # RoPE: positions encoded via rotations inside attention — no pos_encoding needed
            layers = [RoPETransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, max_T)
                      for _ in range(num_layers)]
            self.transformer = nn.ModuleList(layers)
        else:
            # Absolute learned positional encoding
            self.pos_encoding = nn.Parameter(torch.zeros(1, 1 + max_T, d_model))
            nn.init.normal_(self.pos_encoding, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm on CLS output before heads
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)

        # MLP projection head for SupCon loss (with BatchNorm per Khosla et al.)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

        # Linear classification head for CE loss
        self.classification_head = nn.Linear(d_model, num_classes)

    def _project_tokens(self, tokens):
        """Project raw tokens to d_model-dim frame tokens.

        Returns: (B, T, d_model)
        """
        if self.use_triplet:
            body = tokens[:, :, 0, :self.body_proj.in_features]    # (B, T, 69)
            left = tokens[:, :, 1, :self.left_proj.in_features]    # (B, T, 63)
            right = tokens[:, :, 2, :self.right_proj.in_features]  # (B, T, 63)
            body_emb = self.body_proj(body)
            left_emb = self.left_proj(left)
            right_emb = self.right_proj(right)
            return torch.cat([body_emb, left_emb, right_emb], dim=-1)
        else:
            # Flat: concat body(69) + left(63) + right(63) = 195
            body = tokens[:, :, 0, :69]
            left = tokens[:, :, 1, :63]
            right = tokens[:, :, 2, :63]
            flat = torch.cat([body, left, right], dim=-1)  # (B, T, 195)
            return self.flat_proj(flat)

    def encode(self, tokens, mask=None):
        """Shared encoder forward pass — returns full sequence output (B, T+1, d_model).

        Used by both the fine-tuning forward() and the pre-training module.
        """
        B, T, _, _ = tokens.shape

        frame_tokens = self._project_tokens(tokens)  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame_tokens], dim=1)  # (B, T+1, d_model)

        # Build attention mask (prepend False for CLS)
        src_key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)

        if self.use_rope:
            # RoPE: no positional encoding added; positions are in the attention
            for layer in self.transformer:
                seq = layer(seq, src_key_padding_mask)
        else:
            seq = seq + self.pos_encoding[:, :T + 1, :]
            seq = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        return seq

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: (B, T, 3, max_dim) where dim 2 is [body, left, right]
            mask: (B, T) — True for padded positions (optional)

        Returns:
            projections: (B, proj_dim) - L2 normalized for SupCon
            logits: (B, num_classes) - for CE loss
        """
        encoded = self.encode(tokens, mask)

        # Extract [CLS] output, normalize and dropout before heads
        cls_out = self.head_dropout(self.norm(encoded[:, 0, :]))

        # Two heads
        projections = self.projection_head(cls_out)
        projections = F.normalize(projections, dim=-1)

        logits = self.classification_head(cls_out)

        return projections, logits
