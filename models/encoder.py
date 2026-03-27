import torch
from torch import nn


class SignLanguageEncoder(nn.Module):
    def __init__(self, num_classes, body_dim=69, left_hand_dim=63, right_hand_dim=63,
                 emb_dim=64, nhead=8, num_layers=4, proj_dim=128, max_T=512):
        super().__init__()

        # Per-body-part projections (equal capacity)
        self.body_proj = nn.Linear(body_dim, emb_dim)
        self.left_proj = nn.Linear(left_hand_dim, emb_dim)
        self.right_proj = nn.Linear(right_hand_dim, emb_dim)

        # Token dim after concatenation: 64 * 3 = 192
        d_model = emb_dim * 3

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (max T+1 tokens: CLS + T frame tokens)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1 + max_T, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP projection head for SupCon loss -> L2 normalized embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

        # Linear classification head for CE loss
        self.classification_head = nn.Linear(d_model, num_classes)

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: (B, T, 3, max_dim) where dim 2 is [body, left, right]
                    body uses [:, :, 0, :body_dim], hands use [:, :, 1/2, :hand_dim]
            mask: (B, T) — True for padded positions (optional)

        Returns:
            projections: (B, proj_dim) - L2 normalized for SupCon
            logits: (B, num_classes) - for CE loss
        """
        B, T, _, _ = tokens.shape

        body = tokens[:, :, 0, :self.body_proj.in_features]    # (B, T, 69)
        left = tokens[:, :, 1, :self.left_proj.in_features]    # (B, T, 63)
        right = tokens[:, :, 2, :self.right_proj.in_features]  # (B, T, 63)

        # Project each body part independently, then concatenate into one token
        body_emb = self.body_proj(body)     # (B, T, d_model)
        left_emb = self.left_proj(left)     # (B, T, d_model)
        right_emb = self.right_proj(right)  # (B, T, d_model)
        frame_tokens = torch.cat([body_emb, left_emb, right_emb], dim=-1)  # (B, T, 3*d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 3*d_model)
        seq = torch.cat([cls, frame_tokens], dim=1)  # (B, T+1, 3*d_model)

        # Add positional encoding
        seq = seq + self.pos_encoding[:, :T + 1, :]

        # Build attention mask if padding mask provided (prepend False for CLS)
        src_key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer encoder
        encoded = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        # Extract [CLS] output
        cls_out = encoded[:, 0, :]  # (B, 3*d_model)

        # Two heads
        projections = self.projection_head(cls_out)
        projections = nn.functional.normalize(projections, dim=-1)

        logits = self.classification_head(cls_out)

        return projections, logits
