"""BERT-style Masked Pose Modeling for pre-training the SignLanguageEncoder.

Masks a fraction of frame tokens, replaces them with a learnable [MASK] embedding,
and trains the encoder to reconstruct the original token values (MSE loss).
This gives the encoder a strong understanding of temporal pose dynamics before
it ever sees labels.

Usage:
    python -m models.pretrain --epochs 50 --mask_ratio 0.15
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Re-use the dataset and collate from train.py (eval mode = single view, no augment)
from train import ASLKeypointDataset, collate_eval


class MaskedPoseModeling(nn.Module):
    """Wraps a SignLanguageEncoder for masked pose pre-training.

    Masks random frame positions, feeds them through the encoder,
    and reconstructs the original frame tokens via a lightweight MLP decoder.
    """
    def __init__(self, encoder, mask_ratio=0.15):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d_model = encoder.d_model

        # Learnable [MASK] token (replaces masked frame tokens)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Reconstruction head: predicts original d_model-dim frame token
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: (B, T, 3, max_dim)
            mask: (B, T) — True for padded positions

        Returns:
            loss: MSE reconstruction loss on masked positions
        """
        B, T, _, _ = tokens.shape
        device = tokens.device

        # Build the target: project tokens to get ground truth frame embeddings
        with torch.no_grad():
            target = self.encoder._project_tokens(tokens)  # (B, T, d_model)

        # Create mask for which frame positions to mask (exclude padded positions)
        # rand_mask: True = will be masked for reconstruction
        rand_mask = torch.rand(B, T, device=device) < self.mask_ratio
        if mask is not None:
            rand_mask = rand_mask & ~mask  # Don't mask padded positions

        # Replace masked frame tokens with [MASK] before encoding
        frame_tokens = self.encoder._project_tokens(tokens)  # (B, T, d_model)

        # Replace masked positions
        mask_expand = rand_mask.unsqueeze(-1).expand_as(frame_tokens)
        frame_tokens = torch.where(mask_expand, self.mask_token.expand_as(frame_tokens), frame_tokens)

        # Prepend [CLS] and run through transformer
        cls = self.encoder.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame_tokens], dim=1)

        src_key_padding_mask = None
        if mask is not None:
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
            src_key_padding_mask = torch.cat([cls_pad, mask], dim=1)

        if self.encoder.use_rope:
            for layer in self.encoder.transformer:
                seq = layer(seq, src_key_padding_mask)
        else:
            seq = seq + self.encoder.pos_encoding[:, :T + 1, :]
            seq = self.encoder.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        # Reconstruct masked positions (skip CLS at index 0)
        frame_output = seq[:, 1:, :]  # (B, T, d_model)
        reconstructed = self.reconstruction_head(frame_output)

        # MSE loss only on masked positions
        loss = ((reconstructed - target) ** 2)  # (B, T, d_model)
        loss = (loss * rand_mask.unsqueeze(-1)).sum() / rand_mask.sum().clamp(min=1) / target.shape[-1]

        return loss


def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pre-training] Using device: {device}")

    # Load training data (no augmentation — we just need raw sequences)
    train_dir = os.path.join('data', 'keypoints', 'train')
    dataset = ASLKeypointDataset(train_dir, augment=False)
    num_classes = len(dataset.labels)
    print(f"[Pre-training] Samples: {len(dataset)}, Classes: {num_classes}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_eval, num_workers=args.num_workers)

    # Build encoder + pre-training wrapper
    from models.encoder import SignLanguageEncoder
    encoder = SignLanguageEncoder(
        num_classes=num_classes, use_rope=args.use_rope, use_triplet=args.use_triplet
    ).to(device)
    model = MaskedPoseModeling(encoder, mask_ratio=args.mask_ratio).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule (no warmup needed for pre-training — MSE is stable)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for tokens, mask, _ in loader:
            tokens, mask = tokens.to(device), mask.to(device)

            loss = model(tokens, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | lr {lr:.6f} | recon loss {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, 'pretrained_encoder.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'loss': best_loss,
                'use_rope': args.use_rope,
            }, save_path)
            print(f"  -> Saved best pre-trained encoder (loss: {best_loss:.6f})")

    print(f"\n[Pre-training complete] Best reconstruction loss: {best_loss:.6f}")
    print(f"Encoder weights saved to: {os.path.join(args.save_dir, 'pretrained_encoder.pt')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT-style masked pose pre-training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--use_triplet', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    pretrain(args)
