import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2021).

    For anchor i with embedding z_i and positive set P(i) = {all views sharing
    i's label, excluding i}:

    L = sum_i  -1/|P(i)| * sum_{p in P(i)} log(
            exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau)
        )

    where A(i) = all indices except i.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        """
        Args:
            projections: (2B, proj_dim) — L2-normalized embeddings from both views
                         concatenated along batch dim [view1; view2]
            labels: (2B,) — repeated labels for both views

        Returns:
            Scalar loss.
        """
        device = projections.device
        N = projections.shape[0]

        # Cosine similarity matrix scaled by temperature
        sim = projections @ projections.T / self.temperature  # (N, N)

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float('-inf'))

        # Positive mask: same label, excluding self
        labels = labels.unsqueeze(0)  # (1, N)
        pos_mask = (labels == labels.T) & ~self_mask  # (N, N)

        # Log-softmax over each row (denominator = all non-self entries)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Mean of log-prob over positive pairs for each anchor
        pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

        return -pos_log_prob.mean()


class TotalLoss(nn.Module):
    """Combined SupCon + CE loss: L = L_SupCon + lambda * L_CE"""
    def __init__(self, temperature=0.07, ce_weight=0.1):
        super().__init__()
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, proj1, proj2, logits1, logits2, labels):
        """
        Args:
            proj1, proj2: (B, proj_dim) — L2-normalized projections from each view
            logits1, logits2: (B, num_classes) — classification logits from each view
            labels: (B,) — ground truth labels

        Returns:
            total_loss, supcon_loss, ce_loss (for logging)
        """
        # Concatenate both views for SupCon
        projections = torch.cat([proj1, proj2], dim=0)       # (2B, proj_dim)
        repeated_labels = torch.cat([labels, labels], dim=0)  # (2B,)

        supcon_loss = self.supcon(projections, repeated_labels)

        # CE on both views
        ce_loss = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2

        total = supcon_loss + self.ce_weight * ce_loss
        return total, supcon_loss, ce_loss
