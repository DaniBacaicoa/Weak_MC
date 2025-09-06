import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- core: scoring rules ----------
def scoring_matrix(p: torch.Tensor, loss_code: str, eps: float = 1e-8) -> torch.Tensor:
    """
    Given class probabilities p (B, C), return S(p, y=c) for all c as a matrix (B, C).
    Each column c is the loss value if the true label is c.
    """
    p = p.clamp_min(eps)                     # avoid log/zero issues
    B, C = p.shape

    if loss_code == "cross_entropy":
        # S(p,y=c) = -log p_c
        return -torch.log(p)

    elif loss_code in ("brier", "squared", "mse"):
        # S(p,y=c) = (p - e_c)^2 summed over classes = 1 - 2 p_c + sum_j p_j^2
        sumsq = (p * p).sum(dim=1, keepdim=True)          # (B,1)
        return 1.0 - 2.0 * p + sumsq                      # (B,C)

    elif loss_code == "spherical":
        # S(p,y=c) = - p_c / ||p||_2
        denom = p.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return -p / denom

    elif loss_code.startswith("ps_"):  # pseudo-spherical: β > 1
        beta = float(loss_code.split("_", 1)[1])
        if beta <= 1:
            raise ValueError("pseudo-spherical requires beta > 1")
        # S(p,y=c) = - p_c^β / (β * ||p||_β)
        denom = (p.pow(beta).sum(dim=1, keepdim=True)).pow(1.0 / beta).clamp_min(eps)
        return -(p.pow(beta)) / (beta * denom)

    elif loss_code.startswith("tsallis_"):  # Tsallis score: α ≠ 1
        alpha = float(loss_code.split("_", 1)[1])
        if abs(alpha - 1.0) < 1e-6:
            # limit α→1 equals log score
            return -torch.log(p)
        # One standard Tsallis (power) scoring rule:
        # S(p,y=c) = (p_c^{α-1} - sum_j p_j^α) / (α - 1)
        a = alpha - 1.0
        sum_pow = p.pow(alpha).sum(dim=1, keepdim=True)
        return (p.pow(alpha - 1.0) - sum_pow) / a

    else:
        raise ValueError(f"Unknown proper loss code: {loss_code}")

# ---------- EM-style marginal-chain objective ----------
class MarginalChainProperLoss(nn.Module):
    """
    General EM-style marginal-chain loss:
      E-step: Q ∝ stop_grad(p) ◦ M[z]
      M-step: minimize E_Q[ S(p, ·) ] using the chosen proper scoring rule.
    """
    def __init__(self, M, loss_code: str, reduction: str = "mean"):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.M = torch.as_tensor(M, dtype=torch.float32)
        self.loss_code = loss_code
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.long()
        logp = self.logsoftmax(logits)        # (B,C)
        p = logp.exp()                        # (B,C)

        # E-step responsibilities (detached)
        Mz = self.M.to(logits.device)[z]      # (B,C) rows indexed by z
        Q = (p.detach() * Mz)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # M-step objective: E_Q[ S(p,·) ]
        S = scoring_matrix(p, self.loss_code) # (B,C)
        loss_per_sample = (Q * S).sum(dim=1)  # (B,)
        return loss_per_sample.mean() if self.reduction == "mean" else loss_per_sample.sum()

# ---------- Forward (plug-in) marginal-chain objective ----------
class ForwardProperLoss(nn.Module):
    """
    Plug-in marginal-chain loss:
      r = (F @ p)^T, then apply S(r, y=z).
    Works for any proper scoring rule implemented in scoring_matrix.
    """
    def __init__(self, F_mat, loss_code: str, reduction: str = "mean"):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.F = torch.as_tensor(F_mat, dtype=torch.float32)
        self.loss_code = loss_code
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.long()
        p = self.softmax(logits - logits.mean(dim=1, keepdim=True))  # (B,C)
        r = (self.F.to(logits.device) @ p.T).T                       # (B,C)

        S = scoring_matrix(r, self.loss_code)                        # (B,C)
        loss_per_sample = S.gather(1, z.view(-1, 1)).squeeze(1)      # S(r,y=z)
        return loss_per_sample.mean() if self.reduction == "mean" else loss_per_sample.sum()
