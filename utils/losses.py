import torch
import torch.nn as nn




class FwdLoss(nn.Module):
    def __init__(self, F):
        super(FwdLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsotmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)

    def forward(self, inputs, z):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        z = z.long()

        # Loss is computed as phi(Mf)
        Mp = self.F @ p.T
        L = - torch.sum(torch.log(Mp[z,range(Mp.size(1))]))
        #L = - torch.sum(torch.log(Mp[z,range(Mp.size(1))]+1e-10))
        return L

class FwdBwdLoss(nn.Module):
    def __init__(self, B, F, k = 0, beta = 1):
        super(FwdBwdLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsotmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)
        self.k = torch.tensor(k, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)

    def forward(self, inputs, z):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        z = z.long()

        # Loss is computed as z'B'*phi(Ff)
        Ff = self.F @ p.T 
        log_Ff = torch.log(Ff+1e-8)
        B_log_Ff = self.B.T @ log_Ff
        L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))]) + 0.5 * self.k * torch.sum(torch.abs(v)**self.beta)
        #L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))]+1e-10)
        return L
    
class EMLoss(nn.Module):
    def __init__(self,M):
        super(EMLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.M = torch.tensor(M)
        
    def forward(self,out,z):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logp = self.logsoftmax(out)

        p = torch.exp(logp)
        M_on_device = self.M.to(out.device)
        Q = p.detach() * M_on_device[z]
        #Q = p.detach() * torch.tensor(self.M[z])
        Q /= torch.sum(Q,dim=1,keepdim=True)

        L = -torch.sum(Q*logp)

        return L
    

class My_loss(nn.Module):
    def __init__(self):
        super(My_loss).__init__()

    def forward(self, inputs, targets):
        # Example loss function: Mean Squared Error
        loss = torch.mean((inputs - targets) ** 2)
        return loss

class MarginalChainLoss(nn.Module):
    """Marginal‑Chain loss (Chiang & Sugiyama, 2025).

    The *marginal chain* strategy repeatedly applies the same noise channel
    ``F`` **s** times, producing the *marginal* corruption matrix ``F^s``.  The loss
    then mirrors *FwdLoss* but with this compounded matrix.

    Args:
        F (array‑like, shape ``(C, C)``): base forward/transition matrix.
        steps (int): length of the chain (``s``). ``steps=1`` reduces to **FwdLoss**.
        eps (float): numerical stabiliser.
    """

    def __init__(self, F, steps: int = 2, eps: float = 1e-8):
        super().__init__()
        if steps < 1:
            raise ValueError("'steps' must be a positive integer (>=1).")
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        F_tensor = torch.as_tensor(F, dtype=torch.float32, device=dev)
        self.F_chain = torch.linalg.matrix_power(F_tensor, steps)  # F^steps

    def forward(self, inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        v = inputs - inputs.mean(dim=1, keepdim=True)
        p = self.softmax(v)
        z = z.long()

        Mp = self.F_chain @ p.T
        log_prob = torch.log(Mp[z, torch.arange(Mp.size(1))] + self.eps)
        return -log_prob.sum()
