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