import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(d):
    return F.normalize(d, p=2, dim=-1)

class VAT(nn.Module):
    def __init__(self, device, eps, xi=1e-1, k=1):
        super().__init__()
        self.device = device
        self.eps = eps
        self.k = k
        self.xi = xi
        if device >= 0:
            self.push = lambda x: x.to(self.device)
        else:
            self.push = lambda x: x
        self.kl_div = self.push(nn.KLDivLoss(
            size_average=None, reduction='none'))

    def __call__(self, model, int_score, emb):

        prob_int = F.softmax(int_score.detach(), dim=-1)
        
        d = self.push(_l2_normalize(torch.randn_like(emb)))

        emb_hat = emb
        for _ in range(self.k):
            emb_hat = emb_hat + self.xi * d
            int_score_hat = model(emb_hat)
            adv_distance = 0
            adv_distance_int = torch.mean(self.kl_div(
                F.log_softmax(int_score_hat, dim=-1), prob_int).sum(dim=-1))
            adv_distance = adv_distance + adv_distance_int
        
            d = self.push(_l2_normalize(torch.autograd.grad(adv_distance, emb_hat)[0]))

        emb_hat = emb + self.eps * d.detach()
        int_score_hat = model(emb_hat)
        
        a = self.kl_div(F.log_softmax(int_score_hat, dim=-1), prob_int)
        kl_intents = a.sum(dim=-1)
        return kl_intents
