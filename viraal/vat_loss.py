import torch
import torch.nn as nn
import torch.nn.functional as F


def _unsqueeze_to(source, target):
    new_size = torch.ones(len(target.size()), dtype=torch.int)
    source_size = torch.tensor(source.size())
    new_size[:len(source_size)] = source_size
    return source.view(new_size.tolist())

def _normalize_perturbation(perturb):
    #Refer to https://github.com/tensorflow/models/blob/master/research/adversarial_text/adversarial_losses.py
    batch_size = perturb.size(0)
    perturb_hat = perturb
    perturb_hat = perturb_hat / _unsqueeze_to(perturb_hat.view(batch_size,-1).max(dim=1)[0] + 1e-12, perturb_hat)
    perturb_hat = perturb_hat / _unsqueeze_to(perturb_hat.view(batch_size,-1).norm(dim=1)+ 1e-6, perturb_hat)

    return perturb_hat

def _prepare_perturb(pert, scale, input_mask):
    pert_hat = pert*_unsqueeze_to(input_mask, pert)
    pert_hat = scale*_normalize_perturbation(pert_hat)
    return pert_hat.detach()

class VatLoss(nn.Module):
    def __init__(self,
                 epsilon:float,
                 xi:float,
                 power_iterations:int):
        super().__init__()
        self.epsilon = epsilon
        self.xi = xi
        self.power_iterations = power_iterations
        self.kl_div = nn.KLDivLoss(reduction='none')

    def _kl_div_from_logits(self, original_logits, perturbed_logits):
        #We detach the input logits as specified in Miyato el al 2017
        original_logits = original_logits.detach()
        input_prob = F.softmax(original_logits, dim=-1)
        logprob_pert = F.log_softmax(perturbed_logits, dim=-1)
        # We calculate the KL Div without reduction and manually add up the class probabilities
        # that we suppose are in the last dimension. We then average over the batch 
        # (and sequence if tagging) dimension
        kl_div = torch.sum(self.kl_div(logprob_pert, input_prob), dim=-1)
        return kl_div

    def _generate_vadv_perturbation(self, 
                                    model_forward,
                                    original_logits, 
                                    model_input, 
                                    model_input_mask):

        d = torch.randn_like(model_input)

        # Perform finite difference method and power iteration.
        # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
        # Adding small noise to input and taking gradient with respect to the noise
        for k in range(self.power_iterations):
            d = _prepare_perturb(d, scale=self.xi, input_mask=model_input_mask)
            d.requires_grad_(True)
            
            perturbed_logits = model_forward(model_input + d)
            kl_div = torch.mean(self._kl_div_from_logits(original_logits, perturbed_logits))

            d = torch.autograd.grad(kl_div, d)[0]
        
        r_vadv = _prepare_perturb(d, self.epsilon, model_input_mask)
        return r_vadv
        

    def forward(self, 
                model_forward, 
                original_logits,
                model_input,
                model_input_mask):
        
        r_vadv = _generate_vadv_perturbation(model_forward, original_logits, model_input, model_input_mask)
        
        perturbed_logits = model_forward(model_input + r_vadv)
        kl_div = torch.mean(self._kl_div_from_logits(original_logits, perturbed_logits))

        return kl_div

