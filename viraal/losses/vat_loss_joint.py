import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from viraal.losses.vat_loss import VatLoss, _unsqueeze_to, _normalize_perturbation, _prepare_perturb

class VatLossJoint(VatLoss):
    def _generate_vadv_perturbation(self, 
                                    model_forward,
                                    original_logits, 
                                    model_input, 
                                    model_input_mask=None):

        d = torch.randn_like(model_input)

        # Perform finite difference method and power iteration.
        # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
        # Adding small noise to input and taking gradient with respect to the noise
        for k in range(self.power_iterations):
            d = _prepare_perturb(d, scale=self.xi, input_mask=model_input_mask)
            d.requires_grad_(True)
            
            perturbed_logits = model_forward(model_input + d)
            
            kl_div = lambda logits : torch.mean(self._kl_div_from_logits(logits[0], logits[1]))

            all_kl_div = sum(map(kl_div, zip(original_logits, perturbed_logits)))

            d = torch.autograd.grad(all_kl_div, d)[0]
        
        r_vadv = _prepare_perturb(d, self.epsilon, model_input_mask)
        return r_vadv
        

    def forward(self, 
                original_logits,
                model_forward,
                model_input,
                model_input_mask=None):
        
        r_vadv = self._generate_vadv_perturbation(model_forward, original_logits, model_input, model_input_mask)
        
        perturbed_logits = model_forward(model_input + r_vadv)

        kl_div_func = lambda logits : self._kl_div_from_logits(logits[0], logits[1])

        kl_div = tuple(map(kl_div_func, zip(original_logits, perturbed_logits)))

        return kl_div