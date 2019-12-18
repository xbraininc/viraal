from viraal.vat_loss import VatLoss
import torch
import pytest

def prob_distribution(prob):
    return torch.stack([prob,1-prob], dim=-1) 

def simple_linear_model(x):
    prob = torch.nn.functional.sigmoid(x.view(x.size(0),-1).sum(dim=1))
    return prob_distribution(prob)

def simple_non_linear_model(X):
    x,y = X[:,:,0], X[:,:,1]
    prob = torch.cos(x+y-3.14)+(x**3+y**3)/10
    return prob_distribution(prob)

class TestVatLoss:
    def setup(self):
        self.loss = VatLoss(1, 1e-6, 10)

    def test_perturbation(self):
        dummy_mask = torch.BoolTensor([[1,1,0,0],[1,1,1,0]])
        dummy_input = torch.zeros(2, 4, 2)
        dummy_logits = simple_non_linear_model(dummy_input)
        r_vadv = self.loss._generate_vadv_perturbation(simple_non_linear_model, dummy_logits, dummy_input, dummy_mask)
        value_1 = torch.tensor(0.5)
        value_2 = torch.tensor(0.4082)
        assert r_vadv[0,:2,:].allclose(value_1, atol=0.001)
        assert (r_vadv[0,2:,:] == 0).all()
        assert r_vadv[1,:3,:].allclose(value_2, atol=0.001)
        assert (r_vadv[1,3:,:] == 0).all()
