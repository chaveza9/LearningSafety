"""
Elementwise nonlinear tensor operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_exp(alpha, x):
    """
    Helper function for SoftExponential learnable activation class. Also used in neuromancer.operators.InterpolateAddMultiply
    :param alpha: (float) Parameter controlling shape of the function.
    :param x: (torch.Tensor) Arbitrary shaped tensor input
    :return: (torch.Tensor) Result of the function applied elementwise on the tensor.
    """
    if alpha == 0.0:
        return x
    elif alpha < 0.0:
        return -torch.log(1 - alpha * (x + alpha)) / alpha
    else:
        return (torch.exp(alpha * x) - 1) / alpha + alpha
    
class SoftExponential(nn.Module):
    """
    Soft exponential activation: https://arxiv.org/pdf/1602.01321.pdf
    """

    def __init__(self, alpha=0.0, tune_alpha=True):
        """
        :param alpha: (float) Value to initialize parameter controlling the shape of the function
        :param tune_alpha: (bool) Whether alpha is a learnable parameter or fixed
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=tune_alpha)

    def forward(self, x):
        """
        :param x: (torch.Tensor) Arbitrary shaped tensor
        :return: (torch.Tensor) Tensor same shape as input after elementwise application of soft exponential function
        """
        return soft_exp(self.alpha, x)

    
class ReHU(nn.Module):
    """
    ReLU with a quadratic region in [0,d]; Rectified Huber Unit;
    Used to make the Lyapunov function continuously differentiable
    https://arxiv.org/pdf/2001.06116.pdf
    """
    def __init__(self, d=1.0, tune_d=True):
        super().__init__()
        self.d = nn.Parameter(torch.tensor(d), requires_grad=tune_d)

    def forward(self, x):
        alpha = 1.0 / F.softplus(self.d)
        beta = - F.softplus(self.d) / 2
        return torch.max(torch.clamp(torch.sign(x) * torch.div(alpha, 2.0) * x ** 2, min=0, max=-beta.item()), x + beta)