from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from PNNL's slim code https://github.com/pnnl/slim/blob/master/slim/linear.py
class LinearBase(nn.Module, ABC):
    """
    Base class defining linear map interface.
    """

    def __init__(self, insize, outsize, bias=False, provide_weights=True):
        """
        :param insize: (int) Input dimensionality
        :param outsize: (int) Output dimensionality
        :param bias: (bool) Whether to use affine or linear map
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        if bias:
            bound = 1 / math.sqrt(insize)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        if provide_weights:
            self.weight = nn.Parameter(torch.Tensor(insize, outsize))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def device(self):
        return next(self.parameters()).device

    def reg_error(self):
        """
        Regularization error associated with linear map parametrization.
        :return: (torch.float)
        """
        return torch.tensor(0.0).to(self.device)

    def eig(self, eigenvectors=False):
        """
        Returns the eigenvalues (optionally eigenvectors) of the linear map used in matrix multiplication.
        :param eigenvectors: (bool) Whether to return eigenvectors along with eigenvalues.
        :return: (torch.Tensor) Vector of eigenvalues, optionally a tuple including a matrix of eigenvectors.
        """
        return torch.eig(self.effective_W(), eigenvectors=eigenvectors)

    @abstractmethod
    def effective_W(self):
        """
        The matrix used in the equivalent matrix multiplication for the parametrization
        :return: (torch.Tensor, shape=[insize, outsize]) Matrix used in matrix multiply
        """
        pass

    def forward(self, x):
        """
0-
        :param x: (torch.Tensor, shape=[batchsize, in_features])
        :return: (torch.Tensor, shape=[batchsize, out_features])
        """
        return torch.matmul(x, self.effective_W()) + self.bias


class Linear(LinearBase):
    """
    Wrapper for torch.nn.Linear with additional slim methods returning matrix,
    eigenvectors, eigenvalues and regularization error.
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.linear = nn.Linear(insize, outsize, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)

class NonNegativeLinear(LinearBase):
    """
    Positive parametrization of linear map via Relu.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
        self.weight = nn.Parameter(torch.abs(self.weight)*0.1)

    def effective_W(self):
        return F.relu(self.weight)
