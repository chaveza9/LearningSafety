import torch
import torch.nn as nn
import slim
from activations import SoftExponential, ReHU


class MLP(nn.Module):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """

    def __init__(
            self,
            insize,
            outsize,
            bias=True,
            linear_map=slim.Linear,
            nonlin=SoftExponential,
            hsizes=[64],
            linargs=dict(),
    ):
        """
        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def forward(self, x):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x


class InputConvexNN(MLP):
    """
    Input convex neural network
    z1 =  sig(W0(x) + b0)
    z_i+1 = sig_i(Ui(zi) + Wi(x) + bi),  i = 1, ..., k-1
    V = g(x) = zk
    Equation 11 from https://arxiv.org/abs/2001.06116
    """

    def __init__(self,
                 insize,
                 outsize,
                 bias=True,
                 linear_map=slim.Linear,
                 nonlin=nn.ReLU,
                 hsizes=[64],
                 linargs=dict()
                 ):
        super().__init__(
            insize,
            outsize,
            bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes,
            linargs=linargs,

        )
        assert (
                len(set(hsizes)) == 1
        ), "All hidden sizes should be equal for residual network"

        sizes = hsizes + [outsize]
        self.linear = nn.ModuleList(
            [
                linear_map(insize, sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden)
            ]
        )
        self.poslinear = nn.ModuleList(
            [
                slim.NonNegativeLinear(sizes[k], sizes[k + 1], bias=False, **linargs)
                for k in range(self.nhidden)
            ]
        )
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden + 1)])

        self.inmap = linear_map(insize, hsizes[0], bias=bias, **linargs)
        self.in_features, self.out_features = insize, outsize

    def forward(self, x):
        xi = x
        px = self.inmap(xi)
        x = self.nonlin[0](px)
        for layer, (linU, nlin, linW) in enumerate(zip(self.poslinear, self.nonlin[1:], self.linear)):
            px = linW(xi)
            ux = linU(x)
            x = nlin(ux + px)
        return x


class PosDef(nn.Module):
    """
    Enforce positive-definiteness of lyapunov function ICNN, V = g(x)
    Equation 12 from https://arxiv.org/abs/2001.06116
    """

    def __init__(self, g, max=None, eps=0.01, d=1.0):
        """
        :param g: (nn.Module) An ICNN network
        :param eps: (float)
        :param d: (float)
        :param max: (float)
        """
        super().__init__()
        self.g = g
        self.in_features = self.g.in_features
        self.out_features = self.g.out_features
        self.zero = nn.Parameter(torch.zeros(1, self.g.in_features), requires_grad=False)
        self.eps = eps
        self.d = d
        self.smReLU = ReHU(self.d)
        self.max = max

    def forward(self, x):
        shift_to_zero = self.smReLU(self.g(x) - self.g(self.zero))
        quad_psd = self.eps * (x ** 2).sum(1, keepdim=True)
        z = shift_to_zero + quad_psd
        if self.max is not None:
            z = z - torch.relu(z - self.max)
        return z

class PosDefICNN(InputConvexNN):
    """ Creates a positive definite ICNN. (Class Kappa function) """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.posdef = PosDef(self)

    def forward(self, x):
        return self.posdef(x)
