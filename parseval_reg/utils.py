import torch.nn as nn
import torch
import torch.nn.functional as F


def create_block_diag_matrix(block_size, num_blocks):
    # Create a single block of 1s
    block = torch.ones(block_size, block_size)

    # Create a list of blocks
    blocks = [block] * num_blocks

    # Use torch.block_diag to create a block diagonal matrix
    block_diag_matrix = torch.block_diag(*blocks)

    return block_diag_matrix

class IdentityActivation(nn.Module):
    """Applies the rectified linear unit function element-wise:

    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return input

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


from torch import Tensor
from torch.nn.parameter import Parameter

class DiagLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = x * c + b`.
        Multiplies by a scalar and adds a bias.

        Based on Linear layer code from Pytorch

    Args:
        in_features: size of each input sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.

        - Output: :math:`(*, H_{in})` . Same as input.

    Attributes:
        weight: the learnable weights of the module of shape H_{in}
                All initialized to 1

        bias:   the learnable bias of the module of shape H_{in}
                All initialized to 0

    """
    __constants__ = ['in_features']
    def __init__(self, in_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, input: Tensor) -> Tensor:
        return input * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, learnable=False):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        self.scale.requires_grad = learnable

    def forward(self, input):
        return input * self.scale



class GroupSort(nn.Module):
    ''' num_units is the number of groups to divide all the features in
    e.g. num_units=1 means sorting all the values in the layer '''
    def __init__(self, num_units, axis=1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.num_units, self.axis)
        # assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "
        # print(x)
        # print(group_sorted)
        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def process_group_size(x, num_units, axis=1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def group_sort(x, num_units, axis=1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))

    return sorted_x


# Redone version of maxmin where we the number of groups is determined dynamically by grouping into pairs
class Maxmin(nn.Module):
    ''' num_units is the number of groups to divide all the features in
    e.g. num_units=1 means sorting all the values in the layer '''
    def __init__(self, axis=1):
        super(Maxmin, self).__init__()
        self.axis = axis

    def forward(self, x):
        group_sorted = maxmin(x, self.axis)
        # assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "
        # print(x)
        # print(group_sorted)
        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def maxmin_process_group_size(x, axis=1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % 2:
        raise ValueError('number of features({}) is not a '
                         'multiple of ({})'.format(num_channels, 2))
    # num_units =
    size[axis] = -1
    if axis == -1:
        size += [2] #[num_channels // num_units]
    else:
        size.insert(axis+1, 2 )# num_channels // num_units)
    return size


def maxmin(x, axis=1):
    size = maxmin_process_group_size(x, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim, descending=True)
    sorted_x = sorted_grouped_x.view(*list(x.shape))
    return sorted_x


class ConcatReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ConcatReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),-1)
        return F.relu(x)

def orthogonal_layer_init(layer, gain=1, bias_const=0.0):
    if gain is None:
        gain = 1
    torch.nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


if __name__ == '__main__':
    ...