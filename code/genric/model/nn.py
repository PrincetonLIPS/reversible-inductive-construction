""" Utility custom pytorch modules """

import torch


class Sequential(torch.nn.Module):
    """ Similar to `torch.nn.Sequential`, except can pass through modules which
    take multiple input arguments, and return tuples.
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.__modules = torch.nn.ModuleList(args)

    def forward(self, *args):
        for module in self.__modules:
            if not isinstance(args, (list, tuple)):
                args = [args]
            args = module(*args)

        return args


class SumModule(torch.nn.Module):
    """ A simple module which sums all its inputs """
    def __init__(self):
        super(SumModule, self).__init__()

    def forward(self, *args):
        result = args[0]

        for other in args[1:]:
            result = torch.add(result, other)

        return result


class FunctionalModule(torch.nn.Module):
    """ Module which applies the given function to its input. """
    def __init__(self, function):
        super(FunctionalModule, self).__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


class PassthroughModule(torch.nn.Module):
    def __init__(self, module):
        super(PassthroughModule, self).__init__()
        self.module = module

    def forward(self, *args):
        result = self.module(*args)
        if not isinstance(result, (list, tuple)):
            result = (result,)

        return args + result


class AvgAndMaxPool(torch.jit.ScriptModule):
    def __init__(self, input_size):
        super(AvgAndMaxPool, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.input_size = input_size
        self.output_size = input_size * 2

    @torch.jit.script_method
    def forward(self, x):
        x = x.t().unsqueeze(0)
        x_avg = self.avg_pool(x).squeeze()
        x_max = self.max_pool(x).squeeze()

        return torch.cat([x_avg, x_max], dim=0)
