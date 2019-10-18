import inspect

import torch.nn as nn


class LambdaLayer(nn.Module):
    """Module/Layer that encapsulates a single function for PyTorch

    This is to make it easier to a lambda in an nn.Sequential() container.
    """
    def __init__(self, lm):
        """

        Args:
            lm (Callable): function to use/call when the module is called.
        """
        super().__init__()
        self.lm = lm
        # this is because I want to see whatever the anonymous function is
        # but I do not know how to parse python syntax or want to learn to write a parser now
        self.src = inspect.getsourcelines(self.lm)
        if len(self.src[0]) == 1:
            module_code_str: str = self.src[0][0]
            lam_start_pos = module_code_str.find("lambda")
            # the case where def f(x): ... is a one liner
            if lam_start_pos == -1 and module_code_str[:4] == 'def ':
                xtr_repr = module_code_str.strip('\r\n')
            else:
                xtr_repr = module_code_str[lam_start_pos:]  # finds the start of "lambda..."
                xtr_repr = xtr_repr.strip(')\r\n')  # removes trailing parenthesis and newlines
            self.xtr_repr = xtr_repr
        else:
            self.xtr_repr = '(not lambda)'

    def forward(self, *input):
        return self.lm(*input)

    def extra_repr(self) -> str:
        return self.xtr_repr
