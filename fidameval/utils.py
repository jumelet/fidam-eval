import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    """ Base class for configuration objects. """

    def __init__(self, **kwargs):
        for kwarg, val in kwargs.items():
            setattr(self, kwarg, val)

    def __repr__(self):
        representation = ""

        max_len = max(map(len, self.__dict__.keys()))
        for key, value in self.__dict__.items():
            str_value = str(value).split("\n")[0][:20]
            if len(str(value).split("\n")) > 1 or len(str(value).split("\n")[0]) > 20:
                str_value += " [..]"
            representation += f"{key:<{max_len + 3}}{str_value}\n"

        return representation


def unpad_sequence(tensor, lengths):
    """Casts a padded tensor to a concated tensor that omits the pad
    positions.
    """
    concat_fn = torch.stack if tensor.ndim > 2 else torch.tensor
    tensor_list = [value for row, idx in zip(tensor, lengths) for value in row[:idx]]

    return concat_fn(tensor_list).to(DEVICE)


def flatten(container):
    """ Flatten iterator of iterators to single iterator. """
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
