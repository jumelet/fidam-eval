import torch


def gen_dyck_deps(input_list):
    stack = []
    deps = []
    for idx, x in enumerate(input_list):
        if x < 2:
            stack.append((idx, x))
        else:
            prev_idx, y = stack.pop()
            assert (x - 2) == y, "string not well formed"
            deps.append((prev_idx, idx))

    return deps


def gen_fe_deps(_):
    return [(0, 1)]


def gen_palin_deps(input_list):
    sen_len = len(input_list)

    return [(i, sen_len - i - 1) for i in range(sen_len // 2)]


def invert_dyck(item):
    return torch.tensor([{0: 1, 1: 0, 2: 3, 3: 2}[x.item()] for x in item])


def invert_palin(item, n_items):
    new_item = []

    for idx, sym in enumerate(item):
        if idx < len(item) // 2:
            new_item.append((sym + 1) % n_items)
        else:
            new_item.append((sym + 1) % n_items + n_items)

    return torch.tensor(new_item)
