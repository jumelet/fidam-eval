""" Methods for computing the true dependency relations of interest. """


def gen_fe_deps(_):
    return [(0, 1)]


def gen_dyck_deps(input_list):
    stack = []
    deps = []
    for idx, x in enumerate(input_list):
        if x < 2:
            stack.append((idx, x))
        else:
            prev_idx, y = stack.pop()
            assert (x - 2) == y, f"string not well formed: {input_list}"
            deps.append((prev_idx, idx))

    return deps


def gen_palindrome_deps(input_list):
    sen_len = len(input_list)

    return [(i, sen_len - i - 1) for i in range(sen_len // 2)]
