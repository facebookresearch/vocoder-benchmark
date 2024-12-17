# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


import numpy as np


# https://github.com/tensorflow/tensor2tensor/issues/280#issuecomment-339110329
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def noam_learning_rate_decay(init_lr, global_step, warmup_steps: float = 4000):
    # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    step = global_step + 1.0
    lr = init_lr * warmup_steps**0.5 * np.minimum(step * warmup_steps**-1.5, step**-0.5)
    return lr


# pyre-fixme[3]: Return type must be annotated.
def step_learning_rate_decay(
    # pyre-fixme[2]: Parameter must be annotated.
    init_lr,
    # pyre-fixme[2]: Parameter must be annotated.
    global_step,
    anneal_rate: float = 0.98,
    anneal_interval: int = 30000,
):
    return init_lr * anneal_rate ** (global_step // anneal_interval)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def cyclic_cosine_annealing(init_lr, global_step, T, M):
    """Cyclic cosine annealing
    https://arxiv.org/pdf/1704.00109.pdf
    Args:
        init_lr (float): Initial learning rate
        global_step (int): Current iteration number
        T (int): Total iteration number (i,e. nepoch)
        M (int): Number of ensembles we want
    Returns:
        float: Annealed learning rate
    """
    TdivM = T // M
    return init_lr / 2.0 * (np.cos(np.pi * ((global_step - 1) % TdivM) / TdivM) + 1.0)
