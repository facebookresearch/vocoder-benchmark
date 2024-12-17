# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


"""
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import sys
from functools import partial
from typing import TextIO

import numpy as np
import torch.nn as nn


# pyre-fixme[3]: Return type must be annotated.
def get_model_complexity_info(
    # pyre-fixme[2]: Parameter must be annotated.
    model,
    # pyre-fixme[2]: Parameter must be annotated.
    input,
    print_per_layer_stat: bool = False,
    as_strings: bool = False,
    # pyre-fixme[2]: Parameter must be annotated.
    input_constructor=None,
    ost: TextIO = sys.stdout,
    verbose: bool = False,
    # pyre-fixme[2]: Parameter must be annotated.
    ignore_modules=[],
    # pyre-fixme[2]: Parameter must be annotated.
    custom_modules_hooks={},
):
    assert len(input) >= 1
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose, ignore_list=ignore_modules)
    _ = flops_model(*input)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def flops_to_string(flops, units: str = "GMac", precision: int = 2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GMac"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MMac"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KMac"
        else:
            return str(flops) + " Mac"
    else:
        if units == "GMac":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MMac":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KMac":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " Mac"


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def params_to_string(params_num, units=None, precision: int = 2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(
    # pyre-fixme[2]: Parameter must be annotated.
    model,
    total_flops: int,
    # pyre-fixme[2]: Parameter must be annotated.
    total_params,
    units: str = "GMac",
    precision: int = 3,
    ost: TextIO = sys.stdout,
) -> None:
    if total_flops < 1:
        total_flops = 1

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    # pyre-fixme[53]: Captured variable `model` is not annotated.
    # pyre-fixme[53]: Captured variable `total_params` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        return ", ".join(
            [
                params_to_string(
                    accumulated_params_num, units="M", precision=precision
                ),
                "{:.3%} Params".format(accumulated_params_num / total_params),
                flops_to_string(
                    accumulated_flops_cost, units=units, precision=precision
                ),
                "{:.3%} MACs".format(accumulated_flops_cost / total_flops),
                self.original_extra_repr(),
            ]
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def del_extra_repr(m):
        if hasattr(m, "original_extra_repr"):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, "accumulate_flops"):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


# pyre-fixme[2]: Parameter must be annotated.
def get_model_parameters_number(model) -> int:
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module
    )

    net_main_module.reset_flops_count()

    return net_main_module


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, "accumulate_flops"):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


# pyre-fixme[2]: Parameter must be annotated.
def start_flops_count(self, **kwargs) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    # pyre-fixme[53]: Captured variable `seen_types` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, "__flops_handle__"):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)]
                )
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if (
                verbose
                and not type(module) in (nn.Sequential, nn.ModuleList)
                and not type(module) in seen_types
            ):
                print(
                    "Warning: module "
                    + type(module).__name__
                    + " is treated as a zero-op.",
                    file=ost,
                )
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


# pyre-fixme[2]: Parameter must be annotated.
def stop_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


# pyre-fixme[2]: Parameter must be annotated.
def reset_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
# pyre-fixme[2]: Parameter must be annotated.
def empty_flops_counter_hook(module, input, output) -> None:
    module.__flops__ += 0


# pyre-fixme[2]: Parameter must be annotated.
def upsample_flops_counter_hook(module, input, output) -> None:
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


# pyre-fixme[2]: Parameter must be annotated.
def relu_flops_counter_hook(module, input, output) -> None:
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


# pyre-fixme[2]: Parameter must be annotated.
def linear_flops_counter_hook(module, input, output) -> None:
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


# pyre-fixme[2]: Parameter must be annotated.
def pool_flops_counter_hook(module, input, output) -> None:
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


# pyre-fixme[2]: Parameter must be annotated.
def bn_flops_counter_hook(module, input, output) -> None:
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


# pyre-fixme[2]: Parameter must be annotated.
def conv_flops_counter_hook(conv_module, input, output) -> None:
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    )

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


# pyre-fixme[2]: Parameter must be annotated.
def batch_counter_hook(module, input, output) -> None:
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print(
            "Warning! No positional inputs found for a module,"
            " assuming batch size is 1."
        )
    module.__batch_counter__ += batch_size


# pyre-fixme[2]: Parameter must be annotated.
def rnn_flops(flops: int, rnn_module, w_ih, w_hh, input_size) -> int:
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard product and add for C state
        flops += (
            rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        )
        # final hadamard
        flops += (
            rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        )
    return flops


# pyre-fixme[2]: Parameter must be annotated.
def rnn_flops_counter_hook(rnn_module, input, output) -> None:
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


# pyre-fixme[2]: Parameter must be annotated.
def rnn_cell_flops_counter_hook(rnn_cell_module, input, output) -> None:
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


# pyre-fixme[2]: Parameter must be annotated.
def multihead_attention_counter_hook(multihead_attention_module, input, output) -> None:
    flops = 0
    q, k, v = input
    batch_size = q.shape[1]

    num_heads = multihead_attention_module.num_heads
    embed_dim = multihead_attention_module.embed_dim
    kdim = multihead_attention_module.kdim
    vdim = multihead_attention_module.vdim
    if kdim is None:
        kdim = embed_dim
    if vdim is None:
        vdim = embed_dim

    # initial projections
    flops = (
        q.shape[0] * q.shape[2] * embed_dim
        + k.shape[0] * k.shape[2] * kdim
        + v.shape[0] * v.shape[2] * vdim
    )
    if multihead_attention_module.in_proj_bias is not None:
        flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

    # attention heads: scale, matmul, softmax, matmul
    head_dim = embed_dim // num_heads
    head_flops = (
        q.shape[0] * head_dim
        + head_dim * q.shape[0] * k.shape[0]
        + q.shape[0] * k.shape[0]
        + q.shape[0] * k.shape[0] * head_dim
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += q.shape[0] * embed_dim * (embed_dim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


# pyre-fixme[2]: Parameter must be annotated.
def add_batch_counter_variables_or_reset(module) -> None:
    module.__batch_counter__ = 0


# pyre-fixme[2]: Parameter must be annotated.
def add_batch_counter_hook_function(module) -> None:
    if hasattr(module, "__batch_counter_handle__"):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


# pyre-fixme[2]: Parameter must be annotated.
def remove_batch_counter_hook_function(module) -> None:
    if hasattr(module, "__batch_counter_handle__"):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


# pyre-fixme[2]: Parameter must be annotated.
def add_flops_counter_variable_or_reset(module) -> None:
    if is_supported_instance(module):
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


# pyre-fixme[5]: Global expression must be annotated.
CUSTOM_MODULES_MAPPING = {}

# pyre-fixme[5]: Global expression must be annotated.
MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,
    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,
}


# pyre-fixme[2]: Parameter must be annotated.
def is_supported_instance(module) -> bool:
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


# pyre-fixme[2]: Parameter must be annotated.
def remove_flops_counter_hook_function(module) -> None:
    if is_supported_instance(module):
        if hasattr(module, "__flops_handle__"):
            module.__flops_handle__.remove()
            del module.__flops_handle__
