import numpy as np
import torch

# from .DirectionNet.direction import Direction  # fundus
from .crack.direction import Direction  # crack


def get_pred_direction_v2(predictor: Direction, pt, use_far: bool = False):
    pred = predictor.dir_pre(pt).squeeze(0).squeeze(0).cpu().detach().numpy()
    return pred[16:] if use_far else pred[:16]


def get_pred_direction_v3(predictor, pt, use_far: bool = False):
    """for crack"""
    raw_pred_tensor, raw_pred_argmax = predictor.dir_pre(pt)

    """merge 1 and 2"""
    squeezed_tensor = raw_pred_tensor.squeeze(0).squeeze(0)
    squeezed_softmax_tensor = torch.softmax(squeezed_tensor, dim=-1)
    squeezed_pred = squeezed_softmax_tensor.cpu().detach().numpy()

    val_0 = squeezed_pred[..., 0]
    val_1_2 = squeezed_pred[..., 1:3]

    merged_1_2 = val_1_2.sum(axis=-1)
    condition = merged_1_2 > val_0
    argmax_1_2 = val_1_2.argmax(axis=-1) + 1

    pred = np.where(condition, argmax_1_2, np.zeros_like(argmax_1_2))

    """raw pred argmax"""
    squeezed_argmax = raw_pred_argmax.squeeze(0).squeeze(0).cpu().detach().numpy()
    # pred = squeezed_argmax

    if use_far:
        return pred[16:], merged_1_2[16:]
    else:
        return pred[:16], merged_1_2[:16]


def agg_16_to_8(prob_16):
    prob_even = prob_16[0::2]
    prob_odd = prob_16[1::2]

    prob_8 = prob_even + 0.5 * prob_odd + 0.5 * np.roll(prob_odd, 1)
    prob_8_scaled = prob_8 / 2.0

    return prob_8_scaled
