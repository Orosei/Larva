from ..utils import process_curve_v2, process_skeleton


def traj_process(img, label, seg_mask, img_name, iter_num=1):
    """trajectory mode"""
    seg_mask_2 = seg_mask.copy()
    ske_num, ske_mask, junc_mask = process_skeleton(seg_mask_2)
    curve_mask = process_curve_v2(seg_mask_2, ske_num, ske_mask)

    info = 1

    return