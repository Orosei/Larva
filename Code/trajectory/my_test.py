"""Minimal single-image DACE inference interface."""

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    __package__ = "Code.trajectory"

from ..model.DACE.core.my_trace_utils import my_trace_point
from ..model.DACE.core.trace_utils import TraceEnv, load_ckpt, resolve_indices
from .init_traj import init_traj
from .my_loader import get_my_file_paths, load_my_config, load_my_image
from .my_visualizer import TraceVisualizer
from ..utils import process_curve_v2, process_skeleton


def my_test_image(args, dace_model, image_path, label_path, coarse_seg_path):
    """Generate one complete trajectory per initialized skeleton trajectory."""
    image, label, coarse_seg, image_bgr = load_my_image(
        image_path, label_path, coarse_seg_path
    )

    ske_num, ske_mask, junc_mask = process_skeleton(coarse_seg)
    curve_mask = process_curve_v2(coarse_seg, ske_num, ske_mask)
    initial_trajs = init_traj(ske_mask, junc_mask, curve_mask)

    if not initial_trajs:
        print(f"No initial trajectories for {image_path}; skipped.")
        return []

    visualizer = TraceVisualizer(
        image_bgr,
        label,
        args.SAVE_DIR,
        image_path,
        args.WRITE_VIDEO,
        args.SHOW_GUI,
        parallel=False,
        coarse_seg=coarse_seg,
    )
    env = TraceEnv(args, dace_model, image, label, visualizer)

    trajs = []
    for _idx, _end_idx, sampled_points in initial_trajs:
        initial_traj = sampled_points[:, ::-1]
        traj = my_trace_point(env, initial_traj)
        trajs.append(traj)

        visualizer.update_display(traj)

    visualizer.close()
    return trajs


def my_test_all(config_path=None, model=None):
    """Run ``my_test_image`` for all images selected by the test configuration.

    Returns a mapping from each selected image path to its generated trajectories.
    """
    if config_path is None:
        default_config_path = Path(__file__).resolve().parent / "my_test.json"
        config_path = os.environ.get("DACE_MY_TEST_CONFIG", default_config_path)

    args = load_my_config(config_path)
    if model is None:
        model = load_ckpt(args)

    image_paths, label_paths, coarse_seg_paths = get_my_file_paths(args)
    indices = resolve_indices(args.IMAGE_IDX, len(image_paths))

    results = {}
    for index in indices:
        image_path = image_paths[index]
        results[image_path] = my_test_image(
            args,
            model,
            image_path,
            label_paths[index],
            coarse_seg_paths[index],
        )
    return results


if __name__ == "__main__":
    my_test_all()
