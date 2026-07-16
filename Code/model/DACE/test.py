import os
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "Code.model.DACE"

from .core.loader_test import get_file_paths, load_image, load_point
from .core.metrics import TraceMetrics
from .core.trace_utils import (
    TraceEnv,
    load_ckpt,
    load_config,
    resolve_indices,
    trace_point,
    trace_point_parallel,
)
from .core.visualizer import TraceVisualizer


def test_all(args, model=None):
    model = model or load_ckpt(args)
    image_paths, label_paths = get_file_paths(args)
    summary = TraceMetrics(base_name=args.NAME)

    parallel = getattr(args, "PARALLEL", False)
    for idx in resolve_indices(args.IMAGE_IDX, len(image_paths)):
        metrics = _test_image(args, model, image_paths[idx], label_paths[idx], parallel=parallel)
        summary.append(metrics)
    summary.report(args.SAVE_DIR, show_avg=True)


def _test_image(args, dace_model, image_path, label_path=None, parallel=False):
    image, label, image_bgr = load_image(image_path, label_path)
    start_pts = load_point(args, image_path, image_bgr)

    limit = args.POINTS_PER_IMAGE
    if limit != -1:
        start_pts = start_pts[:limit]

    if not start_pts:
        print(f"No start points for {image_path}; skipped.")
        return None

    visualizer = TraceVisualizer(
        image_bgr,
        label,
        args.SAVE_DIR,
        image_path,
        args.WRITE_VIDEO,
        args.SHOW_GUI,
        parallel=parallel,
    )
    metrics = TraceMetrics(label, base_name=Path(image_path).stem)
    env = TraceEnv(args, dace_model, image, label, visualizer)

    if parallel:
        t0 = time.time()
        trajs = trace_point_parallel(env, start_pts)
        for traj in trajs:
            metrics.eval(traj, t0)
    else:
        for pt in start_pts:
            t0 = time.time()
            traj = trace_point(env, pt)
            metrics.eval(traj, t0)

    metrics.report(args.SAVE_DIR)
    visualizer.close()
    return metrics


if __name__ == "__main__":
    default_config_path = Path(__file__).resolve().parent / "config" / "test.json"
    config_path = os.environ.get("DACE_TEST_CONFIG", default_config_path)
    args = load_config(config_path)
    test_all(args)
