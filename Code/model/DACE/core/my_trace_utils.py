"""Custom trajectory inference utilities."""

from .trace_utils import TraceState, _check_zero_width, _execute_prediction, get_observation


def connect_traj_points(traj):
    """Connect consecutive trajectory points with ordered Bresenham pixels."""
    if not traj:
        return []

    first_x, first_y = traj[0]
    connected_traj = [(int(round(first_x)), int(round(first_y)))]
    for start, end in zip(traj, traj[1:]):
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(end[0])), int(round(end[1]))
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        while True:
            point = (x0, y0)
            if point != connected_traj[-1]:
                connected_traj.append(point)
            if point == (x1, y1):
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x0 += sx
            if doubled_error <= dx:
                error += dx
                y0 += sy

    return connected_traj


def my_trace_point(env, initial_traj):
    """Continue tracing from an ordered multi-point trajectory."""
    if len(initial_traj) == 0:
        raise ValueError("initial_traj must contain at least one point")

    args = env.args
    traj = [(float(point[0]), float(point[1])) for point in initial_traj]
    state = TraceState(
        traj=traj,
        h=env.image.shape[0],
        w=env.image.shape[1],
        width=[2.0] * len(traj),
    )

    env.model.eval()
    while state.active and len(state.traj) < args.EXEC_STEPS:
        patch_info = get_observation(env.image, state.traj, patch_size=args.PATCH_SIZE, args=args)
        prediction = env.model.forward_infer(patch_info["patch_obs"], args.USE_WIDTH)
        _execute_prediction(
            state,
            prediction.squeeze(0),
            patch_info["patch_o"],
            args.ACTION_CHUNK_SIZE,
        )
        _check_zero_width(state, getattr(args, "MAX_ZERO_WIDTH_STEPS", 1))

    return connect_traj_points(state.traj)
