import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


@dataclass
class TraceEnv:
    args: object
    model: object
    image: np.ndarray
    label: np.ndarray | None
    visualizer: object | None


@dataclass
class TraceState:
    traj: list
    h: int
    w: int
    width: list = None
    reason: str = "healthy"
    active: bool = True


def load_config(config_path="config.json"):
    config_path = os.path.abspath(os.path.expanduser(os.fspath(config_path)))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    config_dir = os.path.dirname(config_path)
    path_keys = (
        "DATA_ROOT",
        "IMAGE_DIR",
        "LABEL_DIR",
        "PT_DIR",
        "OUTPUT_DIR",
        "SAVE_DIR",
        "WEIGHTS_FILE_NAME",
    )
    for key in path_keys:
        value = cfg.get(key)
        if value and not os.path.isabs(value):
            cfg[key] = os.path.normpath(os.path.join(config_dir, value))

    cfg["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = cfg.get("DATA_ROOT")
    if root:
        cfg.setdefault("IMAGE_DIR", os.path.join(root, "images"))
        cfg.setdefault("LABEL_DIR", os.path.join(root, "labels"))
        cfg.setdefault("PT_DIR", os.path.join(root, "pts"))
    cfg.setdefault("SAVE_DIR", os.path.join(cfg["OUTPUT_DIR"], cfg["DATA_NAME"] + "_" + cfg["NAME"]))

    class Config:
        def __init__(self, d):
            self.__dict__ = d

    return Config(cfg)


def _print_params(model, name, weights_path=None):
    if weights_path:
        print(f"\U0001f4e6 Model: {weights_path}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model   Params(M)   Trainable(M)")
    print(f"{name}    {total / 1e6:<12.1f} {trainable / 1e6:<14.1f}")
    return total, trainable


def _build_scheduler(optimizer, total):
    warmup = 10
    warm_sch = LambdaLR(optimizer, lambda e: (e + 1) / warmup)
    cos_sch = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warm_sch, cos_sch], milestones=[warmup])


def _load_optim(model, args, ckpt):
    opt = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    sch = _build_scheduler(opt, total=args.EPOCHS)
    epoch = 1
    if ckpt is not None:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sch.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            epoch = ckpt["epoch"] + 1
    return opt, sch, epoch


def load_ckpt(args, test=True):
    from .model_dace import DaceNet

    model = DaceNet(args).to(args.DEVICE)
    ckpt = None
    if os.path.exists(args.WEIGHTS_FILE_NAME):
        ckpt = torch.load(args.WEIGHTS_FILE_NAME, map_location=args.DEVICE)
        model.load_state_dict(ckpt["model"], strict=False)
    _print_params(model, args.NAME, args.WEIGHTS_FILE_NAME)

    if test:
        return model
    return model, *_load_optim(model, args, ckpt)


def save_ckpt(path, model, epoch=None, optimizer=None, scheduler=None):
    ckpt = {"model": model.state_dict()}
    if epoch is not None:
        ckpt["epoch"] = epoch
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)
    return path


def traj_overlay(canvas, traj, args=None):
    if args.USE_OVERLAY:
        color = args.OVERLAY_COLOR
        thickness = args.OVERLAY_THICKNESS
        cv2.polylines(canvas, [np.array(traj, dtype=np.int32)], False, color, thickness)


def _init_traj(img, start_pt, seed_len=7, radius=16):
    cx, cy = int(round(start_pt[0])), int(round(start_pt[1]))
    h, w = img.shape[:2]
    fail = [start_pt, start_pt]

    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    roi = img[y1:y2, x1:x2]
    if roi.shape[0] < 5 or roi.shape[1] < 5 or np.std(roi) < 1e-4:
        return fail

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    Sxx, Syy, Sxy = np.sum(gx * gx), np.sum(gy * gy), np.sum(gx * gy)
    trace_val = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    disc = trace_val * trace_val - 4.0 * det
    if disc <= 0:
        return fail

    lambda2 = (trace_val - np.sqrt(disc)) / 2.0
    if abs(Sxy) > 1e-8:
        vx, vy = lambda2 - Syy, Sxy
    else:
        vx, vy = (1.0, 0.0) if Sxx < Syy else (0.0, 1.0)
    norm = np.hypot(vx, vy)
    if norm < 1e-8:
        return fail
    vx, vy = vx / norm, vy / norm

    ends = np.array([[cx + sign * vx * seed_len, cy + sign * vy * seed_len] for sign in (1, -1)])
    ends = np.round(ends).astype(int)
    valid = (0 <= ends[:, 0]) & (ends[:, 0] < w) & (0 <= ends[:, 1]) & (ends[:, 1] < h)
    if not valid.any():
        return fail

    intensities = img[ends[valid, 1], ends[valid, 0]]
    seed_end = tuple(ends[valid][np.argmin(intensities)].tolist())
    return [start_pt, seed_end]


def _execute_prediction(state, pred_chunk, origin, action_chunk_size):
    h, w = state.h, state.w
    chunk = pred_chunk[1 : action_chunk_size + 1]
    has_width = pred_chunk.shape[-1] >= 3

    coords = chunk[:, :2] + origin
    widths = chunk[:, 2] if has_width else torch.full((len(chunk),), 2.0, device=chunk.device)

    for i in range(len(coords)):
        x, y = coords[i]
        if not (0 <= x < w and 0 <= y < h):
            state.reason = "out of boundary"
            state.active = False

        w_val = widths[i].item()
        state.traj.append((float(x), float(y)))
        state.width.append(w_val)


def _check_zero_width(state, max_steps):
    zero_total = sum(1 for w in state.width if w < 1)
    zero_max_consec = 0
    cur = 0
    for w in state.width:
        if w < 1:
            cur += 1
            zero_max_consec = max(zero_max_consec, cur)
        else:
            cur = 0

    if zero_total >= 2 + max_steps or zero_max_consec >= max_steps:
        state.reason = "width is 0"
        state.active = False
        print(f"\033[31m❌ 累计{zero_total}步/连续{zero_max_consec}步宽度<1, 追踪{len(state.traj)}步\033[0m")


def pad_traj_to_pred_steps(target, args, H=None, W=None):
    target = np.asarray(target, dtype=np.float32)
    cur_len = len(target)
    need = getattr(args, "PRED_STEPS", 16) - cur_len
    if need <= 0:
        return target

    # Snap direction to nearest integer 8-direction step
    if cur_len >= 2:
        dx = target[-1, 0] - target[-2, 0]
        dy = target[-1, 1] - target[-2, 1]
    else:
        dx, dy = 1.0, 0.0
    m = max(abs(dx), abs(dy))
    if m < 1e-8:
        sx, sy = 1, 0
    else:
        sx = int(round(dx / m))
        sy = int(round(dy / m))
    if sx == 0 and sy == 0:
        sx = 1

    has_width = target.shape[1] >= 3

    ext = []
    for i in range(1, need + 1):
        nx = target[-1, 0] + sx * i
        ny = target[-1, 1] + sy * i
        if W is not None:
            nx = np.clip(nx, 0, W - 1)
        if H is not None:
            ny = np.clip(ny, 0, H - 1)
        if has_width:
            ext.append([nx, ny, 0.0])  # width < 1
        else:
            ext.append([nx, ny])

    return np.concatenate([target, np.array(ext, dtype=np.float32)], axis=0)


def render_heatmaps_cpu(delta, H, sigma=1.5):
    T = delta.shape[0]
    gy, gx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="ij",
    )
    gauss_list = []

    for t in range(T):
        xc = delta[t, 0]
        yc = delta[t, 1]
        dist2 = (gx - xc) ** 2 + (gy - yc) ** 2

        if delta.shape[1] < 3:
            s = sigma
            g = np.exp(-dist2 / (2 * s**2))
        else:
            w = delta[t, 2]
            if w < 0.5:
                g = np.zeros((H, H), dtype=np.float32)
            else:
                s = max(w / 2.0, 0.5)
                g = np.exp(-dist2 / (2 * s**2))
        gauss_list.append(g)

    gauss = np.stack(gauss_list, axis=0)
    return gauss.astype(np.float32)


def _crop_patch(src, ox, oy, patch_size, border=0):
    """Crop a patch_size × patch_size window at (ox, oy) from src, padding with border."""
    h, w = src.shape[:2]
    patch = np.full((patch_size, patch_size) + src.shape[2:], border, dtype=np.float32)
    sx1, sy1 = max(0, ox), max(0, oy)
    sx2, sy2 = min(w, ox + patch_size), min(h, oy + patch_size)
    if sx1 < sx2 and sy1 < sy2:
        patch[sy1 - oy : sy2 - oy, sx1 - ox : sx2 - ox] = src[sy1:sy2, sx1:sx2]
    return patch


def pt_shift(point, sigma=2, max_shift=4):
    shift_x = np.clip(int(np.round(np.random.normal(0, sigma))), -max_shift, max_shift)
    shift_y = np.clip(int(np.round(np.random.normal(0, sigma))), -max_shift, max_shift)
    return int(point[0]) + shift_x, int(point[1]) + shift_y


def get_observation(img, traj, patch_size, curve_mask=None, width_map=None, target=None, args=None):
    center = traj[-1]
    max_shift = getattr(args, "PT_SHIFT", 0)
    if max_shift > 0:
        center = pt_shift(center, max_shift=max_shift)

    ox = int(round(center[0])) - patch_size // 2
    oy = int(round(center[1])) - patch_size // 2

    # Local trajectory coordinates (vectorized)
    traj_arr = np.asarray(traj, dtype=np.float32)
    local_traj = traj_arr - np.float32((ox, oy))
    canvas = np.zeros((patch_size, patch_size), dtype=np.float32)
    traj_overlay(canvas, local_traj, args)

    img_patch = _crop_patch(img, ox, oy, patch_size)
    img_patch = (img_patch - img_patch.mean()) / (img_patch.std() + 1e-6)
    patch_obs = np.stack([img_patch, canvas], axis=0)

    sample = {"patch_obs": patch_obs, "patch_o": (ox, oy)}

    if target is not None:
        target = np.asarray(target, dtype=np.float32)
        delta = target - np.float32((ox, oy))
        if getattr(args, "USE_WIDTH", False) and width_map is not None:
            target = pad_traj_to_pred_steps(target, args, H=img.shape[0], W=img.shape[1])
            tw = width_map[target[:, 1].astype(int), target[:, 0].astype(int)]
            delta = target - np.float32((ox, oy))
            delta = np.column_stack([delta, tw])

        sample["traj_gt"] = delta
        sample["patch_curve_mask"] = _crop_patch(curve_mask, ox, oy, patch_size)
        sample["patch_width"] = _crop_patch(width_map, ox, oy, patch_size)
        sample["traj_gt_heatmaps"] = render_heatmaps_cpu(delta, patch_size)

        # vis_sample(sample, wait=True, window="sample")
    return sample


def vis_sample(sample, wait=True, window="sample"):
    import cv2

    ps = sample["patch_obs"].shape[-1]  # patch_size

    # ---- row 0: img_patch | canvas | traj_gt on img ----
    img_patch = sample["patch_obs"][0]
    img_disp = (np.clip(img_patch * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

    canvas = sample["patch_obs"][1]
    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min() + 1e-8)
    canvas_disp = (canvas * 255).astype(np.uint8)
    canvas_disp = cv2.cvtColor(canvas_disp, cv2.COLOR_GRAY2BGR)

    # img with traj_gt overlay
    traj_img = img_disp.copy()
    if "traj_gt" in sample:
        gt = sample["traj_gt"]
        pts = gt[:, :2].astype(np.int32)
        if gt.shape[1] >= 3:
            widths = gt[:, 2]
            mask_normal = widths >= 1
            mask_small = widths < 1
            for px, py in gt[mask_normal, :2].astype(np.int32):
                traj_img[py, px] = (255, 0, 0)
            for px, py in gt[mask_small, :2].astype(np.int32):
                traj_img[py, px] = (0, 0, 255)
        else:
            cv2.polylines(traj_img, [pts], False, (0, 255, 0), 1)

    row0 = np.hstack([img_disp, canvas_disp, traj_img])

    # ---- row 1: heatmap | curve_mask | width_map ----
    row1_parts = []
    for key in ("patch_curve_mask", "patch_width"):
        if key in sample:
            data = sample[key]
            if data.ndim == 3:
                data = data[0]
            v = (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            v = np.zeros((ps, ps), dtype=np.float32)
        disp = (v * 255).astype(np.uint8)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        row1_parts.append(disp)

    row1_parts.append(np.zeros_like(row1_parts[0]))
    row1 = np.hstack(row1_parts)

    vis = np.vstack([row0, row1])

    labels = ["img", "canvas", "gt_traj", "curve", "width", ""]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, lbl in enumerate(labels):
        x = i * ps + 4
        cv2.putText(vis, lbl, (x, 14), font, 0.4, (0, 255, 255), 1)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, vis)
    if wait:
        cv2.waitKey(0)
    return vis


def trace_point(env, start_pt):
    args = env.args
    traj = _init_traj(env.image, start_pt)
    state = TraceState(traj, env.image.shape[0], env.image.shape[1], width=[2] * len(traj))
    metrics = None
    if env.label is not None:
        from .metrics import TraceMetrics

        metrics = TraceMetrics(env.label)

    env.model.eval()
    while state.active and len(state.traj) < args.EXEC_STEPS:
        t0 = time.time() if metrics is not None else None

        patch_info = get_observation(env.image, state.traj, patch_size=args.PATCH_SIZE, args=args)
        prediction = env.model.forward_infer(patch_info["patch_obs"], args.USE_WIDTH)
        _execute_prediction(state, prediction.squeeze(0), patch_info["patch_o"], args.ACTION_CHUNK_SIZE)
        _check_zero_width(state, getattr(args, "MAX_ZERO_WIDTH_STEPS", 1))
        if metrics is not None:
            metrics.eval(state.traj, t0)

        if env.visualizer is not None:
            env.visualizer.update_display(state.traj, state.reason, metrics, state.width)
    return state.traj


def trace_point_parallel(env, start_pts):
    args = env.args
    states = []
    for pt in start_pts:
        traj = _init_traj(env.image, pt)
        states.append(TraceState(traj, env.image.shape[0], env.image.shape[1], width=[2] * len(traj)))

    env.model.eval()
    while any(s.active and len(s.traj) < args.EXEC_STEPS for s in states):
        patches, origins, idxs = [], [], []
        for i, s in enumerate(states):
            if not s.active:
                continue
            info = get_observation(env.image, s.traj, patch_size=args.PATCH_SIZE, args=args)
            patches.append(info["patch_obs"])
            origins.append(info["patch_o"])
            idxs.append(i)

        if not patches:
            break

        predictions = env.model.forward_infer(np.stack(patches, axis=0), args.USE_WIDTH)

        for i, pred, origin in zip(idxs, predictions, origins):
            _execute_prediction(states[i], pred, origin, args.ACTION_CHUNK_SIZE)
            _check_zero_width(states[i], getattr(args, "MAX_ZERO_WIDTH_STEPS", 1))
        env.visualizer.display_all([s.traj for s in states], [s.width for s in states])
    return [s.traj for s in states]


def resolve_indices(indices, total):
    if indices == -1:
        return list(range(total))
    if isinstance(indices, int):
        indices = list(range(min(indices, total)))
    return [i for i in indices if 0 <= i < total]
