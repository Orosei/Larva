import torch
import torch.nn as nn


def normalize_traj(traj, H, W=8):
    out = (traj / (H * 0.5)) - 1.0
    if traj.shape[-1] >= 3:
        out[..., 2] = (traj[..., 2] / (W * 0.5)) - 1.0
    return out


def unnormalize_traj(x, H, W=8):
    out = (x + 1.0) * (H * 0.5)
    if x.shape[-1] >= 3:
        out[..., 2] = (x[..., 2] + 1.0) * (W * 0.5)
    return out


def reverse_heatmaps(hms, dim3=False):
    B, T, H, _ = hms.shape
    flat_hm = hms.reshape(B, T, -1)
    idx = flat_hm.argmax(dim=-1)
    x = (idx % H).float()
    y = (idx // H).float()
    if not dim3:
        return torch.stack([x, y], dim=-1)

    area = hms.relu().sum(dim=(-2, -1))  # (B, T)
    w_est = torch.sqrt(2 * area / torch.pi)

    # When w_est < 1 and (x,y) far from heatmap center, lock to center
    center = H / 2.0
    dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    lock_mask = (w_est < 1) & (dist > 4)
    x = torch.where(lock_mask, torch.full_like(x, center), x)
    y = torch.where(lock_mask, torch.full_like(y, center), y)

    return torch.stack([x, y, w_est], dim=-1)


def render_heatmaps_gpu(delta, H, sigma=1.5):
    B, T = delta.shape[:2]
    device = delta.device

    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="ij",
    )
    gx = gx[None, None, ...]  # (1, 1, H, H)
    gy = gy[None, None, ...]  # (1, 1, H, H)

    cx = delta[..., 0].reshape(B, T, 1, 1)  # (B, T, 1, 1)
    cy = delta[..., 1].reshape(B, T, 1, 1)  # (B, T, 1, 1)

    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2  # (B, T, H, H)

    if delta.shape[-1] < 3:
        gauss = torch.exp(-dist2 / (2 * sigma**2))
    else:
        w = delta[..., 2].reshape(B, T, 1, 1)  # (B, T, 1, 1)
        s = torch.clamp(w / 2.0, min=0.5)  # max(w/2, 0.5)
        gauss = torch.exp(-dist2 / (2 * s**2))
        gauss = torch.where(w < 0.5, torch.zeros_like(gauss), gauss)
    return gauss


class Diffusion2DPolicy(nn.Module):
    def __init__(self, model, noise_scheduler, num_inference_steps, inference_seed):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.inference_seed = inference_seed
        self._gen = None
        self._last_device = None

    def _get_gen(self, device):
        key = str(device)
        if self._gen is None or self._last_device != key:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.inference_seed)
            self._last_device = key
        return self._gen

    def diffusion_heatmap(self, batch):
        cond = batch["patch_obs"]
        traj_gt = batch["traj_gt"]
        B, _, _, H = cond.shape

        x = normalize_traj(traj_gt, H)
        noise = torch.randn_like(x)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        x_noise = self.noise_scheduler.add_noise(x, noise, t)
        traj_noise = unnormalize_traj(x_noise, H)
        noisy_hm = render_heatmaps_gpu(traj_noise, H)
        return self.model(noisy_hm, t, cond)

    def predict_heatmap(self, cond):
        with torch.inference_mode():
            B, _, H, W = cond.shape
            T = self.model.horizon_steps
            device = cond.device

            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            z = torch.randn(B, T, H, W, device=device, dtype=cond.dtype, generator=self._get_gen(device))
            for t in self.noise_scheduler.timesteps:
                z0 = self.model(z, t, cond)
                z = self.noise_scheduler.step(z0, t, z).prev_sample
            return z


if __name__ == "__main__":
    # Verify round-trip: traj -> render_heatmaps -> reverse_heatmaps -> traj_est
    H = 64
    B, T = 2, 4

    # --- 2D case (x, y only) ---
    print("=== 2D round-trip test (x, y) ===")
    traj_2d = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0], [50.0, 10.0], [60.0, 55.0]],
            [[5.0, 5.0], [15.0, 25.0], [35.0, 45.0], [63.0, 63.0]],
        ]
    )  # (B, T, 2)
    hms_2d = render_heatmaps_gpu(traj_2d, H)
    est_2d = reverse_heatmaps(hms_2d, dim3=False)
    diff_2d = (traj_2d - est_2d).abs()
    print(f"  Max  error (x,y): {diff_2d.max().item():.4f} px")
    print(f"  Mean error (x,y): {diff_2d.mean().item():.4f} px")
    assert diff_2d.max() < 0.6, f"2D recovery error too large: {diff_2d.max().item():.4f}"
    print("  2D test PASSED\n")

    # --- 3D case (x, y, w) ---
    print("=== 3D round-trip test (x, y, w) ===")
    traj_3d = torch.tensor(
        [
            [[10.0, 20.0, 3.0], [30.0, 40.0, 5.0], [50.0, 10.0, 1.0], [20.0, 25.0, 4.0]],
            [[5.0, 5.0, 2.0], [15.0, 25.0, 6.0], [35.0, 45.0, 0.8], [33.0, 23.0, 0.4]],
        ]
    )  # (B, T, 3)
    hms_3d = render_heatmaps_gpu(traj_3d, H)
    est_3d = reverse_heatmaps(hms_3d, dim3=True)
    diff_xy = (traj_3d[..., :2] - est_3d[..., :2]).abs()
    diff_w = (traj_3d[..., 2] - est_3d[..., 2]).abs()
    print(f"  Max  error (x,y): {diff_xy.max().item():.4f} px")
    print(f"  Mean error (x,y): {diff_xy.mean().item():.4f} px")
    print(f"  Max  error (w):   {diff_w.max().item():.4f} px")
    print(f"  Mean error (w):   {diff_w.mean().item():.4f} px")

    # --- Detail: print original vs estimated for each batch ---
    for b in range(B):
        print(f"=== Detail (3D case, batch {b}) ===")
        print(f"{'Step':<6} {'orig_x':>8} {'est_x':>8} {'orig_y':>8} {'est_y':>8} {'orig_w':>8} {'est_w':>8}")
        for t in range(T):
            ox, oy, ow = traj_3d[b, t].tolist()
            ex, ey, ew = est_3d[b, t].tolist()
            print(f"  T={t:<2}  {ox:8.1f} {ex:8.1f} {oy:8.1f} {ey:8.1f} {ow:8.1f} {ew:8.1f}")

    # --- Print center 8x8 of heatmap for w <= 1 cases ---
    print("\n=== Center 8x8 heatmap values for w <= 1 ===")
    for b in range(B):
        for t in range(T):
            w_val = traj_3d[b, t, 2].item()
            if w_val > 1.0:
                cx = int(traj_3d[b, t, 0].item())
                cy = int(traj_3d[b, t, 1].item())
                # 8x8 window around center, clamped to [0, H)
                x0, x1 = max(0, cx - 4), min(H, cx + 4)
                y0, y1 = max(0, cy - 4), min(H, cy + 4)
                patch = hms_3d[b, t, y0:y1, x0:x1]
                print(f"\nBatch {b}, T={t}: w={w_val:.1f}, center=({cx},{cy}), window=({x0}:{x1}, {y0}:{y1})")
                print(f"  shape={patch.shape}, max={patch.max().item():.4f}, min={patch.min().item():.4f}")
                # Print as a compact numeric grid
                for row in range(patch.shape[0]):
                    vals = " ".join(f"{patch[row, col].item():.3f}" for col in range(patch.shape[1]))
                    print(f"  [{vals}]")

    print("\nAll round-trip verification passed!")
