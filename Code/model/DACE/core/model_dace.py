import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn

from .conditional_unet2D import ConditionalUnet2D
from .diffusion_2d_policy import Diffusion2DPolicy, reverse_heatmaps


class DaceNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lambda_inside = float(getattr(args, "LAMBDA_INSIDE", 1.0))
        self.hm_loss_weight = float(getattr(args, "HM_LOSS_WEIGHT", 50.0))
        self.device = args.DEVICE

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=int(getattr(args, "NUM_DIFFUSION_STEPS", 1000)),
            clip_sample=True,
            prediction_type="sample",
        )

        self.unet = ConditionalUnet2D(
            horizon_steps=int(getattr(args, "PRED_STEPS", 16)),
            cond_channels=2,
            down_dims=[64, 128, 256],
            cond_predict_scale=True,
        ).to(self.device)

        self.policy = Diffusion2DPolicy(
            model=self.unet,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=int(getattr(args, "INFERENCE_STEPS", 15)),
            inference_seed=int(getattr(args, "INFERENCE_SEED", 42)),
        ).to(self.device)

    def forward_train(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        hm = self.policy.diffusion_heatmap(batch)
        heatmap_loss = self.compute_hm_loss(hm, batch["traj_gt_heatmaps"])
        inside_loss = self.lambda_inside * self.compute_inside_loss(hm, batch["patch_curve_mask"])

        return {
            "heatmap_loss": heatmap_loss,
            "inside_loss": inside_loss,
            "total_loss": heatmap_loss + inside_loss,
        }

    def forward_infer(self, patch_obs, dim3):
        with torch.inference_mode():
            x = torch.as_tensor(patch_obs, dtype=torch.float32, device=self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)

            hm = self.policy.predict_heatmap(x)
            traj = reverse_heatmaps(hm, dim3)
            return traj.cpu().numpy()

    def compute_inside_loss(self, pred_hm, mask):
        penalty = torch.relu(pred_hm) * (1.0 - mask.unsqueeze(1))
        return penalty.mean()

    def compute_hm_loss(self, pred_hm, gt_hm):
        return ((pred_hm - gt_hm) ** 2 * (1.0 + gt_hm * self.hm_loss_weight)).mean()
