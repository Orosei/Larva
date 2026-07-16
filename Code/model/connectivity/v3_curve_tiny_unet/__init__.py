from .curve_tiny_unet_multitask import (
    CurveTinyUNetMultiTask,
    build_curve_tiny_unet_multitask,
)


def build_connectivity_predictor(cfg: dict) -> CurveTinyUNetMultiTask:
    model_type = str(cfg.get("model", {}).get("type", "curve_tiny_unet"))
    if model_type != "curve_tiny_unet":
        raise ValueError(f"curve_tiny_unet package only supports model.type='curve_tiny_unet', got {model_type!r}")
    return build_curve_tiny_unet_multitask(cfg)


__all__ = [
    "CurveTinyUNetMultiTask",
    "build_connectivity_predictor",
    "build_curve_tiny_unet_multitask",
]
