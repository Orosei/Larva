from dataclasses import dataclass
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_NAME = "DRIVE"
MODEL_FILES_BY_MODEL_NAME = {
    "v3_curve_tiny_unet": {
        "model_file": "curve_tiny_unet_multitask.py",
        "weight_file": "best_connectivity_predictor_v3.pth",
    },
}


@dataclass(frozen=True)
class ModelWeightPaths:
    dataset_name: str
    model_name: str
    model_path: Path
    weight_path: Path


def _find_named_child(parent: Path, name: str, kind: str) -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"{kind} root not found: {parent}")

    exact_path = parent / name
    if exact_path.exists():
        return exact_path

    lower_name = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == lower_name:
            return child

    available = ", ".join(sorted(child.name for child in parent.iterdir() if child.is_dir())) or "<none>"
    raise FileNotFoundError(f"{kind} folder not found: {name!r}; available: {available}")


def get_model_weight_paths(
    dataset_name: str | None = None,
    model_name: str | None = None,
    model_file: str | None = None,
    weight_file: str | None = None,
) -> ModelWeightPaths:
    dataset_name = dataset_name or DEFAULT_DATASET_NAME
    model_name = model_name or next(iter(MODEL_FILES_BY_MODEL_NAME))

    model_dir = _find_named_child(MODEL_ROOT / "connectivity", str(model_name), "model")
    weight_dir = _find_named_child(model_dir / "weight", str(dataset_name), "dataset weight")

    try:
        file_config = MODEL_FILES_BY_MODEL_NAME[model_dir.name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_FILES_BY_MODEL_NAME)) or "<none>"
        raise KeyError(f"model files are not registered for {model_dir.name!r}; available: {available}") from exc

    model_file = model_file or file_config["model_file"]
    weight_file = weight_file or file_config["weight_file"]
    model_path = model_dir / model_file
    weight_path = weight_dir / weight_file

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"weight file not found: {weight_path}")

    return ModelWeightPaths(
        dataset_name=weight_dir.name,
        model_name=model_dir.name,
        model_path=model_path,
        weight_path=weight_path,
    )


__all__ = [
    "ModelWeightPaths",
    "DEFAULT_DATASET_NAME",
    "MODEL_FILES_BY_MODEL_NAME",
    "get_model_weight_paths",
]
