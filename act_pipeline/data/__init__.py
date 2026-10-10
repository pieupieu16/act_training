"""Dataset utilities; import PyTorch-backed modules only when requested."""

from importlib import import_module

__all__ = [
    "preprocess_rgbd",
    "compute_norm_stats",
    "save_norm_stats",
    "load_norm_stats",
    "normalize_data",
    "unnormalize_data",
    "BimanualEpisodicDataset",
]

_MODULES = {
    "preprocess_rgbd": "act_pipeline.data.preprocess",
    "compute_norm_stats": "act_pipeline.data.normalization",
    "save_norm_stats": "act_pipeline.data.normalization",
    "load_norm_stats": "act_pipeline.data.normalization",
    "normalize_data": "act_pipeline.data.normalization",
    "unnormalize_data": "act_pipeline.data.normalization",
    "BimanualEpisodicDataset": "act_pipeline.data.dataset",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name]), name)
    globals()[name] = value
    return value
