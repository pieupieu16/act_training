"""OpenArm ACT package. Heavy training dependencies are imported on demand."""

from importlib import import_module

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "ACTPolicy",
    "BimanualEpisodicDataset",
    "preprocess_rgbd",
    "compute_norm_stats",
    "save_norm_stats",
    "load_norm_stats",
    "MetricLogger",
    "CheckpointManager",
    "TemporalEnsemblePolicy",
]

_MODULES = {
    "ModelConfig": "act_pipeline.config",
    "TrainConfig": "act_pipeline.config",
    "EvalConfig": "act_pipeline.config",
    "ACTPolicy": "act_pipeline.models.act_model",
    "BimanualEpisodicDataset": "act_pipeline.data.dataset",
    "preprocess_rgbd": "act_pipeline.data.preprocess",
    "compute_norm_stats": "act_pipeline.data.normalization",
    "save_norm_stats": "act_pipeline.data.normalization",
    "load_norm_stats": "act_pipeline.data.normalization",
    "MetricLogger": "act_pipeline.utils.logger",
    "CheckpointManager": "act_pipeline.utils.checkpoint",
    "TemporalEnsemblePolicy": "act_pipeline.utils.temporal_ensemble",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name]), name)
    globals()[name] = value
    return value
