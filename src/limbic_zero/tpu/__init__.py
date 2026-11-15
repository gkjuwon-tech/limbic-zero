"""Stage 14 Google Cloud TPU v5e-8 Pure-BF16 8-Chip GSPMD Neocortical Engine."""

from limbic_zero.tpu.xla_spmd_cortex import (
    DEFAULT_PREFILL_BUCKETS,
    TPUBucketedStaticGenerator,
    TPUSPMDContext,
    XLAPinnedStaticCache,
    build_static_hybrid_masks_decode,
    build_static_hybrid_masks_prefill,
    compute_gemma4_lora_partition_specs,
    compute_gemma4_parameter_partition_spec,
    initialize_tpu_spmd_runtime,
    is_tpu_xla_available,
    resolve_kaggle_or_hf_model_source,
    sample_next_token_static,
    shard_model_to_tpu_spmd,
)

__all__ = [
    "DEFAULT_PREFILL_BUCKETS",
    "TPUBucketedStaticGenerator",
    "TPUSPMDContext",
    "XLAPinnedStaticCache",
    "build_static_hybrid_masks_decode",
    "build_static_hybrid_masks_prefill",
    "compute_gemma4_lora_partition_specs",
    "compute_gemma4_parameter_partition_spec",
    "initialize_tpu_spmd_runtime",
    "is_tpu_xla_available",
    "resolve_kaggle_or_hf_model_source",
    "sample_next_token_static",
    "shard_model_to_tpu_spmd",
]
