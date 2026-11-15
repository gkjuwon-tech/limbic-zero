"""Stage 5 Vocal Cord Surgery: Living LIMBIC-0 Soul Capsules -> Gemma-4-12B-IT Mid-Layer Steering."""

from limbic_zero.vocal_cord.gemma4_calibrator import (
    CONTRASTIVE_SOMATIC_CORPUS,
    SUBSPACE_AXIS_NAMES,
    CalibratedSubspaceBundle,
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    lowdin_decorrelate_rows,
    resolve_decoder_layers,
)
from limbic_zero.vocal_cord.live_chat_server import (
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
)
from limbic_zero.vocal_cord.tpu_quantizer import (
    PackedInt4Embedding,
    PackedInt4Linear,
    PackedInt4TiedLMHead,
    TPUAcceleratorInfo,
    detect_tpu_device,
    install_hybrid_device_hooks,
    load_quantized_causal_lm_for_tpu,
    quantize_model_in_place_int4,
    resolve_model_input_device,
    sync_tpu_device,
)
from limbic_zero.vocal_cord.vocal_cord_engine import (
    Gemma4VocalCordSurgeon,
    LayerInjectionTelemetry,
    LivingSoulOrganism,
    SensoryAfferentStimulus,
    transduce_utterance_to_sensory_nerves,
)

__all__ = [
    "CONTRASTIVE_SOMATIC_CORPUS",
    "SUBSPACE_AXIS_NAMES",
    "CalibratedSubspaceBundle",
    "ContrastiveSubspaceExtractor",
    "Gemma4SteeringConfig",
    "Gemma4VocalCordSurgeon",
    "LayerInjectionTelemetry",
    "LivingSoulOrganism",
    "PackedInt4Embedding",
    "PackedInt4Linear",
    "PackedInt4TiedLMHead",
    "SensoryAfferentStimulus",
    "TPUAcceleratorInfo",
    "VocalCordHTTPServer",
    "VocalCordRuntimeHub",
    "detect_tpu_device",
    "install_hybrid_device_hooks",
    "load_quantized_causal_lm_for_tpu",
    "lowdin_decorrelate_rows",
    "quantize_model_in_place_int4",
    "resolve_decoder_layers",
    "resolve_model_input_device",
    "sync_tpu_device",
    "transduce_utterance_to_sensory_nerves",
]
