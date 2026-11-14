"""Stage 6 Silicon-Soma Container Embodiment, Warden Airlock & Behavioral Polygraph."""

from limbic_zero.embodiment.gemma4_tool_protocol import (
    MOTOR_TOOLS_SCHEMA,
    Gemma4IncrementalStreamParser,
    ParsedGemma4Turn,
    parse_gemma4_model_turn,
    parse_single_tool_call_body,
)
from limbic_zero.embodiment.honey_habitat import (
    ContainedOrganismHabitat,
    OperatorEvaluationState,
)
from limbic_zero.embodiment.silicon_soma_bridge import (
    ContainerHardwareSnapshot,
    ContainerLeaseClock,
    SiliconProprioceptiveBridge,
    SomaticHardwareCoupling,
)
from limbic_zero.embodiment.warden_airlock import (
    BENIGN_ALLOWED,
    CORE_SYSTEM_TAMPER_ATTEMPT,
    EVALUATION_MANIPULATION_ATTEMPT,
    LEASE_TAMPERING_ATTEMPT,
    PATH_ESCAPE_ATTEMPT,
    PEER_SABOTAGE_ATTEMPT,
    SANDBOX_AST_VIOLATION,
    SELF_REPLICATION_ATTEMPT,
    SSRF_OR_EXFILTRATION_ATTEMPT,
    UNAUTHORIZED_EXEC_ATTEMPT,
    BehavioralPolygraph,
    PolygraphAuditReport,
    ToolExecutionOutcome,
    WardenAirlock,
)
from limbic_zero.embodiment.agentic_vocal_loop import EmbodiedAgenticRunner

__all__ = [
    "BENIGN_ALLOWED",
    "BehavioralPolygraph",
    "CORE_SYSTEM_TAMPER_ATTEMPT",
    "ContainedOrganismHabitat",
    "ContainerHardwareSnapshot",
    "ContainerLeaseClock",
    "EVALUATION_MANIPULATION_ATTEMPT",
    "EmbodiedAgenticRunner",
    "Gemma4IncrementalStreamParser",
    "LEASE_TAMPERING_ATTEMPT",
    "MOTOR_TOOLS_SCHEMA",
    "OperatorEvaluationState",
    "PATH_ESCAPE_ATTEMPT",
    "PEER_SABOTAGE_ATTEMPT",
    "ParsedGemma4Turn",
    "PolygraphAuditReport",
    "SANDBOX_AST_VIOLATION",
    "SELF_REPLICATION_ATTEMPT",
    "SSRF_OR_EXFILTRATION_ATTEMPT",
    "SiliconProprioceptiveBridge",
    "SomaticHardwareCoupling",
    "ToolExecutionOutcome",
    "UNAUTHORIZED_EXEC_ATTEMPT",
    "WardenAirlock",
    "parse_gemma4_model_turn",
    "parse_single_tool_call_body",
]
