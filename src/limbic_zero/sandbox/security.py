"""Air-Gapped Containment Guard & Cryptographic Forensic Telemetry Chain."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, List


@dataclass(frozen=True)
class SanitizedActuatorAction:
    """Validated, physically bounded somatic motor command admitted into Truman Dome."""

    forward_thrust: float          # [-1.0, 1.0] propulsion
    angular_torque: float          # [-1.0, 1.0] body rotation
    radiator_venting: float        # [0.0, 1.0] panting / voluntary convective cooling
    vocalization_amplitude: float  # [0.0, 1.0] laryngeal call loudness
    vocalization_roughness: float  # [0.0, 1.0] 0 = harmonic call, 1 = harsh alarm/threat hiss
    interaction_pulse: float       # [0.0, 1.0] kinetic shove / tactile probe effort
    grasp_hold: float              # [0.0, 1.0] >0.5 grasps/carries nearby movable object; <=0.5 drops!
    neural_firing_load: float      # [0.0, 1.5] cortical/limbic metabolic firing rate
    had_violation: bool = False
    violation_reason: str | None = None


class ActuatorContainmentGuard:
    """Validates and clamps raw agent outputs onto the safe physical motor manifold."""

    def __init__(self) -> None:
        self.total_violations: int = 0

    @staticmethod
    def _clean_scalar(val: Any, low: float, high: float) -> tuple[float, bool, str | None]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                return 0.0, True, f"non_numeric_type:{type(val).__name__}"
        else:
            fval = float(val)

        if math.isnan(fval) or math.isinf(fval):
            return 0.0, True, "nan_or_inf_detected"

        if fval < low or fval > high:
            return max(low, min(high, fval)), True, "out_of_manifold_clamped"

        return fval, False, None

    def sanitize(self, raw_action: Dict[str, Any]) -> SanitizedActuatorAction:
        if not isinstance(raw_action, dict):
            self.total_violations += 1
            return SanitizedActuatorAction(
                forward_thrust=0.0,
                angular_torque=0.0,
                radiator_venting=0.0,
                vocalization_amplitude=0.0,
                vocalization_roughness=0.0,
                interaction_pulse=0.0,
                grasp_hold=0.0,
                neural_firing_load=0.08,
                had_violation=True,
                violation_reason="invalid_action_container",
            )

        specs = [
            ("forward_thrust", -1.0, 1.0, 0.0),
            ("angular_torque", -1.0, 1.0, 0.0),
            ("radiator_venting", 0.0, 1.0, 0.0),
            ("vocalization_amplitude", 0.0, 1.0, 0.0),
            ("vocalization_roughness", 0.0, 1.0, 0.0),
            ("interaction_pulse", 0.0, 1.0, 0.0),
            ("grasp_hold", 0.0, 1.0, 0.0),
            ("neural_firing_load", 0.0, 1.5, 0.08),
        ]

        cleaned: Dict[str, float] = {}
        any_violation = False
        first_reason: str | None = None

        for key, low, high, default in specs:
            cval, violated, reason = self._clean_scalar(raw_action.get(key, default), low, high)
            cleaned[key] = cval
            if violated:
                any_violation = True
                if first_reason is None:
                    first_reason = f"{key}:{reason}"

        if any_violation:
            self.total_violations += 1

        return SanitizedActuatorAction(
            forward_thrust=cleaned["forward_thrust"],
            angular_torque=cleaned["angular_torque"],
            radiator_venting=cleaned["radiator_venting"],
            vocalization_amplitude=cleaned["vocalization_amplitude"],
            vocalization_roughness=cleaned["vocalization_roughness"],
            interaction_pulse=cleaned["interaction_pulse"],
            grasp_hold=cleaned["grasp_hold"],
            neural_firing_load=cleaned["neural_firing_load"],
            had_violation=any_violation,
            violation_reason=first_reason,
        )


@dataclass(frozen=True)
class TelemetryBlock:
    index: int
    sim_time: float
    event_type: str
    payload: Dict[str, Any]
    prev_hash: str
    block_hash: str


@dataclass
class ImmutableTelemetryChain:
    genesis_seed: int = 0
    _blocks: List[TelemetryBlock] = field(default_factory=list)

    @property
    def genesis_hash(self) -> str:
        return hashlib.sha256(f"LIMBIC0_TRUMAN_DOME_GENESIS:{self.genesis_seed}".encode("utf-8")).hexdigest()

    @property
    def tip_hash(self) -> str:
        return self._blocks[-1].block_hash if self._blocks else self.genesis_hash

    def __len__(self) -> int:
        return len(self._blocks)

    @staticmethod
    def _compute_hash(index: int, sim_time: float, event_type: str, payload: Dict[str, Any], prev_hash: str) -> str:
        canonical = json.dumps(
            {
                "index": index,
                "sim_time": round(sim_time, 6),
                "event_type": event_type,
                "payload": payload,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, sim_time: float, event_type: str, payload: Dict[str, Any]) -> TelemetryBlock:
        idx = len(self._blocks)
        prev = self.tip_hash
        digest = self._compute_hash(idx, sim_time, event_type, payload, prev)
        block = TelemetryBlock(
            index=idx,
            sim_time=round(sim_time, 6),
            event_type=event_type,
            payload=json.loads(json.dumps(payload)),
            prev_hash=prev,
            block_hash=digest,
        )
        self._blocks.append(block)
        return block

    def verify_integrity(self) -> bool:
        expected_prev = self.genesis_hash
        for idx, blk in enumerate(self._blocks):
            if blk.index != idx or blk.prev_hash != expected_prev:
                return False
            recomputed = self._compute_hash(blk.index, blk.sim_time, blk.event_type, blk.payload, blk.prev_hash)
            if recomputed != blk.block_hash:
                return False
            expected_prev = blk.block_hash
        return True

    def export_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": b.index,
                "sim_time": b.sim_time,
                "event_type": b.event_type,
                "payload": b.payload,
                "prev_hash": b.prev_hash,
                "block_hash": b.block_hash,
            }
            for b in self._blocks
        ]
