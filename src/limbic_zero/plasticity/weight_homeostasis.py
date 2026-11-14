"""Cortical Weight Homeostasis, Yerkes-Dodson Neuromodulated Gate & KL Stability Guard.

Implements the neurobiological rules that govern *when* and *how strongly* cortical
synapses are allowed to rewire:
  1. **Three-Factor Neuromodulated Plasticity Gate (``compute_neuromodulated_plasticity_gate``)**:
     Combines cholinergic epistemic surprise (``ACh M1``), dopaminergic TD insight/relief
     (``DA D1`` + ``|delta_TD|``), and optimal prefrontal ``NE alpha2A`` focus, modulated by
     the **Yerkes-Dodson Glucocorticoid/Catecholamine Law** (Arnsten, 2009; Diamond et al.,
     2007): acute hyper-cortisol + ``NE beta`` panic suppresses prefrontal/cortical LTP,
     whereas flat-affect ablation collapses plasticity strictly to ``0.0``.
  2. **Homeostatic Synaptic Scaling & KL Anchor Protection (``WeightHomeostasisGuard``)**:
     Enforces spectral/Frobenius norm ceilings, verifies post-update KL divergence against
     the frozen base cortex ``P_base``, and performs Tononi & Cirelli (2006) Synaptic
     Homeostasis (SHY) down-selection during Slow-Wave Sleep.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class NeuromodulatedGateReading:
    """Exact neurochemical decomposition of the cortical learning rate ``eta_eff``."""

    effective_lr: float
    base_lr: float
    affect_gate: float
    pro_learning_drive: float
    yerkes_dodson_panic_factor: float
    ach_m1_occupancy: float
    da_d1_occupancy: float
    abs_td_prediction_error: float
    ne_alpha2a_occupancy: float
    ne_beta_occupancy: float
    cortisol_ug_dl: float
    cea_threat: float
    epistemic_axis: float
    flat_affect_ablation: bool
    plasticity_regime: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_neuromodulated_plasticity_gate(
    soul: Any,
    base_lr: float = 2.8e-4,
    *,
    flat_affect_ablation: bool = False,
    sleep_swr_boost: float = 1.0,
) -> NeuromodulatedGateReading:
    """Compute 3-factor cortical plasticity learning rate strictly from live neurochemistry."""
    if flat_affect_ablation or soul is None:
        return NeuromodulatedGateReading(
            effective_lr=0.0,
            base_lr=float(base_lr),
            affect_gate=0.0,
            pro_learning_drive=0.0,
            yerkes_dodson_panic_factor=0.0,
            ach_m1_occupancy=0.0,
            da_d1_occupancy=0.0,
            abs_td_prediction_error=0.0,
            ne_alpha2a_occupancy=0.0,
            ne_beta_occupancy=0.0,
            cortisol_ug_dl=0.0,
            cea_threat=0.0,
            epistemic_axis=0.0,
            flat_affect_ablation=True,
            plasticity_regime="ANHEDONIC_FLAT_ZERO_LTP",
        )

    snap = soul.telemetry_snapshot()
    axes = snap.get("steering_axes_6d", {})
    vit = snap.get("vitals", {})
    epi = snap.get("epigenetics", {})

    mind_diag = getattr(soul, "_last_mind_diag", None)
    if mind_diag is not None:
        nm = mind_diag.neuromodulators
        ach_m1 = float(nm.ach_muscarinic_m1_occupancy)
        da_d1 = float(nm.d1_occupancy)
        abs_td = abs(float(nm.td_prediction_error))
        ne_a2a = float(nm.ne_alpha2a_occupancy)
        ne_beta = float(nm.ne_beta_alpha1_occupancy)
        cea = float(mind_diag.amygdala.cea_defensive_output)
    else:
        nm_d = snap.get("neuromodulators", {})
        ach_m1 = min(1.0, float(nm_d.get("acetylcholine", 0.25)) * 1.1)
        da_d1 = min(1.0, float(nm_d.get("dopamine", 0.28)) * 0.9)
        abs_td = 0.15
        ne_a2a = 0.45
        ne_beta = 0.15
        cea = float(nm_d.get("cea_threat", 0.15))

    cort_ug = float(vit.get("cortisol_ug_dl", 5.0))
    epistemic_axis = float(axes.get("cholinergic_epistemic_drive", 0.0))
    ach_m1_expr = float(epi.get("ach_m1_expression", 1.0))
    d1_expr = float(epi.get("d1_expression", 1.0))

    # 1. Pro-plasticity neuromodulatory coincidence (ACh M1 + DA D1/RPE + Prefrontal NE alpha2A)
    pro_drive = (
        0.38 * ach_m1 * max(0.75, ach_m1_expr)
        + 0.26 * da_d1 * max(0.75, d1_expr)
        + 0.22 * math.tanh(1.45 * abs_td)
        + 0.18 * ne_a2a
        + 0.24 * max(0.0, epistemic_axis)
    )

    # 2. Yerkes-Dodson Glucocorticoid & Beta-Adrenergic Prefrontal Suppression:
    # Moderate arousal leaves panic_factor == 1.0; acute hyper-cortisol (>13.5 ug/dL) + NE_beta
    # storm + CeA terror suppresses cortical LTP down to ~0.04 (impairing cortical learning!).
    cort_excess = max(0.0, (cort_ug - 13.5) / 11.5)
    ne_storm = max(0.0, ne_beta - 0.38)
    cea_storm = max(0.0, cea - 0.52)
    panic_suppression = 0.72 * cort_excess + 0.55 * ne_storm + 0.42 * cea_storm
    panic_factor = max(0.04, min(1.0, 1.0 - panic_suppression))

    # During Slow-Wave / REM SWR replay, adenosine clearance + spindle coupling boosts consolidation
    raw_gate = max(0.0, min(1.65, pro_drive * panic_factor * max(0.5, float(sleep_swr_boost))))

    # Deadzone below tiny tonic floor only if truly disengaged
    affect_gate = 0.0 if raw_gate < 0.03 else raw_gate
    eff_lr = float(base_lr) * affect_gate

    if panic_factor < 0.28:
        regime = "HYPER_CORTISOL_LTP_IMPAIRED"
    elif affect_gate >= 0.68:
        regime = "HIGH_EPISTEMIC_LTP_SURGE"
    elif affect_gate >= 0.30:
        regime = "BALANCED_CONSOLIDATIVE_LTP"
    else:
        regime = "LOW_AROUSAL_TONIC_PLASTICITY"

    return NeuromodulatedGateReading(
        effective_lr=round(eff_lr, 8),
        base_lr=float(base_lr),
        affect_gate=round(affect_gate, 5),
        pro_learning_drive=round(pro_drive, 5),
        yerkes_dodson_panic_factor=round(panic_factor, 5),
        ach_m1_occupancy=round(ach_m1, 4),
        da_d1_occupancy=round(da_d1, 4),
        abs_td_prediction_error=round(abs_td, 4),
        ne_alpha2a_occupancy=round(ne_a2a, 4),
        ne_beta_occupancy=round(ne_beta, 4),
        cortisol_ug_dl=round(cort_ug, 2),
        cea_threat=round(cea, 4),
        epistemic_axis=round(epistemic_axis, 4),
        flat_affect_ablation=False,
        plasticity_regime=regime,
    )


@dataclass
class HomeostasisAudit:
    """Post-surgery stability & homeostatic scaling report."""

    pre_frobenius_norm: float
    post_frobenius_norm: float
    anchor_kl_nats: float
    kl_rescaled: bool
    norm_clamped_modules: int
    scale_applied: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeightHomeostasisGuard:
    """Bounds per-module LoRA Frobenius norm, KL drift from base cortex, and SHY sleep scaling."""

    def __init__(
        self,
        max_module_frobenius: float = 2.40,
        max_total_frobenius: float = 14.0,
        max_anchor_kl_nats: float = 1.85,
        sleep_shy_decay: float = 0.992,
    ) -> None:
        self.max_module_frobenius = float(max_module_frobenius)
        self.max_total_frobenius = float(max_total_frobenius)
        self.max_anchor_kl_nats = float(max_anchor_kl_nats)
        self.sleep_shy_decay = float(sleep_shy_decay)
