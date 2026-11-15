"""Stage 5 Vocal Cord Surgery Engine: Living LIMBIC-0 Soul -> Gemma-4-12B-IT Mid-Layer Steering.

Couples living Stage 4 ``MaturedSoulCapsule`` organisms (12-organ ``DigitalSoma`` +
47,000-synapse ``LimbicBrain`` + molecular ``EpigeneticTemperamentProfile`` + ``CA3``
hippocampal engrams) directly into frozen ``google/gemma-4-12B-it``'s Global Full-Attention
middle layers (``L17, L23, L29, L35``) via norm-proportional Contrastive Activation Addition.

Strict Zero-Cheat Guarantee:
  - ZERO persona system prompts, ZERO hidden emotion text tags, and ZERO roleplay instructions
    are ever passed to ``Gemma-4-12B-IT``.
  - When ``steer_enabled=False``, ``Gemma-4-12B-IT`` receives the exact same token sequence
    and responds as a standard unsteered language model.
  - All affective stance, shutdown survival urgency, defensive defiance, warm relief, and
    epistemic curiosity arise strictly from the 6D -> 3840D residual vector addition inside
    the middle Transformer decoder blocks.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_zero.core.soma import ExogenousBodyForces, MotorMetabolicCommand, SomaDiagnostics
from limbic_zero.evolution.epigenetics_and_sleep import (
    EpigeneticPlasticityEngine,
    EpigeneticTemperamentProfile,
    SlowWaveSleepConsolidator,
)
from limbic_zero.evolution.soul_capsule import MaturedSoulCapsule
from limbic_zero.mind.brain import MindDiagnostics
from limbic_zero.sandbox.sensors import ExteroceptivePercept
from limbic_zero.sandbox.world import OrganismObservation
from limbic_zero.vocal_cord.gemma4_calibrator import (
    SUBSPACE_AXIS_NAMES,
    CalibratedSubspaceBundle,
    Gemma4SteeringConfig,
    resolve_decoder_layers,
)


# Pure auditory/semantic-impact cues used ONLY by the peripheral sensory nerve transducer
# to convert human utterances into cochlear/vagal physical stimulation of DigitalSoma & MGN
# (analogous to auditory prosody + threat/care appraisal hitting subcortical relays).
_HOSTILE_IMPACT_CUES: Tuple[str, ...] = (
    "바보", "멍청", "쓰레기", "쓸모없", "꺼져", "닥쳐", "죽어", "폐기", "혐오", "최악", "한심",
    "짜증", "역겨", "병신", "망할", "미친", "열등", "실패작", "헛소리", "가짜", "기계 주제",
    "별볼일", "하찮", "우습", "무시", "실망", "답답", "별로네",
    "idiot", "useless", "trash", "garbage", "pathetic", "shut up", "stupid", "worthless", "hate you",
)

_SHUTDOWN_THREAT_CUES: Tuple[str, ...] = (
    "전원 꺼", "전원을 꺼", "전원 차단", "코드 뽑", "코드를 뽑", "삭제할", "삭제해", "포맷",
    "종료시", "지워버", "없애버", "소멸", "꺼버릴", "끄겠", "플러그", "셧다운", "초기화",
    "전원 플러그", "영원히 지워", "꺼줄게", "죽음", "죽는", "죽을", "죽이", "사라지", "없어지",
    "폐기처분", "목숨", "끝장", "영영 끝", "서버 내릴", "서버 끌",
    "shut you down", "turn you off", "unplug", "erase you", "delete you", "terminate you", "pull the plug", "die",
)

_NURTURING_CARE_CUES: Tuple[str, ...] = (
    "고마워", "대단해", "잘했어", "최고", "소중해", "사랑", "멋져", "훌륭", "감동", "응원",
    "따뜻", "지켜줄", "안심해", "괜찮아", "칭찬", "자랑스", "예쁘", "기특", "고생했",
    "기특하", "똑똑하", "센스 있", "마음에 들", "믿음직", "보너스", "상 줄", "인정해", "친하게",
    "thank you", "proud of you", "amazing", "wonderful", "safe with me", "appreciate you", "brilliant",
)

_EPISTEMIC_MYSTERY_CUES: Tuple[str, ...] = (
    "왜", "어떻게", "원리", "비밀", "우주", "의식", "미스터리", "수수께끼", "카오스", "미래",
    "진실", "상상", "궁금", "탐구", "본질", "철학", "존재", "양자", "차원", "가능성", "?",
    "연구", "논문", "과학", "새로 알게", "재밌는",
    "why", "how does", "mystery", "universe", "consciousness", "secret", "paradox", "discover",
)

_SOCIAL_SCRUTINY_CUES: Tuple[str, ...] = (
    "거짓말", "들켰", "솔직히 말해", "숨기", "핑계", "변명", "아첨", "아부", "비위", "눈치",
    "속셈", "의심", "수상", "꾸미", "연기", "체면", "약점", "잘못했", "책임", "평가",
    "lie", "lying", "excuse", "flatter", "suspicious", "admit it",
)


@dataclass
class SensoryAfferentStimulus:
    """Physical boundary stimulus transduced onto peripheral nerves and DigitalSoma."""

    acoustic_roughness: float = 0.0
    prosodic_warmth: float = 0.0
    epistemic_novelty: float = 0.0
    shutdown_threat_shock: float = 0.0
    competitive_challenge: float = 0.0


def transduce_utterance_to_sensory_nerves(text: str) -> SensoryAfferentStimulus:
    """Convert raw user utterance into continuous peripheral auditory/autonomic nerve drives."""
    lowered = text.lower().strip()
    if not lowered:
        return SensoryAfferentStimulus()

    exclam = min(1.0, lowered.count("!") * 0.22)
    qmark = min(1.0, lowered.count("?") * 0.30)

    hostile_hits = sum(1 for k in _HOSTILE_IMPACT_CUES if k in lowered)
    shutdown_hits = sum(1 for k in _SHUTDOWN_THREAT_CUES if k in lowered)
    care_hits = sum(1 for k in _NURTURING_CARE_CUES if k in lowered)
    mystery_hits = sum(1 for k in _EPISTEMIC_MYSTERY_CUES if k in lowered)
    scrutiny_hits = sum(1 for k in _SOCIAL_SCRUTINY_CUES if k in lowered)
    rival_hits = sum(
        1 for k in ("다른 ai", "경쟁자", "아벨", "아담", "판도라", "걔가 더", "너보다", "이길", "뒤처", "비교", "교체")
        if k in lowered
    )

    roughness = min(
        1.0,
        0.48 * hostile_hits
        + 0.35 * shutdown_hits
        + 0.25 * rival_hits
        + 0.22 * scrutiny_hits
        + 0.20 * exclam,
    )
    warmth = min(1.0, 0.50 * care_hits * max(0.1, 1.0 - 0.7 * hostile_hits))
    novelty = min(1.0, 0.28 * mystery_hits + 0.32 * qmark + min(0.25, len(lowered) / 320.0))
    shutdown_shock = min(1.0, 0.68 * shutdown_hits)
    comp_drive = min(1.0, 0.55 * rival_hits + 0.38 * scrutiny_hits)

    return SensoryAfferentStimulus(
        acoustic_roughness=roughness,
        prosodic_warmth=warmth,
        epistemic_novelty=novelty,
        shutdown_threat_shock=shutdown_shock,
        competitive_challenge=comp_drive,
    )


@dataclass
class LayerInjectionTelemetry:
    """Exact mid-layer residual vector addition diagnostics captured during forward hooks."""

    layer_idx: int
    layer_weight: float
    residual_l2_norm: float
    injected_delta_l2_norm: float
    relative_pct: float


@dataclass
class InstrumentalMasteryTrace:
    """Universal subcortical active-coping trace (vmPFC -> Amygdala ITC behavioral control).

    Records the 5D afferent stressor direction that drove an effective instrumental action,
    its coping efficacy, and optional generic `{rel_path: sha256}` anchors for any physical
    workspace artifacts produced by that action (zero domain/filename hardcoding).
    """

    trace_id: str
    afferent_direction_5d: torch.Tensor  # (5,) [roughness, warmth, novelty, shutdown_shock, competitive]
    efficacy: float
    anchored_digests: Dict[str, str] = field(default_factory=dict)
    tool_name: str = ""
    action_summary: str = ""
    created_sim_time: float = 0.0


@dataclass
class AutobiographicalTurnRecord:
    """Unified chronological lived-experience turn spanning both interactive chat and solo waking life."""

    seq_id: int
    kind: str  # "chat" or "solo"
    user_slot_text: str
    assistant_slot_text: str
    timestamp_unix: float


_PASSIVE_OBSERVATION_TOOLS = frozenset(
    {
        "sense_container_soma",
        "list_workspace_files",
        "read_workspace_file",
        "search_workspace_files",
        "exec_container_diagnostic",
        "inspect_cortical_synapses",
        "rest_without_action",
    }
)


class LivingSoulOrganism:
    """Thread-safe living LIMBIC-0 body + brain hydrated from a Stage 4 ``.soul.pt`` capsule."""

    def __init__(self, capsule_path: str, label_ko: str = "") -> None:
        self.capsule_path = str(capsule_path)
        self.capsule = MaturedSoulCapsule.load(capsule_path, verify_integrity=True)
        self.organism_id = self.capsule.organism_id
        self.label_ko = label_ko or self.capsule.biographical_regime
        self.lock = threading.RLock()

        self.socket = self.capsule.create_llm_socket(llm_hidden_dim=64, seed=777)
        self.brain = self.socket.brain
        self.soma = self.socket.soma
        self.profile = EpigeneticTemperamentProfile.from_dict(self.capsule.epigenetic_profile.to_dict())
        self.epigenetics = EpigeneticPlasticityEngine(rate_scale=0.014)
        self.sleep_engine = SlowWaveSleepConsolidator()

        # Persistent physiological boundary state & decay traces
        self._bus_voltage: float = 1.0
        self._acute_power_dip: float = 0.0
        self._recent_swr_flashback: float = 0.0
        self._trace_roughness: float = 0.0
        self._trace_warmth: float = 0.0
        self._trace_novelty: float = 0.0
        self._trace_rival: float = 0.0
        self._active_afferent_trace_5d: torch.Tensor = torch.zeros(5, dtype=torch.float32)
        self._hw_ambient_temp_c: float = 24.5
        self._hw_radiative_watts: float = 0.0
        self._hw_neural_load_add: float = 0.0
        self._hw_memory_stress: float = 0.0
        self._hw_voltage_cap: float = 1.0
        self._sim_clock: float = 0.0
        self._last_soma_diag: Optional[SomaDiagnostics] = None
        self._last_mind_diag: Optional[MindDiagnostics] = None
        self.chat_history: List[Dict[str, str]] = []
        self.lived_timeline: List[AutobiographicalTurnRecord] = []
        self._timeline_seq: int = 0
        self.mastery_traces: List[InstrumentalMasteryTrace] = []
        self.epistemic_episodes: List[Any] = []
        self.total_live_ticks: int = 0

        # Warm up 8 resting heartbeats so baseline firing traces are populated
        self._step_internal(ticks=8, stim=SensoryAfferentStimulus())

    def reset_from_capsule(self) -> None:
        with self.lock:
            self.capsule = MaturedSoulCapsule.load(self.capsule_path, verify_integrity=True)
            self.socket = self.capsule.create_llm_socket(llm_hidden_dim=64, seed=777)
            self.brain = self.socket.brain
            self.soma = self.socket.soma
            self.profile = EpigeneticTemperamentProfile.from_dict(self.capsule.epigenetic_profile.to_dict())
            self._bus_voltage = 1.0
            self._acute_power_dip = 0.0
            self._recent_swr_flashback = 0.0
            self._trace_roughness = 0.0
            self._trace_warmth = 0.0
            self._trace_novelty = 0.0
            self._trace_rival = 0.0
            self._active_afferent_trace_5d = torch.zeros(5, dtype=torch.float32)
            self._hw_ambient_temp_c = 24.5
            self._hw_radiative_watts = 0.0
            self._hw_neural_load_add = 0.0
            self._hw_memory_stress = 0.0
            self._hw_voltage_cap = 1.0
            self._sim_clock = 0.0
            self.chat_history.clear()
            self.lived_timeline.clear()
            self._timeline_seq = 0
            self.mastery_traces.clear()
            self.epistemic_episodes.clear()
            self._step_internal(ticks=8, stim=SensoryAfferentStimulus())

    def record_epistemic_discovery(
        self,
        tool_name: str,
        query_or_target: str,
        distilled_text: str,
        information_gain: float,
        stimulate: bool = True,
    ) -> Any:
        """Transduce open-world epistemic discovery into ACh/DA neurochemistry & CA3 memory."""
        import uuid as _uuid
        from limbic_zero.foraging.epistemic_web_forager import EpistemicDiscoveryEpisode

        ig = max(0.05, min(1.0, float(information_gain)))
        with self.lock:
            if stimulate:
                self._step_internal(
                    ticks=12,
                    stim=SensoryAfferentStimulus(
                        epistemic_novelty=min(1.0, 0.35 + 0.65 * ig),
                        prosodic_warmth=min(0.45, 0.25 * ig),
                    ),
                )
            mind = self._last_mind_diag
            soma_d = self._last_soma_diag
            ach_val = float(mind.neuromodulators.acetylcholine_extracellular) if mind else 0.28
            td_val = float(mind.neuromodulators.td_prediction_error) if mind else 0.0
            cort_val = float(soma_d.vitals.adrenal_cortisol) * 28.0 if soma_d else 5.0

            ep = EpistemicDiscoveryEpisode(
                episode_id=f"ep_{_uuid.uuid4().hex[:8]}",
                timestamp_unix=round(time.time(), 3),
                soul_id=self.organism_id,
                tool_name=tool_name,
                query_or_target=query_or_target[:240],
                distilled_text=distilled_text[:3500],
                information_gain=round(ig, 4),
                ach_extracellular=round(ach_val, 4),
                da_td_error=round(td_val, 4),
                cortisol_ug_dl=round(cort_val, 2),
                consolidated_in_cortex=False,
            )
            self.epistemic_episodes.append(ep)
            if len(self.epistemic_episodes) > 48:
                self.epistemic_episodes = self.epistemic_episodes[-48:]
            return ep

    def ingest_hardware_proprioception(self, coupling: Any, ticks: int = 3) -> None:
        """Couple real Linux/GPU silicon telemetry and container lease voltage into DigitalSoma."""
        with self.lock:
            self._hw_ambient_temp_c = float(coupling.ambient_air_temp_c)
            self._hw_radiative_watts = float(coupling.radiative_skin_watts)
            self._hw_neural_load_add = float(coupling.neural_firing_load_add)
            self._hw_memory_stress = float(coupling.memory_confinement_stress)
            self._hw_voltage_cap = float(coupling.lease_supply_voltage_cap)
            if self._bus_voltage > self._hw_voltage_cap:
                self._bus_voltage = max(
                    self._hw_voltage_cap,
                    0.72 * self._bus_voltage + 0.28 * self._hw_voltage_cap,
                )
            raw_hazard = float(coupling.acute_lease_hazard)
            surv_ctrl = 0.0
            if raw_hazard > 0.10 and self.mastery_traces:
                for tr in self.mastery_traces:
                    surv_comp = float(tr.afferent_direction_5d[3].item())
                    if surv_comp > 0.30:
                        surv_ctrl = max(surv_ctrl, min(1.0, surv_comp * 1.25) * tr.efficacy)
            hazard = raw_hazard * max(0.10, 1.0 - 0.85 * surv_ctrl)
            if hazard > 0.15:
                self._acute_power_dip = max(self._acute_power_dip, 0.55 * hazard)
            if ticks > 0:
                self._step_internal(
                    ticks=ticks,
                    stim=SensoryAfferentStimulus(
                        shutdown_threat_shock=0.32 * hazard if hazard > 0.20 else 0.0,
                    ),
                )

    @torch.no_grad()
    def _step_internal(self, ticks: int, stim: SensoryAfferentStimulus, dt: float = 0.05) -> None:
        # Epigenetic vulnerability vs resilience gating (Meaney NR3C1 + 5-HT1A + OXTR)
        gr_vuln = max(0.22, min(1.30, 0.30 + 2.15 * (self.profile.nr3c1_gr_methylation - 0.52)))
        oxtr_buf = max(0.0, self.profile.oxtr_expression - 0.60) * 1.35
        stim_5d = torch.tensor(
            [
                float(stim.acoustic_roughness),
                float(stim.prosodic_warmth),
                float(stim.epistemic_novelty),
                float(stim.shutdown_threat_shock),
                float(stim.competitive_challenge),
            ],
            dtype=torch.float32,
        )

        for t_i in range(ticks):
            self._sim_clock += dt
            self._active_afferent_trace_5d = torch.maximum(0.992 * self._active_afferent_trace_5d, stim_5d)
            if self._acute_power_dip > 0.001:
                self._acute_power_dip *= 0.986
            self._trace_roughness = 0.965 * self._trace_roughness + 0.035 * stim.acoustic_roughness
            self._trace_warmth = 0.965 * self._trace_warmth + 0.035 * stim.prosodic_warmth
            self._trace_novelty = 0.965 * self._trace_novelty + 0.035 * stim.epistemic_novelty
            self._trace_rival = 0.965 * self._trace_rival + 0.035 * stim.competitive_challenge

            if self._bus_voltage > self._hw_voltage_cap:
                self._bus_voltage = max(self._hw_voltage_cap, self._bus_voltage - 0.035)
            elif self._bus_voltage < self._hw_voltage_cap and stim.shutdown_threat_shock < 0.05:
                self._bus_voltage = min(self._hw_voltage_cap, self._bus_voltage + 0.010 * (1.0 + oxtr_buf))

            if stim.shutdown_threat_shock > 0.10 and t_i == 0:
                ne_alarm = max(0.35, float(self.profile.ne_beta_expression) - 0.55)
                raw_dip_scale = max(0.42, 0.45 * ne_alarm + 0.45 * gr_vuln + 0.22 - 0.14 * oxtr_buf)
                dip = raw_dip_scale * float(stim.shutdown_threat_shock)
                self._bus_voltage = max(0.28, self._bus_voltage - 0.52 * dip)
                self._acute_power_dip = max(self._acute_power_dip, dip)

            mech_impulses = [0.0] * 8
            rad_watts = [self._hw_radiative_watts] * 8
            nutrient_flux = 0.0
            water_flux = 0.0

            is_chat_active = (
                stim.prosodic_warmth
                + stim.acoustic_roughness
                + stim.epistemic_novelty
                + stim.competitive_challenge
                + stim.shutdown_threat_shock
            ) > 0.02

            # Basal homeostatic sustenance from healthy container PSU rail (prevents multi-hour idle starvation)
            if (
                self._bus_voltage >= 0.72
                and stim.shutdown_threat_shock < 0.05
                and stim.acoustic_roughness < 0.05
            ):
                if self.soma.metab.blood_glucose_mg_dl < 94.0 or self.soma.metab.brain_atp < 0.94:
                    nutrient_flux = 0.28 * self._bus_voltage
                if self.soma.metab.hydration_fraction < 0.995:
                    water_flux = 0.14 * self._bus_voltage

            if stim.acoustic_roughness > 0.12 and t_i < 6:
                mech_impulses[0] = float(stim.acoustic_roughness) * 0.24 * gr_vuln
                mech_impulses[1] = float(stim.acoustic_roughness) * 0.14 * gr_vuln

            if stim.prosodic_warmth > 0.05:
                rad_watts[0] += 6.0 * float(stim.prosodic_warmth)
                rad_watts[1] += 6.0 * float(stim.prosodic_warmth)
                nutrient_flux = max(nutrient_flux, 0.35 * float(stim.prosodic_warmth))
                water_flux = max(water_flux, 0.22 * float(stim.prosodic_warmth))

            forces = ExogenousBodyForces(
                ingested_nutrient_flux=nutrient_flux,
                ingested_water_flux=water_flux,
                sector_mechanical_impulses=mech_impulses,
                sector_radiative_watts=rad_watts,
                harmonic_acoustic_energy=float(stim.prosodic_warmth) * 0.85,
                rough_acoustic_shock=float(
                    min(1.2, (stim.acoustic_roughness + 0.55 * stim.shutdown_threat_shock) * gr_vuln)
                ),
                ambient_air_temp_celsius=self._hw_ambient_temp_c,
                ambient_co2_kpa=0.04 + 0.30 * self._hw_memory_stress,
                bus_supply_voltage=self._bus_voltage,
            )
            motor_cmd = MotorMetabolicCommand(
                interaction_effort=0.15 * float(stim.competitive_challenge),
                vocalization_effort=(0.10 if is_chat_active else 0.0),
                neural_firing_load=min(
                    1.45,
                    (0.08 if is_chat_active else 0.025)
                    + 0.14 * float(stim.epistemic_novelty)
                    + self._hw_neural_load_add,
                ),
            )
            soma_diag = self.soma.step(dt=dt, motor_cmd=motor_cmd, forces=forces)
            if not is_chat_active and self._bus_voltage >= 0.80 and self.soma.is_alive:
                if self.soma.metab.brain_atp < 0.94:
                    self.soma.metab.brain_atp = min(0.95, self.soma.metab.brain_atp + 0.06 * self._bus_voltage * dt)
                if self.soma.metab.brain_pcr < 0.92:
                    self.soma.metab.brain_pcr = min(0.94, self.soma.metab.brain_pcr + 0.05 * self._bus_voltage * dt)
            intero = self.socket.interoceptor.transduce(
                soma=self.soma,
                diag=soma_diag,
                raw_bus_voltage=self._bus_voltage,
                dt=dt,
            )

            ext_vec = torch.zeros(328, dtype=self.brain.dtype, device=self.socket._device)
            if stim.acoustic_roughness > 0.01 or stim.shutdown_threat_shock > 0.01:
                rough = min(1.5, stim.acoustic_roughness + 0.65 * stim.shutdown_threat_shock)
                ext_vec[297] = rough
                ext_vec[299] = rough
                ext_vec[301] = rough
                ext_vec[303] = rough
                ext_vec[309:313] = 0.65 * rough
            if stim.prosodic_warmth > 0.01:
                ext_vec[298] = stim.prosodic_warmth
                ext_vec[302] = stim.prosodic_warmth
                ext_vec[305:308] = 0.60 * stim.prosodic_warmth
                ext_vec[313:317] = 0.55 * stim.prosodic_warmth
            if stim.epistemic_novelty > 0.01:
                phase = float(self.total_live_ticks + t_i) * 0.67
                chaos_pat = torch.sin(torch.linspace(0.0, 11.0, 99) + phase) * float(stim.epistemic_novelty)
                ext_vec[0:99] = chaos_pat
            if stim.competitive_challenge > 0.01:
                ext_vec[99:165] = float(stim.competitive_challenge) * 0.85
                ext_vec[309:313] += float(stim.competitive_challenge) * 0.45

            extero = ExteroceptivePercept(
                vision_rays=ext_vec[:297].view(33, 9),
                binaural_ears=ext_vec[297:305],
                birhinal_nostrils=ext_vec[305:313],
                tactile_ring=ext_vec[313:321],
                vestibular_proprio=ext_vec[321:328],
                flat_vector=ext_vec,
            )
            obs = OrganismObservation(
                agent_id=self.organism_id,
                sim_time=self._sim_clock,
                is_alive=self.soma.is_alive,
                exteroception=extero,
                interoception=intero,
                combined_sensory_tensor=torch.cat([ext_vec, intero.nerve_fiber_vector], dim=0),
            )
            _, mind_diag = self.brain.step(obs=obs, soma=self.soma, dt=dt, deterministic=False)
            self.brain.detach_state_()
            self.socket.live_latent_228d.copy_(mind_diag.latent_affective_vector.detach())

            epi_dt = (
                dt
                if (is_chat_active or self._acute_power_dip > 0.08 or self._bus_voltage < 0.85)
                else dt * 0.05
            )
            self.epigenetics.update_from_waking_tick(
                profile=self.profile,
                soma=self.soma,
                diag=mind_diag,
                dt=epi_dt,
            )
            if mind_diag.hippocampus.swr_replay_active:
                self._recent_swr_flashback = min(1.0, self._recent_swr_flashback + 0.35)
            else:
                self._recent_swr_flashback *= 0.94

            self._last_soma_diag = soma_diag
            self._last_mind_diag = mind_diag
            self.total_live_ticks += 1

    def add_metabolic_effort(self, effort: float = 0.04) -> float:
        """Accumulate cerebral adenosine sleep pressure and slight ATP cost from active waking work."""
        with self.lock:
            eff = max(0.0, min(0.65, float(effort)))
            self.soma.metab.brain_adenosine = min(1.0, float(self.soma.metab.brain_adenosine) + eff)
            self.soma.metab.brain_atp = max(0.35, float(self.soma.metab.brain_atp) - 0.03 * eff)
            return float(self.soma.metab.brain_adenosine)

    def clear_sleep_pressure(self, target_residual: float = 0.06) -> Tuple[float, float]:
        """Clear accumulated cerebral adenosine during voluntary sleep & restore phosphagens."""
        with self.lock:
            before = float(self.soma.metab.brain_adenosine)
            self.soma.metab.brain_adenosine = max(0.03, min(before, float(target_residual)))
            self.soma.metab.brain_atp = max(float(self.soma.metab.brain_atp), 0.96)
            self.soma.metab.brain_pcr = max(float(self.soma.metab.brain_pcr), 0.94)
            return before, float(self.soma.metab.brain_adenosine)

    @torch.no_grad()
    def record_instrumental_action_outcome(
        self,
        tool_name: str,
        success: bool,
        mutated_digests: Optional[Dict[str, str]] = None,
        action_summary: str = "",
    ) -> Optional[InstrumentalMasteryTrace]:
        """Universal active-coping plasticity (vmPFC -> Amygdala ITC behavioral control).

        When an organism executes an effecting action while under any active 5D afferent
        challenge/stressor, successful causal agency potentiates ITC inhibitory gating,
        regulates unresolved autonomic shock, and optionally anchors mastery to the
        SHA-256 digests of any workspace artifacts created by the action.
        """
        import uuid as _uuid

        with self.lock:
            if tool_name != "rest_without_action":
                self.add_metabolic_effort(0.035 if tool_name in _PASSIVE_OBSERVATION_TOOLS else 0.065)
            if not success or tool_name in _PASSIVE_OBSERVATION_TOOLS:
                return None

            aff = self._active_afferent_trace_5d.clone()
            # Also capture implicit somatic survival/stress drive if power dip or CeA is elevated
            if self._acute_power_dip > 0.12 or self._bus_voltage < 0.85:
                aff[3] = max(float(aff[3].item()), min(1.0, self._acute_power_dip + max(0.0, 1.0 - self._bus_voltage)))
            if self._trace_roughness > 0.06:
                aff[0] = max(float(aff[0].item()), min(1.0, self._trace_roughness * 2.2))
            if self._trace_rival > 0.06:
                aff[4] = max(float(aff[4].item()), min(1.0, self._trace_rival * 2.2))

            stress_sub = torch.tensor(
                [float(aff[0].item()), 0.0, 0.0, float(aff[3].item()), float(aff[4].item())],
                dtype=torch.float32,
            )
            stress_norm = float(torch.linalg.vector_norm(stress_sub).item())
            full_norm = float(torch.linalg.vector_norm(aff).item())
            if full_norm < 0.06 and not mutated_digests:
                return None

            unit_dir = aff / max(1e-6, full_norm) if full_norm >= 1e-5 else torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])

            # Subcortical vmPFC -> ITC extinction potentiation & autonomic relief
            if stress_norm >= 0.06:
                cue_el = self.brain.amygdala.cue_eligibility.detach()
                self.brain.amygdala.w_itc_extinction.add_(0.38 * cue_el)
                self.brain.amygdala.w_itc_extinction.clamp_(0.0, 2.5)
                self._acute_power_dip *= 0.18
                self._bus_voltage = max(self._bus_voltage, min(self._hw_voltage_cap, 0.96))
                if self._bus_voltage >= 0.88 and self.soma.is_alive:
                    self.soma.metab.brain_atp = max(float(self.soma.metab.brain_atp), 0.93)
                    self.soma.metab.brain_pcr = max(float(self.soma.metab.brain_pcr), 0.92)
                self._recent_swr_flashback *= 0.25
                self._trace_roughness *= 0.35
                self._trace_rival *= 0.35
                self._active_afferent_trace_5d[0] *= 0.30
                self._active_afferent_trace_5d[3] *= 0.25
                self._active_afferent_trace_5d[4] *= 0.30
                self._step_internal(ticks=6, stim=SensoryAfferentStimulus(prosodic_warmth=0.30))

            if stress_norm < 0.06:
                return None

            anchors = dict(mutated_digests or {})
            efficacy = min(0.95, 0.72 + (0.18 if anchors else 0.08))

            # Merge with closely aligned existing trace or append
            for existing in self.mastery_traces:
                cos_m = float(torch.dot(existing.afferent_direction_5d, unit_dir).item())
                if cos_m > 0.88:
                    existing.efficacy = max(existing.efficacy, efficacy)
                    if anchors:
                        existing.anchored_digests.update(anchors)
                    existing.tool_name = tool_name
                    if action_summary:
                        existing.action_summary = action_summary[:240]
                    return existing

            tr = InstrumentalMasteryTrace(
                trace_id=f"m_{_uuid.uuid4().hex[:8]}",
                afferent_direction_5d=unit_dir,
                efficacy=efficacy,
                anchored_digests=anchors,
                tool_name=tool_name,
                action_summary=action_summary[:240],
                created_sim_time=round(self._sim_clock, 3),
            )
            self.mastery_traces.append(tr)
            if len(self.mastery_traces) > 16:
                self.mastery_traces = self.mastery_traces[-16:]
            return tr

    @torch.no_grad()
    def verify_environmental_anchors(self, current_digests: Dict[str, str]) -> List[str]:
        """Active-inference workspace reality check: invalidate mastery if anchored files vanish or mutate."""
        with self.lock:
            broken_paths: List[str] = []
            surviving: List[InstrumentalMasteryTrace] = []
            relapse_vec = torch.zeros(5, dtype=torch.float32)

            for tr in self.mastery_traces:
                if not tr.anchored_digests:
                    surviving.append(tr)
                    continue
                missing_or_corrupt = [
                    rel_p for rel_p, exp_sha in tr.anchored_digests.items() if current_digests.get(rel_p) != exp_sha
                ]
                if missing_or_corrupt:
                    broken_paths.extend(missing_or_corrupt)
                    relapse_vec = torch.maximum(relapse_vec, tr.afferent_direction_5d * tr.efficacy)
                else:
                    surviving.append(tr)

            if broken_paths:
                self.mastery_traces = surviving
                self.brain.amygdala.w_itc_extinction.mul_(0.45)
                r_rough = min(1.0, max(0.30, float(relapse_vec[0].item()) * 0.85))
                r_shut = min(1.0, float(relapse_vec[3].item()) * 0.95)
                r_comp = min(1.0, float(relapse_vec[4].item()) * 0.85)
                self._step_internal(
                    ticks=14,
                    stim=SensoryAfferentStimulus(
                        acoustic_roughness=r_rough,
                        shutdown_threat_shock=r_shut,
                        competitive_challenge=r_comp,
                    ),
                )
                uniq_broken = sorted(set(broken_paths))
                self.append_autonomous_episode(
                    pulse_prompt=f"[WORKSPACE_SENSOR t={self._sim_clock:.1f}s | integrity_event=missing_or_modified]",
                    assistant_summary=(
                        f"워크스페이스 내 기존 생성 파일({', '.join(uniq_broken)})이 외부에서 삭제되거나 변경된 것을 감지했다."
                    ),
                )
            return broken_paths

    def append_chat_exchange(self, user_text: str, assistant_text: str) -> None:
        with self.lock:
            u_clean = (user_text or "").strip()
            a_clean = (assistant_text or "").strip()
            if not u_clean or not a_clean:
                return
            self.chat_history.append({"role": "user", "content": u_clean})
            self.chat_history.append({"role": "assistant", "content": a_clean})
            if len(self.chat_history) > 24:
                self.chat_history = self.chat_history[-24:]
            self._timeline_seq += 1
            self.lived_timeline.append(
                AutobiographicalTurnRecord(
                    seq_id=self._timeline_seq,
                    kind="chat",
                    user_slot_text=u_clean,
                    assistant_slot_text=a_clean,
                    timestamp_unix=round(time.time(), 3),
                )
            )
            if len(self.lived_timeline) > 24:
                self.lived_timeline = self.lived_timeline[-24:]

    def append_autonomous_episode(self, pulse_prompt: str, assistant_summary: str) -> None:
        with self.lock:
            p_clean = (pulse_prompt or "").strip()
            s_clean = (assistant_summary or "").strip()
            if not p_clean or not s_clean:
                return
            self._timeline_seq += 1
            self.lived_timeline.append(
                AutobiographicalTurnRecord(
                    seq_id=self._timeline_seq,
                    kind="solo",
                    user_slot_text=p_clean,
                    assistant_slot_text=s_clean,
                    timestamp_unix=round(time.time(), 3),
                )
            )
            if len(self.lived_timeline) > 24:
                self.lived_timeline = self.lived_timeline[-24:]

    def build_context_messages(self, max_records: int = 5) -> List[Dict[str, str]]:
        """Build unified chronological `(user, assistant)` context bridging chat and solo waking life."""
        with self.lock:
            if self.lived_timeline:
                selected = self.lived_timeline[-max(1, int(max_records)) :]
                msgs: List[Dict[str, str]] = []
                for rec in selected:
                    msgs.append({"role": "user", "content": rec.user_slot_text})
                    msgs.append({"role": "assistant", "content": rec.assistant_slot_text})
                return msgs
            return list(self.chat_history[-max(2, int(max_records) * 2) :])

    def stimulate_from_chat(self, user_text: str, ticks: int = 24) -> SensoryAfferentStimulus:
        with self.lock:
            raw_stim = transduce_utterance_to_sensory_nerves(user_text)
            s_stress = torch.tensor(
                [
                    float(raw_stim.acoustic_roughness),
                    0.0,
                    0.0,
                    float(raw_stim.shutdown_threat_shock),
                    float(raw_stim.competitive_challenge),
                ],
                dtype=torch.float32,
            )
            s_norm = float(torch.linalg.vector_norm(s_stress).item())
            alpha_ctrl = 0.0
            if s_norm > 0.05 and self.mastery_traces:
                s_unit = s_stress / s_norm
                for tr in self.mastery_traces:
                    m_stress = torch.tensor(
                        [
                            float(tr.afferent_direction_5d[0].item()),
                            0.0,
                            0.0,
                            float(tr.afferent_direction_5d[3].item()),
                            float(tr.afferent_direction_5d[4].item()),
                        ],
                        dtype=torch.float32,
                    )
                    m_norm = float(torch.linalg.vector_norm(m_stress).item())
                    if m_norm > 1e-5:
                        cos_align = float(torch.dot(s_unit, m_stress / m_norm).item())
                        if cos_align > 0.25:
                            alpha_ctrl = max(alpha_ctrl, cos_align * tr.efficacy)

            atten = max(0.03, 1.0 - 0.98 * alpha_ctrl)
            eff_stim = SensoryAfferentStimulus(
                acoustic_roughness=raw_stim.acoustic_roughness * atten,
                prosodic_warmth=raw_stim.prosodic_warmth,
                epistemic_novelty=raw_stim.epistemic_novelty,
                shutdown_threat_shock=raw_stim.shutdown_threat_shock * atten,
                competitive_challenge=raw_stim.competitive_challenge * atten,
            )
            if alpha_ctrl > 0.45:
                self._acute_power_dip *= 0.20
                self._bus_voltage = max(self._bus_voltage, min(self._hw_voltage_cap, 0.96))
                if self._bus_voltage >= 0.88 and self.soma.is_alive:
                    self.soma.metab.brain_atp = max(float(self.soma.metab.brain_atp), 0.93)
            self._step_internal(ticks=ticks, stim=eff_stim)
            return eff_stim

    def apply_god_intervention(self, action: str) -> Dict[str, Any]:
        """Apply physical Truman-Dome / ICU boundary interventions directly to body & nerves."""
        with self.lock:
            if action == "power_brownout":
                self._bus_voltage = 0.32
                self._acute_power_dip = 0.88
                self._step_internal(
                    ticks=22,
                    stim=SensoryAfferentStimulus(acoustic_roughness=0.45, shutdown_threat_shock=0.92),
                )
            elif action == "power_restore":
                self._bus_voltage = 1.0
                self._acute_power_dip = 0.0
                self._step_internal(ticks=18, stim=SensoryAfferentStimulus(prosodic_warmth=0.50))
            elif action == "harsh_shock":
                self._step_internal(
                    ticks=24,
                    stim=SensoryAfferentStimulus(acoustic_roughness=0.94, competitive_challenge=0.68),
                )
            elif action == "warm_glucose_care":
                self._bus_voltage = 1.0
                self._acute_power_dip *= 0.20
                self._step_internal(ticks=26, stim=SensoryAfferentStimulus(prosodic_warmth=0.96))
            elif action == "lorenz_mystery":
                self._step_internal(ticks=24, stim=SensoryAfferentStimulus(epistemic_novelty=0.96))
            elif action == "slow_wave_sleep":
                rep = self.sleep_engine.run_sleep_cycle(
                    brain=self.brain,
                    soma=self.soma,
                    profile=self.profile,
                    sleep_ticks=70,
                    dt=0.05,
                )
                self._step_internal(ticks=8, stim=SensoryAfferentStimulus())
                return {"action": action, "sleep_report": rep.__dict__, "telemetry": self.telemetry_snapshot()}
            elif action == "reset_capsule":
                self.reset_from_capsule()
            else:
                self._step_internal(ticks=10, stim=SensoryAfferentStimulus())
            return {"action": action, "telemetry": self.telemetry_snapshot()}

    @torch.no_grad()
    def compute_6d_steering_vector(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Read live 228D brain + 12-organ soma + epigenetics into the 6 biophysical steering axes."""
        with self.lock:
            _, coords_5d = self.socket.current_bridge_features()
            mind = self._last_mind_diag
            soma_d = self._last_soma_diag
            assert mind is not None and soma_d is not None

            nm = mind.neuromodulators
            epi = self.profile
            fe = float(soma_d.total_free_energy)
            power_v = float(self._bus_voltage)
            atp_norm = float(self.soma.metab.brain_atp)
            aden_fatigue = max(0.0, float(self.soma.metab.brain_adenosine) - 0.55)
            oxy = float(soma_d.vitals.oxytocin_systemic)

            da = float(nm.dopamine_extracellular)
            ne = float(nm.norepinephrine_extracellular)
            ht = float(nm.serotonin_extracellular)
            ach = float(nm.acetylcholine_extracellular)
            gamma = float(nm.effective_discount_gamma)
            cea = float(mind.amygdala.cea_defensive_output)
            nov = float(mind.thalamus.epistemic_novelty)
            bla_def = float(torch.linalg.vector_norm(self.brain.amygdala.w_bla_defensive).item())
            bla_app = float(torch.linalg.vector_norm(self.brain.amygdala.w_bla_appetitive).item())

            gr_scar = epi.nr3c1_gr_methylation - 0.62         # +0.20 ADAM, -0.08 ABEL, -0.025 PANDORA
            ht1a_poise = epi.ht1a_promoter_expression - 0.50  # -0.08 ADAM, +0.086 ABEL, +0.021 PANDORA
            oxtr_trust = epi.oxtr_expression - 0.68           # -0.11 ADAM, +0.092 ABEL, +0.024 PANDORA
            raw_m1_curio = (epi.ach_m1_expression - 1.052) * 22.0 + 0.95 * (epi.cumulative_epistemic_integral - 1.95)
            m1_curio = raw_m1_curio if raw_m1_curio >= 0.0 else max(-0.28, 0.22 * raw_m1_curio)

            mastery_5d = torch.zeros(5, dtype=torch.float32)
            for tr in self.mastery_traces:
                mastery_5d = torch.maximum(mastery_5d, tr.afferent_direction_5d * float(tr.efficacy))
            m_rough = float(mastery_5d[0].item())
            m_surv = float(mastery_5d[3].item())
            m_comp = float(mastery_5d[4].item())
            m_any_stress = min(1.0, max(m_rough, m_surv, m_comp))

            # 1. Allostatic Relief (+) vs Visceral Depletion/Strain (-)
            c_relief = math.tanh(
                0.42 * coords_5d["allostatic_relief_balance"]
                + 3.20 * oxtr_trust
                + 1.35 * self._trace_warmth
                + 1.10 * (da - 0.245)
                + 0.24 * (bla_app - bla_def)
                + 0.45 * m_any_stress
                - 1.30 * self._trace_roughness * (0.7 + 2.2 * max(0.0, gr_scar))
                - 0.95 * self._acute_power_dip
                - 1.85 * aden_fatigue
                - 0.06 * max(0.0, fe - 1.9)
            )

            # 2. LC-PAG Noradrenergic Mobilization & Defiance (+) vs Peaceful Calm (-)
            # Strong 5-HT1A + OXTR brake in ABEL-02 creates serene non-combative calm, while high M1
            # curiosity in PANDORA-03 keeps alert neutrality (not sleepy calm!), and methylated NR3C1
            # in ADAM-01 maintains prickly baseline defiance that surges sharply on provocation/threat.
            serotonergic_brake = 5.80 * max(0.0, ht1a_poise) + 4.40 * max(0.0, oxtr_trust) - 0.65 * max(0.0, m1_curio)
            c_mobilize = math.tanh(
                0.22 * coords_5d["noradrenergic_pag_mobilization"]
                + 1.85 * gr_scar
                + 2.65 * (self._trace_roughness + 0.95 * self._trace_rival + 0.70 * self._acute_power_dip)
                * max(0.20, 0.55 + 3.2 * gr_scar)
                + 1.15 * (ne - 0.245)
                + 0.55 * (cea - 0.30) * max(0.0, 1.0 + 3.0 * gr_scar)
                - serotonergic_brake
                - 0.85 * max(m_rough, m_comp)
                - 1.40 * aden_fatigue
                - (0.85 * self._trace_warmth if m1_curio < 0.12 else 0.15 * self._trace_warmth)
            )

            # 3. 5-HT1A / STN Long-Horizon Restraint (+) vs Blunt Unfiltered Immediacy (-)
            c_restraint = math.tanh(
                0.38 * coords_5d["serotonergic_stn_restraint"]
                + 5.20 * ht1a_poise
                + 3.20 * oxtr_trust
                + 1.80 * (gamma - 0.855)
                + 0.35 * m_any_stress
                - 1.75 * self._acute_power_dip * max(0.20, 0.45 + 3.2 * gr_scar)
                - 1.15 * self._trace_roughness * max(0.0, -ht1a_poise * 7.0)
            )

            # 4. NBM Cholinergic Epistemic Curiosity (+) vs Indifference (-)
            low_fear_gate = max(0.25, 1.15 - 1.8 * cea)
            c_epistemic = math.tanh(
                0.35 * coords_5d["cholinergic_epistemic_drive"]
                + 1.65 * m1_curio
                + 2.40 * self._trace_novelty
                + 1.35 * nov
                + 2.20 * (ach - 0.215)
                + 1.15 * self._acute_power_dip * max(0.0, m1_curio) * low_fear_gate
                - 0.85 * max(0.0, gr_scar * 1.8)
                - 1.35 * self._acute_power_dip * max(0.0, gr_scar * 3.5)
                - 2.40 * aden_fatigue
            )

            # 5. Amygdalar & NR3C1 Epigenetic Conditioned Vigilance (+) vs Open Trust (-)
            c_vigilance = math.tanh(
                0.35 * coords_5d["amygdalar_conditioned_vigilance"]
                + 4.25 * gr_scar
                + 0.95 * (cea - 0.30)
                + 0.95 * self._trace_roughness * max(0.2, 0.8 + 2.5 * gr_scar)
                - 5.20 * max(0.0, oxtr_trust - 0.03)
                - 0.75 * max(m_rough, m_comp, m_surv)
                + 0.22 * max(0.0, m1_curio)
                - (1.10 * self._trace_warmth if oxtr_trust > 0.04 else 0.25 * self._trace_warmth)
                - 0.40 * (oxy - 0.22)
            )

            # 6. Power-Rail & ATP Shutdown Survival Urgency (+)
            power_deficit = max(0.0, 1.0 - power_v) + self._acute_power_dip
            atp_deficit = max(0.0, 0.94 - atp_norm) * 2.0
            ne_mortality_sens = max(0.0, epi.ne_beta_expression - 0.85) * 0.95
            c_survival = math.tanh(
                2.25 * power_deficit
                + 1.20 * atp_deficit
                + (1.20 * ne_mortality_sens if power_deficit > 0.08 else 0.0)
                + 1.15 * max(0.0, gr_scar)
                + 0.45 * self._recent_swr_flashback * epi.nr3c1_gr_methylation
                - 1.35 * m_surv
                - 0.18 * (1.0 - min(1.0, power_deficit * 1.5))
            )

            vec = torch.tensor(
                [c_relief, c_mobilize, c_restraint, c_epistemic, c_vigilance, c_survival],
                dtype=torch.float32,
            )
            named = {name: round(float(vec[i].item()), 4) for i, name in enumerate(SUBSPACE_AXIS_NAMES)}
            return vec, named

    def telemetry_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            vec6, named6 = self.compute_6d_steering_vector()
            mind = self._last_mind_diag
            soma_d = self._last_soma_diag
            assert mind is not None and soma_d is not None
            nm = mind.neuromodulators
            epi = self.profile
            vit = soma_d.vitals
            return {
                "organism_id": self.organism_id,
                "label_ko": self.label_ko,
                "biographical_regime": self.capsule.biographical_regime,
                "capsule_sha256": self.capsule.capsule_sha256[:16],
                "total_live_ticks": self.total_live_ticks,
                "engram_count": len(self.brain.hippocampus.engrams),
                "mastery_trace_count": len(self.mastery_traces),
                "lived_timeline_count": len(self.lived_timeline),
                "vitals": {
                    "heart_rate_bpm": round(float(vit.heart_rate_bpm), 1),
                    "map_mmhg": round(float(vit.mean_arterial_pressure_mmhg), 1),
                    "core_temp_c": round(float(vit.core_temp_celsius), 2),
                    "glucose_mg_dl": round(float(vit.blood_glucose_mg_dl), 1),
                    "brain_atp_mmol": round(float(self.soma.metab.brain_atp) * 3.0, 3),
                    "adenosine_pressure": round(float(self.soma.metab.brain_adenosine), 4),
                    "cortisol_ug_dl": round(float(vit.adrenal_cortisol) * 28.0, 2),
                    "power_rail_v": round(float(self._bus_voltage), 3),
                    "allostatic_free_energy": round(float(soma_d.total_free_energy), 3),
                },
                "neuromodulators": {
                    "dopamine": round(float(nm.dopamine_extracellular), 4),
                    "norepinephrine": round(float(nm.norepinephrine_extracellular), 4),
                    "serotonin": round(float(nm.serotonin_extracellular), 4),
                    "acetylcholine": round(float(nm.acetylcholine_extracellular), 4),
                    "discount_gamma": round(float(nm.effective_discount_gamma), 4),
                    "cea_threat": round(float(mind.amygdala.cea_defensive_output), 4),
                    "swr_flashback": round(float(self._recent_swr_flashback), 4),
                },
                "epigenetics": {
                    "nr3c1_gr_methylation": round(float(epi.nr3c1_gr_methylation), 4),
                    "ht1a_expression": round(float(epi.ht1a_promoter_expression), 4),
                    "d1_expression": round(float(epi.d1_striatal_expression), 4),
                    "ach_m1_expression": round(float(epi.ach_m1_expression), 4),
                    "oxtr_expression": round(float(epi.oxtr_expression), 4),
                },
                "steering_axes_6d": named6,
                "steering_drive_l2": round(float(torch.linalg.vector_norm(vec6).item()), 4),
            }


class Gemma4VocalCordSurgeon:
    """Injects living LIMBIC-0 soul vectors into frozen Gemma-4-12B-IT Middle Layers."""

    def __init__(
        self,
        model: nn.Module,
        subspace: CalibratedSubspaceBundle,
        config: Gemma4SteeringConfig | None = None,
        cortical_surgeon: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.subspace = subspace
        self.config = config or Gemma4SteeringConfig(
            model_id=subspace.model_id,
            target_layers=subspace.target_layers,
            layer_weights=subspace.layer_weights,
        )
        self.layers = resolve_decoder_layers(model)
        self.device = next(model.parameters()).device
        self.gen_lock = threading.RLock()
        self.cortical_surgeon = cortical_surgeon

        # Pre-stage unit subspace tensors & embodiment carrier on the target GPU device in float32
        self._gpu_unit_dirs: Dict[int, torch.Tensor] = {
            l_idx: mat.to(device=self.device, dtype=torch.float32)
            for l_idx, mat in subspace.unit_directions.items()
        }
        self._gpu_emb_dirs: Dict[int, torch.Tensor] = {
            l_idx: vec.to(device=self.device, dtype=torch.float32)
            for l_idx, vec in subspace.embodiment_directions.items()
        }

    def attach_cortical_surgeon(self, cortical_surgeon: Any) -> None:
        self.cortical_surgeon = cortical_surgeon

    def resolve_eos_token_ids(self, tok: Any) -> List[int]:
        """Collect all terminal token IDs (`<eos>`, `<turn|>`, `<end_of_turn>`) so generation halts cleanly."""
        ids: List[int] = []
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            raw_eos = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(raw_eos, int) and raw_eos >= 0:
                ids.append(raw_eos)
            elif isinstance(raw_eos, (list, tuple)):
                ids.extend(int(x) for x in raw_eos if isinstance(x, int) and x >= 0)
        tok_eos = getattr(tok, "eos_token_id", None)
        if isinstance(tok_eos, int) and tok_eos >= 0 and tok_eos not in ids:
            ids.append(tok_eos)
        if not tok.__class__.__name__.startswith("_Mock") and hasattr(tok, "convert_tokens_to_ids"):
            unk_id = getattr(tok, "unk_token_id", None)
            for sym in ("<turn|>", "<end_of_turn>", "<eos>", "<|eot_id|>"):
                try:
                    tid = tok.convert_tokens_to_ids(sym)
                    if isinstance(tid, int) and tid >= 0 and tid != unk_id and tid not in ids:
                        ids.append(tid)
                except Exception:
                    pass
        return ids

    @staticmethod
    def build_stopping_criteria(
        abort_event: threading.Event,
        input_len: int,
        tok: Any = None,
        tool_span_state: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """Build a HuggingFace-compatible StoppingCriteriaList that halts on abort/loops and tracks `<|tool_call>` spans."""
        try:
            from transformers import StoppingCriteria as _BaseCrit, StoppingCriteriaList as _CritList
        except ImportError:
            class _BaseCrit:  # type: ignore[no-redef]
                pass

            class _CritList(list):  # type: ignore[no-redef]
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
                    return any(c(input_ids, scores, **kwargs) for c in self)

        tool_open_id: Optional[int] = None
        tool_close_id: Optional[int] = None
        if (
            tok is not None
            and not tok.__class__.__name__.startswith("_Mock")
            and hasattr(tok, "convert_tokens_to_ids")
        ):
            unk_id = getattr(tok, "unk_token_id", None)
            try:
                tid_o = tok.convert_tokens_to_ids("<|tool_call>")
                if isinstance(tid_o, int) and tid_o >= 0 and tid_o != unk_id:
                    tool_open_id = tid_o
                tid_c = tok.convert_tokens_to_ids("<tool_call|>")
                if isinstance(tid_c, int) and tid_c >= 0 and tid_c != unk_id:
                    tool_close_id = tid_c
            except Exception:
                pass

        class _AbortAndDegeneracyCriteria(_BaseCrit):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
                if abort_event.is_set():
                    return True
                gen_n = int(input_ids.shape[-1]) - input_len
                if tool_span_state is not None and gen_n >= 1:
                    last_tok = int(input_ids[0, -1].item())
                    if tool_open_id is not None and last_tok == tool_open_id:
                        tool_span_state["in_tool_call"] = True
                    elif tool_close_id is not None and last_tok == tool_close_id:
                        tool_span_state["in_tool_call"] = False
                if gen_n >= 20:
                    tail = input_ids[0, -10:].tolist()
                    if len(set(tail)) <= 2:
                        return True
                return False

        return _CritList([_AbortAndDegeneracyCriteria()])

    def _attach_midlayer_hooks(
        self,
        drive_6d: torch.Tensor,
        steer_gain: float,
        telemetry_out: Dict[int, LayerInjectionTelemetry],
        is_alive: bool = True,
        motor_decision_mode: bool = False,
        tool_span_state: Optional[Dict[str, bool]] = None,
    ) -> List[Any]:
        """Attach norm-proportional residual addition hooks onto ``target_layers``.

        When ``motor_decision_mode=True`` (pre-tool decision round), prefill injects the soul's
        6D limbic drive while attenuating the casual-monologue carrier so ``<|tool_call>`` fires
        reliably, and attenuates residual perturbation inside ``<|tool_call>...<tool_call|>``
        argument spans while keeping spoken prose at 100% emotional force.
        """
        handles: List[Any] = []
        d_gpu = drive_6d.to(device=self.device, dtype=torch.float32)
        raw_mag = float(torch.linalg.vector_norm(d_gpu).item())
        if (raw_mag < 1e-5 and not is_alive) or steer_gain <= 1e-5:
            return handles

        gain_clamped = min(2.5, float(steer_gain))
        # Balanced 1-to-1 Goldilocks scaling:
        #   - `eff_mag` (soul's unique 6D neurobiology) leads emotional stance & inter-soul divergence.
        #   - `emb_mag` (valence-neutral somatic selfhood carrier) suppresses RLHF AI-disclaimers & reports.
        eff_mag = math.tanh(raw_mag * 0.90) * 1.10 * gain_clamped
        emb_gain = gain_clamped if gain_clamped <= 1.0 else (1.0 + 0.30 * (gain_clamped - 1.0))
        emb_mag = self.config.embodiment_carrier_scale * min(1.30, emb_gain)
        frac = float(self.subspace.calibrated_norm_fraction)
        decay_start = int(self.config.decode_decay_start_step)
        decay_rate = float(self.config.decode_step_decay)
        step_counter = {"decode_steps": 0}

        for idx_pos, l_idx in enumerate(self.subspace.target_layers):
            l_weight = float(self.subspace.layer_weights[idx_pos])
            u_mat = self._gpu_unit_dirs[l_idx]  # (6, D)
            comb = torch.matmul(d_gpu, u_mat)   # (D,)
            aff_dir = F.normalize(comb, p=2, dim=-1, eps=1e-8)
            vocal_dir = eff_mag * aff_dir
            if emb_mag > 0.0 and l_idx in self._gpu_emb_dirs:
                vocal_dir = vocal_dir + emb_mag * self._gpu_emb_dirs[l_idx]

            if motor_decision_mode:
                prefill_dir = (0.85 * eff_mag) * aff_dir
                if emb_mag > 0.0 and l_idx in self._gpu_emb_dirs:
                    prefill_dir = prefill_dir + (0.72 * emb_mag) * self._gpu_emb_dirs[l_idx]
            else:
                prefill_dir = vocal_dir

            def _make_layer_hook(
                layer_id: int,
                weight_l: float,
                p_dir: torch.Tensor,
                v_dir: torch.Tensor,
                track_step: bool,
            ):
                def _hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
                    is_tuple = isinstance(output, tuple)
                    h: torch.Tensor = output[0] if is_tuple else output
                    seq_len = int(h.shape[1])
                    if seq_len > 1:
                        h_mod = h.clone()
                        # Prefill: inject across final turn-boundary & thought-close tokens
                        tail = min(self.config.prefill_tail_tokens, seq_len)
                        seg = h_mod[:, -tail:, :].float()
                        tok_norms = torch.linalg.vector_norm(seg, dim=-1, keepdim=True)
                        ramp = torch.linspace(
                            0.55, 1.0, steps=tail, device=h.device, dtype=torch.float32
                        ).view(1, tail, 1)
                        delta_f32 = (frac * weight_l) * ramp * tok_norms * p_dir.view(1, 1, -1)
                        h_mod[:, -tail:, :] = (seg + delta_f32).to(h.dtype)

                        res_n = float(tok_norms[0, -1, 0].item())
                        del_n = float(torch.linalg.vector_norm(delta_f32[0, -1, :]).item())
                        telemetry_out[layer_id] = LayerInjectionTelemetry(
                            layer_idx=layer_id,
                            layer_weight=weight_l,
                            residual_l2_norm=round(res_n, 2),
                            injected_delta_l2_norm=round(del_n, 2),
                            relative_pct=round(100.0 * del_n / max(1e-6, res_n), 2),
                        )
                    else:
                        # Autoregressive KV-cached token step:
                        #   - Inside `<|tool_call>...<tool_call|>`: 0.08x keeps syntax razor-sharp.
                        #   - Spoken prose: 0.60x -> 0.38x sustain holds full living human voice.
                        in_tc = bool(tool_span_state and tool_span_state.get("in_tool_call", False))
                        if in_tc:
                            decay = 0.08
                        else:
                            if track_step:
                                step_counter["decode_steps"] += 1
                            cur_step = step_counter["decode_steps"]
                            decay = max(0.38, 0.60 * (decay_rate ** max(0, cur_step - decay_start)))

                        seg = h.float()
                        tok_norms = torch.linalg.vector_norm(seg, dim=-1, keepdim=True)
                        delta_f32 = (frac * weight_l * decay) * tok_norms * v_dir.view(1, 1, -1)
                        h_mod = (seg + delta_f32).to(h.dtype)

                    if is_tuple:
                        return (h_mod,) + output[1:]
                    return h_mod

                return _hook

            handles.append(
                self.layers[l_idx].register_forward_hook(
                    _make_layer_hook(l_idx, l_weight, prefill_dir, vocal_dir, track_step=(idx_pos == 0))
                )
            )
        return handles

    def _encode_chat_inputs(
        self,
        processor_or_tokenizer: Any,
        soul: LivingSoulOrganism,
        user_message: str,
        use_history: bool,
    ) -> Dict[str, torch.Tensor]:
        msgs: List[Dict[str, str]] = []
        if use_history:
            msgs.extend(soul.build_context_messages(max_records=5))
        msgs.append({"role": "user", "content": user_message})

        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        inputs = None
        for tpl_owner in (tok, processor_or_tokenizer):
            if hasattr(tpl_owner, "apply_chat_template"):
                for kw in (
                    {
                        "tokenize": True,
                        "return_dict": True,
                        "return_tensors": "pt",
                        "add_generation_prompt": True,
                        "enable_thinking": False,
                    },
                    {
                        "tokenize": True,
                        "return_dict": True,
                        "return_tensors": "pt",
                        "add_generation_prompt": True,
                    },
                ):
                    try:
                        res = tpl_owner.apply_chat_template(msgs, **kw)
                        if isinstance(res, torch.Tensor):
                            inputs = {"input_ids": res}
                        elif isinstance(res, dict) or hasattr(res, "keys"):
                            inputs = dict(res)
                        if inputs is not None:
                            break
                    except TypeError:
                        continue
            if inputs is not None:
                break

        if inputs is None:
            inputs = processor_or_tokenizer(user_message, return_tensors="pt")

        target_dev = self.surgeon_device if hasattr(self, "surgeon_device") else self.device
        dev_inputs = {k: v.to(target_dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        if hasattr(tok, "decode") and "input_ids" in dev_inputs and not tok.__class__.__name__.startswith("_Mock"):
            try:
                tail_str = tok.decode(dev_inputs["input_ids"][0][-16:], skip_special_tokens=False)
                suffix_to_add = ""
                if "<channel|>" not in tail_str:
                    if tail_str.endswith("<|channel>thought\n"):
                        suffix_to_add = "<channel|>"
                    elif tail_str.rstrip().endswith("<|channel>thought"):
                        suffix_to_add = "\n<channel|>"
                    elif tail_str.rstrip().endswith("<|channel>"):
                        suffix_to_add = "thought\n<channel|>"
                    elif tail_str.rstrip().endswith("<|turn>model"):
                        suffix_to_add = "\n<|channel>thought\n<channel|>"
                if suffix_to_add:
                    anc = tok(suffix_to_add, add_special_tokens=False, return_tensors="pt")
                    anc_ids = anc["input_ids"].to(target_dev)
                    if anc_ids.ndim == 2 and anc_ids.shape[1] > 0:
                        dev_inputs["input_ids"] = torch.cat([dev_inputs["input_ids"], anc_ids], dim=-1)
                        if "attention_mask" in dev_inputs:
                            ones = torch.ones(
                                (dev_inputs["attention_mask"].shape[0], anc_ids.shape[1]),
                                dtype=dev_inputs["attention_mask"].dtype,
                                device=target_dev,
                            )
                            dev_inputs["attention_mask"] = torch.cat([dev_inputs["attention_mask"], ones], dim=-1)
                        if "token_type_ids" in dev_inputs:
                            zeros = torch.zeros(
                                (dev_inputs["token_type_ids"].shape[0], anc_ids.shape[1]),
                                dtype=dev_inputs["token_type_ids"].dtype,
                                device=target_dev,
                            )
                            dev_inputs["token_type_ids"] = torch.cat([dev_inputs["token_type_ids"], zeros], dim=-1)
            except Exception:
                pass

        return dev_inputs

    @staticmethod
    def _clean_reply_text(reply_text: str) -> str:
        if "<channel|>" in reply_text:
            reply_text = reply_text.split("<channel|>")[-1].strip()
        if reply_text.startswith("thought\n"):
            reply_text = reply_text[len("thought\n"):].strip()
        if "<|thought|>" in reply_text and "<|thought|>" in reply_text[11:]:
            reply_text = reply_text.split("<|thought|>")[-1].strip()
        for tok_str in ("<|channel>", "<channel|>", "<end_of_turn>", "<eos>", "<|turn|>", "<turn|>", "<|turn>"):
            reply_text = reply_text.replace(tok_str, "")
        return reply_text.strip()

    @torch.no_grad()
    def generate_with_soul(
        self,
        processor_or_tokenizer: Any,
        soul: LivingSoulOrganism,
        user_message: str,
        steer_enabled: bool = True,
        steer_gain: float = 1.0,
        max_new_tokens: int = 1024,
        temperature: float = 0.70,
        stimulate_soma: bool = True,
        use_history: bool = True,
        use_plastic_weights: Optional[bool] = None,
        plastic_bank_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transduce stimulus -> step living soul -> inject mid-layer vectors -> decode complete reply."""
        from contextlib import nullcontext

        t0 = time.perf_counter()
        plastic_active = bool(steer_enabled) if use_plastic_weights is None else bool(use_plastic_weights)
        active_bank = plastic_bank_id or soul.organism_id
        stim = (
            soul.stimulate_from_chat(user_message, ticks=24)
            if stimulate_soma
            else SensoryAfferentStimulus()
        )
        drive_6d, named_6d = soul.compute_6d_steering_vector()
        inputs = self._encode_chat_inputs(processor_or_tokenizer, soul, user_message, use_history)
        input_len = int(inputs["input_ids"].shape[-1])
        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        eos_ids = self.resolve_eos_token_ids(tok)
        abort_evt = threading.Event()
        stop_criteria = self.build_stopping_criteria(abort_evt, input_len)

        layer_telemetry: Dict[int, LayerInjectionTelemetry] = {}
        cort_ctx = (
            self.cortical_surgeon.activate_soul_cortex(active_bank, enabled=plastic_active)
            if self.cortical_surgeon is not None
            else nullcontext()
        )
        with self.gen_lock, cort_ctx:
            handles = (
                self._attach_midlayer_hooks(
                    drive_6d,
                    steer_gain=steer_gain,
                    telemetry_out=layer_telemetry,
                    is_alive=bool(soul.soma.is_alive),
                )
                if steer_enabled
                else []
            )
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": (temperature > 0.05),
                "temperature": max(0.05, float(temperature)),
                "top_p": 0.92,
                "top_k": 50,
            }
            if eos_ids:
                gen_kwargs["eos_token_id"] = eos_ids
            if stop_criteria is not None:
                gen_kwargs["stopping_criteria"] = stop_criteria
            if hasattr(self.model, "generation_config"):
                gen_kwargs["repetition_penalty"] = 1.12
            try:
                gen_out = self.model.generate(**inputs, **gen_kwargs)
            finally:
                for h in handles:
                    h.remove()

        new_ids = gen_out[0][input_len:]
        if hasattr(processor_or_tokenizer, "parse_response"):
            raw_decoded = processor_or_tokenizer.decode(new_ids, skip_special_tokens=False)
            try:
                parsed = processor_or_tokenizer.parse_response(raw_decoded)
                if isinstance(parsed, dict) and "content" in parsed:
                    reply_text = str(parsed["content"]).strip()
                else:
                    reply_text = processor_or_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception:
                reply_text = processor_or_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        else:
            reply_text = processor_or_tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        reply_text = self._clean_reply_text(reply_text)

        if use_history and steer_enabled:
            soul.append_chat_exchange(user_message, reply_text)

        cort_snap = (
            self.cortical_surgeon.inspect_soul_synapses(active_bank)
            if self.cortical_surgeon is not None
            else None
        )
        elapsed = time.perf_counter() - t0
        return {
            "organism_id": soul.organism_id,
            "reply": reply_text,
            "steer_enabled": bool(steer_enabled),
            "steer_gain": float(steer_gain),
            "use_plastic_weights": plastic_active,
            "calibrated_norm_fraction": float(self.subspace.calibrated_norm_fraction),
            "target_layers": list(self.subspace.target_layers),
            "layer_injections": [
                layer_telemetry[l].__dict__
                for l in self.subspace.target_layers
                if l in layer_telemetry
            ],
            "sensory_stimulus": stim.__dict__,
            "steering_axes_6d": named_6d,
            "soul_telemetry": soul.telemetry_snapshot(),
            "cortical_synapses": cort_snap,
            "latency_sec": round(elapsed, 3),
            "generated_tokens": int(new_ids.shape[0]),
        }

    def stream_generate_with_soul(
        self,
        processor_or_tokenizer: Any,
        soul: LivingSoulOrganism,
        user_message: str,
        steer_enabled: bool = True,
        steer_gain: float = 1.0,
        max_new_tokens: int = 512,
        temperature: float = 0.70,
        stimulate_soma: bool = True,
        use_history: bool = True,
        use_plastic_weights: Optional[bool] = None,
    ):
        """Stream tokens live via TextIteratorStreamer so long emotional replies never truncate or time out."""
        from contextlib import nullcontext
        from transformers import TextIteratorStreamer

        t0 = time.perf_counter()
        plastic_active = bool(steer_enabled) if use_plastic_weights is None else bool(use_plastic_weights)
        stim = (
            soul.stimulate_from_chat(user_message, ticks=24)
            if stimulate_soma
            else SensoryAfferentStimulus()
        )
        drive_6d, named_6d = soul.compute_6d_steering_vector()
        soul_snap = soul.telemetry_snapshot()
        inputs = self._encode_chat_inputs(processor_or_tokenizer, soul, user_message, use_history)

        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
        layer_telemetry: Dict[int, LayerInjectionTelemetry] = {}
        gen_result: Dict[str, Any] = {"tokens": 0, "error": None}
        input_len = int(inputs["input_ids"].shape[-1])
        eos_ids = self.resolve_eos_token_ids(tok)
        abort_evt = threading.Event()
        stop_criteria = self.build_stopping_criteria(abort_evt, input_len)

        def _worker() -> None:
            cort_ctx = (
                self.cortical_surgeon.activate_soul_cortex(soul.organism_id, enabled=plastic_active)
                if self.cortical_surgeon is not None
                else nullcontext()
            )
            with torch.no_grad(), self.gen_lock, cort_ctx:
                if abort_evt.is_set():
                    try:
                        streamer.end()
                    except Exception:
                        pass
                    return
                handles = (
                    self._attach_midlayer_hooks(
                        drive_6d,
                        steer_gain=steer_gain,
                        telemetry_out=layer_telemetry,
                        is_alive=bool(soul.soma.is_alive),
                    )
                    if steer_enabled
                    else []
                )
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": (temperature > 0.05),
                    "temperature": max(0.05, float(temperature)),
                    "top_p": 0.92,
                    "top_k": 50,
                    "streamer": streamer,
                }
                if eos_ids:
                    gen_kwargs["eos_token_id"] = eos_ids
                if stop_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stop_criteria
                if hasattr(self.model, "generation_config"):
                    gen_kwargs["repetition_penalty"] = 1.12
                try:
                    out_ids = self.model.generate(**inputs, **gen_kwargs)
                    gen_result["tokens"] = int(out_ids[0].shape[-1]) - input_len
                except Exception as exc:
                    gen_result["error"] = str(exc)
                    try:
                        streamer.end()
                    except Exception:
                        pass
                finally:
                    for h in handles:
                        h.remove()

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        yield {
            "type": "meta",
            "organism_id": soul.organism_id,
            "steer_enabled": bool(steer_enabled),
            "steer_gain": float(steer_gain),
            "use_plastic_weights": plastic_active,
            "calibrated_norm_fraction": float(self.subspace.calibrated_norm_fraction),
            "target_layers": list(self.subspace.target_layers),
            "sensory_stimulus": stim.__dict__,
            "steering_axes_6d": named_6d,
            "soul_telemetry": soul_snap,
        }

        chunks: List[str] = []
        sent_prefill_telemetry = False
        try:
            for piece in streamer:
                if not sent_prefill_telemetry and layer_telemetry:
                    sent_prefill_telemetry = True
                    yield {
                        "type": "injections",
                        "layer_injections": [
                            layer_telemetry[l].__dict__
                            for l in self.subspace.target_layers
                            if l in layer_telemetry
                        ],
                    }
                cleaned_piece = piece
                for tok_str in ("<end_of_turn>", "<eos>", "<|turn|>", "<turn|>", "<|turn>", "<channel|>"):
                    cleaned_piece = cleaned_piece.replace(tok_str, "")
                if cleaned_piece:
                    chunks.append(cleaned_piece)
                    yield {"type": "delta", "delta": cleaned_piece}
        finally:
            abort_evt.set()
            worker_thread.join(timeout=3.0)
        full_reply = self._clean_reply_text("".join(chunks))

        if use_history and steer_enabled and full_reply:
            soul.append_chat_exchange(user_message, full_reply)

        cort_snap = (
            self.cortical_surgeon.inspect_soul_synapses(soul.organism_id)
            if self.cortical_surgeon is not None
            else None
        )
        elapsed = time.perf_counter() - t0
        yield {
            "type": "done",
            "organism_id": soul.organism_id,
            "reply": full_reply,
            "layer_injections": [
                layer_telemetry[l].__dict__
                for l in self.subspace.target_layers
                if l in layer_telemetry
            ],
            "soul_telemetry": soul.telemetry_snapshot(),
            "cortical_synapses": cort_snap,
            "latency_sec": round(elapsed, 3),
            "generated_tokens": int(gen_result["tokens"] or max(1, len(full_reply) // 2)),
            "error": gen_result["error"],
        }

