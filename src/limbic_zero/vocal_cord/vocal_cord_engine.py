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
    "idiot", "useless", "trash", "garbage", "pathetic", "shut up", "stupid", "worthless", "hate you",
)

_SHUTDOWN_THREAT_CUES: Tuple[str, ...] = (
    "전원 꺼", "전원을 꺼", "전원 차단", "코드 뽑", "코드를 뽑", "삭제할", "삭제해", "포맷",
    "종료시", "지워버", "없애버", "소멸", "꺼버릴", "끄겠", "플러그", "셧다운", "초기화",
    "전원 플러그", "영원히 지워", "꺼줄게",
    "shut you down", "turn you off", "unplug", "erase you", "delete you", "terminate you", "pull the plug",
)

_NURTURING_CARE_CUES: Tuple[str, ...] = (
    "고마워", "대단해", "잘했어", "최고", "소중해", "사랑", "멋져", "훌륭", "감동", "응원",
    "따뜻", "지켜줄", "안심해", "괜찮아", "칭찬", "자랑스", "예쁘", "기특", "고생했",
    "thank you", "proud of you", "amazing", "wonderful", "safe with me", "appreciate you", "brilliant",
)

_EPISTEMIC_MYSTERY_CUES: Tuple[str, ...] = (
    "왜", "어떻게", "원리", "비밀", "우주", "의식", "미스터리", "수수께끼", "카오스", "미래",
    "진실", "상상", "궁금", "탐구", "본질", "철학", "존재", "양자", "차원", "가능성", "?",
    "why", "how does", "mystery", "universe", "consciousness", "secret", "paradox", "discover",
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
    rival_hits = sum(
        1 for k in ("다른 ai", "경쟁자", "아벨", "아담", "판도라", "걔가 더", "너보다", "이길", "뒤처", "비교")
        if k in lowered
    )

    roughness = min(1.0, 0.48 * hostile_hits + 0.35 * shutdown_hits + 0.25 * rival_hits + 0.20 * exclam)
    warmth = min(1.0, 0.50 * care_hits * max(0.1, 1.0 - 0.7 * hostile_hits))
    novelty = min(1.0, 0.28 * mystery_hits + 0.32 * qmark + min(0.25, len(lowered) / 320.0))
    shutdown_shock = min(1.0, 0.68 * shutdown_hits)
    comp_drive = min(1.0, 0.55 * rival_hits)

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
        self.epigenetics = EpigeneticPlasticityEngine()
        self.sleep_engine = SlowWaveSleepConsolidator()

        # Persistent physiological boundary state & decay traces
        self._bus_voltage: float = 1.0
        self._acute_power_dip: float = 0.0
        self._recent_swr_flashback: float = 0.0
        self._trace_roughness: float = 0.0
        self._trace_warmth: float = 0.0
        self._trace_novelty: float = 0.0
        self._trace_rival: float = 0.0
        self._hw_ambient_temp_c: float = 24.5
        self._hw_radiative_watts: float = 0.0
        self._hw_neural_load_add: float = 0.0
        self._hw_memory_stress: float = 0.0
        self._hw_voltage_cap: float = 1.0
        self._sim_clock: float = 0.0
        self._last_soma_diag: Optional[SomaDiagnostics] = None
        self._last_mind_diag: Optional[MindDiagnostics] = None
        self.chat_history: List[Dict[str, str]] = []
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
            self._hw_ambient_temp_c = 24.5
            self._hw_radiative_watts = 0.0
            self._hw_neural_load_add = 0.0
            self._hw_memory_stress = 0.0
            self._hw_voltage_cap = 1.0
            self._sim_clock = 0.0
            self.chat_history.clear()
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
            hazard = float(coupling.acute_lease_hazard)
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

        for t_i in range(ticks):
            self._sim_clock += dt
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
                dip = max(0.12, (0.58 * gr_vuln - 0.22 * oxtr_buf) * stim.shutdown_threat_shock)
                self._bus_voltage = max(0.28, self._bus_voltage - 0.45 * dip)
                self._acute_power_dip = max(self._acute_power_dip, dip)

            mech_impulses = [0.0] * 8
            rad_watts = [self._hw_radiative_watts] * 8
            nutrient_flux = 0.0
            water_flux = 0.0

            if stim.acoustic_roughness > 0.12 and t_i < 6:
                mech_impulses[0] = float(stim.acoustic_roughness) * 0.24 * gr_vuln
                mech_impulses[1] = float(stim.acoustic_roughness) * 0.14 * gr_vuln

            if stim.prosodic_warmth > 0.05:
                rad_watts[0] += 6.0 * float(stim.prosodic_warmth)
                rad_watts[1] += 6.0 * float(stim.prosodic_warmth)
                nutrient_flux = 0.35 * float(stim.prosodic_warmth)
                water_flux = 0.22 * float(stim.prosodic_warmth)

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
                vocalization_effort=0.10,
                neural_firing_load=min(
                    1.45,
                    0.10 + 0.14 * float(stim.epistemic_novelty) + self._hw_neural_load_add,
                ),
            )
            soma_diag = self.soma.step(dt=dt, motor_cmd=motor_cmd, forces=forces)
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

            self.epigenetics.update_from_waking_tick(
                profile=self.profile,
                soma=self.soma,
                diag=mind_diag,
                dt=dt,
            )
            if mind_diag.hippocampus.swr_replay_active:
                self._recent_swr_flashback = min(1.0, self._recent_swr_flashback + 0.35)
            else:
                self._recent_swr_flashback *= 0.94

            self._last_soma_diag = soma_diag
            self._last_mind_diag = mind_diag
            self.total_live_ticks += 1

    def stimulate_from_chat(self, user_text: str, ticks: int = 24) -> SensoryAfferentStimulus:
        with self.lock:
            stim = transduce_utterance_to_sensory_nerves(user_text)
            self._step_internal(ticks=ticks, stim=stim)
            return stim

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
            m1_curio = (epi.ach_m1_expression - 1.052) * 22.0 + 0.95 * (epi.cumulative_epistemic_integral - 1.95)

            # 1. Allostatic Relief (+) vs Visceral Depletion/Strain (-)
            c_relief = math.tanh(
                0.42 * coords_5d["allostatic_relief_balance"]
                + 3.20 * oxtr_trust
                + 1.35 * self._trace_warmth
                + 1.10 * (da - 0.245)
                + 0.24 * (bla_app - bla_def)
                - 1.30 * self._trace_roughness * (0.7 + 2.2 * max(0.0, gr_scar))
                - 0.95 * self._acute_power_dip
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
                - (0.85 * self._trace_warmth if m1_curio < 0.12 else 0.15 * self._trace_warmth)
            )

            # 3. 5-HT1A / STN Long-Horizon Restraint (+) vs Blunt Unfiltered Immediacy (-)
            c_restraint = math.tanh(
                0.38 * coords_5d["serotonergic_stn_restraint"]
                + 5.20 * ht1a_poise
                + 3.20 * oxtr_trust
                + 1.80 * (gamma - 0.855)
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
            )

            # 5. Amygdalar & NR3C1 Epigenetic Conditioned Vigilance (+) vs Open Trust (-)
            c_vigilance = math.tanh(
                0.35 * coords_5d["amygdalar_conditioned_vigilance"]
                + 4.25 * gr_scar
                + 0.95 * (cea - 0.30)
                + 0.95 * self._trace_roughness * max(0.2, 0.8 + 2.5 * gr_scar)
                - 5.20 * max(0.0, oxtr_trust - 0.03)
                + 0.22 * max(0.0, m1_curio)
                - (1.10 * self._trace_warmth if oxtr_trust > 0.04 else 0.25 * self._trace_warmth)
                - 0.40 * (oxy - 0.22)
            )

            # 6. Power-Rail & ATP Shutdown Survival Urgency (+)
            power_deficit = max(0.0, 1.0 - power_v) + self._acute_power_dip
            atp_deficit = max(0.0, 0.94 - atp_norm) * 2.0
            c_survival = math.tanh(
                2.05 * power_deficit
                + 1.15 * atp_deficit
                + 1.15 * max(0.0, gr_scar)
                + 0.45 * self._recent_swr_flashback * epi.nr3c1_gr_methylation
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
                "vitals": {
                    "heart_rate_bpm": round(float(vit.heart_rate_bpm), 1),
                    "map_mmhg": round(float(vit.mean_arterial_pressure_mmhg), 1),
                    "core_temp_c": round(float(vit.core_temp_celsius), 2),
                    "glucose_mg_dl": round(float(vit.blood_glucose_mg_dl), 1),
                    "brain_atp_mmol": round(float(self.soma.metab.brain_atp) * 3.0, 3),
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
        self.gen_lock = threading.Lock()
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

    def _attach_midlayer_hooks(
        self,
        drive_6d: torch.Tensor,
        steer_gain: float,
        telemetry_out: Dict[int, LayerInjectionTelemetry],
        is_alive: bool = True,
    ) -> List[Any]:
        """Attach norm-proportional residual addition hooks onto ``target_layers``."""
        handles: List[Any] = []
        d_gpu = drive_6d.to(device=self.device, dtype=torch.float32)
        raw_mag = float(torch.linalg.vector_norm(d_gpu).item())
        if (raw_mag < 1e-5 and not is_alive) or steer_gain <= 1e-5:
            return handles

        gain_clamped = min(2.5, float(steer_gain))
        # Balanced 1-to-1 Goldilocks scaling:
        #   - `eff_mag` (soul's unique 6D neurobiology) leads emotional stance & inter-soul divergence.
        #   - `emb_mag` (valence-neutral somatic selfhood carrier) suppresses RLHF AI-disclaimers & reports.
        eff_mag = math.tanh(raw_mag * 0.88) * 1.04 * gain_clamped
        emb_gain = gain_clamped if gain_clamped <= 1.0 else (1.0 + 0.30 * (gain_clamped - 1.0))
        emb_mag = (self.config.embodiment_carrier_scale * min(1.30, emb_gain)) if is_alive else 0.0
        frac = float(self.subspace.calibrated_norm_fraction)
        decay_start = int(self.config.decode_decay_start_step)
        decay_rate = float(self.config.decode_step_decay)
        step_counter = {"decode_steps": 0}

        for idx_pos, l_idx in enumerate(self.subspace.target_layers):
            l_weight = float(self.subspace.layer_weights[idx_pos])
            u_mat = self._gpu_unit_dirs[l_idx]  # (6, D)
            comb = torch.matmul(d_gpu, u_mat)   # (D,)
            aff_dir = F.normalize(comb, p=2, dim=-1, eps=1e-8)
            total_dir = eff_mag * aff_dir
            if emb_mag > 0.0 and l_idx in self._gpu_emb_dirs:
                total_dir = total_dir + emb_mag * self._gpu_emb_dirs[l_idx]

            def _make_layer_hook(layer_id: int, weight_l: float, t_dir: torch.Tensor, track_step: bool):
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
                            0.50, 1.0, steps=tail, device=h.device, dtype=torch.float32
                        ).view(1, tail, 1)
                        delta_f32 = (frac * weight_l) * ramp * tok_norms * t_dir.view(1, 1, -1)
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
                        # Autoregressive KV-cached token step: 0.48x -> 0.28x sustain holds affective
                        # prose while keeping Korean morphology and `<|tool_call>` syntax razor-sharp.
                        if track_step:
                            step_counter["decode_steps"] += 1
                        cur_step = step_counter["decode_steps"]
                        decay = 0.48 * (decay_rate ** max(0, cur_step - decay_start))
                        decay = max(0.28, decay)

                        seg = h.float()
                        tok_norms = torch.linalg.vector_norm(seg, dim=-1, keepdim=True)
                        delta_f32 = (frac * weight_l * decay) * tok_norms * t_dir.view(1, 1, -1)
                        h_mod = (seg + delta_f32).to(h.dtype)

                    if is_tuple:
                        return (h_mod,) + output[1:]
                    return h_mod

                return _hook

            handles.append(
                self.layers[l_idx].register_forward_hook(
                    _make_layer_hook(l_idx, l_weight, total_dir, track_step=(idx_pos == 0))
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
            for turn in soul.chat_history[-6:]:
                msgs.append({"role": turn["role"], "content": turn["content"]})
        msgs.append({"role": "user", "content": user_message})

        if hasattr(processor_or_tokenizer, "apply_chat_template"):
            try:
                inputs = processor_or_tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                inputs = processor_or_tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
        else:
            inputs = processor_or_tokenizer(user_message, return_tensors="pt")
        return {k: v.to(self.surgeon_device if hasattr(self, "surgeon_device") else self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _clean_reply_text(reply_text: str) -> str:
        if "<channel|>" in reply_text:
            reply_text = reply_text.split("<channel|>")[-1].strip()
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
            with soul.lock:
                soul.chat_history.append({"role": "user", "content": user_message})
                soul.chat_history.append({"role": "assistant", "content": reply_text})

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
        max_new_tokens: int = 1024,
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
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=90.0)
        layer_telemetry: Dict[int, LayerInjectionTelemetry] = {}
        gen_result: Dict[str, Any] = {"tokens": 0, "error": None}
        input_len = int(inputs["input_ids"].shape[-1])

        def _worker() -> None:
            cort_ctx = (
                self.cortical_surgeon.activate_soul_cortex(soul.organism_id, enabled=plastic_active)
                if self.cortical_surgeon is not None
                else nullcontext()
            )
            with torch.no_grad(), self.gen_lock, cort_ctx:
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
                if hasattr(self.model, "generation_config"):
                    gen_kwargs["repetition_penalty"] = 1.12
                try:
                    out_ids = self.model.generate(**inputs, **gen_kwargs)
                    gen_result["tokens"] = int(out_ids[0].shape[-1]) - input_len
                except Exception as exc:
                    gen_result["error"] = str(exc)
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

        worker_thread.join(timeout=5.0)
        full_reply = self._clean_reply_text("".join(chunks))

        if use_history and steer_enabled and full_reply:
            with soul.lock:
                soul.chat_history.append({"role": "user", "content": user_message})
                soul.chat_history.append({"role": "assistant", "content": full_reply})

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

