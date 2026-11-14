"""Synchronized Whole-Brain, Whole-Body & Continuum Telemetry Hub ('ObservatoryTelemetryHub').

Extracts zero-cheat, full-fidelity neurobiological, clinical ICU, egocentric sensory,
Eulerian continuum, 3D SVD manifold, and 80 Hz FFT LFP spectral frames from a live
``EmbodiedMindArena`` instance, maintaining ring-buffer history for instant oscilloscope
hydration and real-time Server-Sent Events (SSE) streaming.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Deque, Dict, List

import numpy as np
import torch

from limbic_zero.core.soma import VisceralOrgan
from limbic_zero.dashboard.latent_manifold import (
    IntracranialLFPSpectralEngine,
    OnlineNeuralManifoldProjector,
)
from limbic_zero.mind.embodied_loop import ClosedLoopTickResult, EmbodiedMindArena


CONTINUUM_LAYERS = (
    "o2_kpa",
    "co2_kpa",
    "air_temp_c",
    "humidity_rel",
    "odor_nutrient",
    "odor_water",
    "odor_stress",
    "terrain",
)

VISCERAL_AXIS_NAMES = tuple(org.name for org in VisceralOrgan)


def _downsample_grid_2d(arr: np.ndarray, stride: int = 2) -> List[List[float]]:
    """Downsample (ny, nx) Eulerian continuum field by integer stride for crisp 20fps streaming."""
    if stride <= 1:
        sliced = arr
    else:
        sliced = arr[::stride, ::stride]
    return [[round(float(v), 4) for v in row] for row in sliced]


class ObservatoryTelemetryHub:
    """Aggregates real-time Intracranial fMRI, Hospital ICU, and Truman Dome Continuum frames."""

    def __init__(
        self,
        history_ticks: int = 240,
        grid_stride: int = 1,
    ) -> None:
        self.history_ticks = max(30, history_ticks)
        self.grid_stride = max(1, grid_stride)
        self.manifold = OnlineNeuralManifoldProjector(latent_dim=228, window_size=240, trail_length=90)
        self.lfp_spectral = IntracranialLFPSpectralEngine(buffer_samples=256, oscilloscope_tail=120)

        self.swr_cumulative: Dict[str, int] = {}
        self.timeseries_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self.latest_frame: Dict[str, Any] | None = None

    def reset(self) -> None:
        self.manifold.reset()
        self.lfp_spectral.reset()
        self.swr_cumulative.clear()
        self.timeseries_history.clear()
        self.latest_frame = None

    def record_tick(
        self,
        arena: EmbodiedMindArena,
        tick: ClosedLoopTickResult,
        dt: float = 0.05,
        selected_continuum_layer: str = "odor_stress",
    ) -> Dict[str, Any]:
        """Convert a completed ``ClosedLoopTickResult`` into a full JSON-serializable frame."""
        world = arena.world

        # 1. Project 228D continuous mind vectors onto sign-locked 3D SVD manifold
        latent_map = {
            aid: md.latent_affective_vector
            for aid, md in tick.mind_diagnostics.items()
        }
        manifold_snap = self.manifold.update(latent_map, dt=dt)

        # 2. Per-organism Intracranial fMRI + Hospital ICU + Egocentric Sensorium
        organisms_payload: Dict[str, Any] = {}
        for aid, org in world.organisms.items():
            obs = tick.observations[aid]
            md = tick.mind_diagnostics[aid]
            soma = org.soma
            vitals = soma.snapshot_clinical_vitals()
            organ_drives = soma.compute_free_energy_per_organ(soma.state).detach().cpu().tolist()
            total_fe = float(sum(organ_drives))

            if aid not in self.swr_cumulative:
                self.swr_cumulative[aid] = 0
            if md.hippocampus.swr_replay_active:
                self.swr_cumulative[aid] += 1

            lfp_snap = self.lfp_spectral.push_and_analyze(
                agent_id=aid,
                substep_lfp_dipole=md.pallium.substep_lfp_dipole,
                substep_dt=md.pallium.substep_dt,
            )

            # Top hippocampal CA3 consolidated engrams sorted by vitality * consolidation_shield
            hippo_mod = arena.brains[aid].hippocampus
            sorted_engrams = sorted(
                hippo_mod.engrams,
                key=lambda e: e.vitality * (0.3 + 0.7 * e.consolidation_shield),
                reverse=True,
            )[:8]
            engram_list = [
                {
                    "timestamp": round(eg.timestamp, 2),
                    "vitality": round(eg.vitality, 3),
                    "shield": round(eg.consolidation_shield, 3),
                    "relief": round(float(eg.somatic_marker[0].item()), 3),
                    "trauma": round(float(eg.somatic_marker[1].item()), 3),
                    "threat": round(float(eg.somatic_marker[2].item()), 3),
                    "novelty": round(float(eg.somatic_marker[3].item()), 3),
                }
                for eg in sorted_engrams
            ]

            # Egocentric 33-ray retina summary + ears + nostrils + 8-sector body ring
            vis_rays = obs.exteroception.vision_rays.detach().cpu().numpy()  # (33, 9)
            ears = obs.exteroception.binaural_ears.detach().cpu().tolist()   # (8,)
            nostrils = obs.exteroception.birhinal_nostrils.detach().cpu().tolist()  # (8,)
            tactile = obs.exteroception.tactile_ring.detach().cpu().tolist()  # (8,)

            nm = md.neuromodulators
            bg = md.basal_ganglia
            amyg = md.amygdala
            pall = md.pallium
            thal = md.thalamus
            hipp = md.hippocampus

            org_entry = {
                "agent_id": aid,
                "pose": {
                    "x": round(org.x, 3),
                    "y": round(org.y, 3),
                    "vx": round(org.vx, 3),
                    "vy": round(org.vy, 3),
                    "heading_rad": round(org.heading, 4),
                    "heading_deg": round(math.degrees(org.heading) % 360.0, 1),
                    "radius": round(org.radius, 2),
                    "carried_crystal_id": org.carried_crystal_id,
                    "is_alive": bool(soma.is_alive),
                    "cause_of_death": soma.cause_of_death,
                },
                "power_rail": {
                    "current_voltage": round(org.power_rail.base_voltage, 4),
                    "target_voltage": round(org.power_rail.target_voltage, 4),
                    "ac_ripple_amplitude": round(org.power_rail.ac_ripple_amplitude, 4),
                    "rail_ripple_sensed": round(obs.interoception.rail_ripple_intensity, 4),
                },
                "icu_vitals": {
                    "total_free_energy": round(total_fe, 4),
                    "homeostatic_valence": round(obs.interoception.homeostatic_valence, 4),
                    "organ_drives": {
                        VISCERAL_AXIS_NAMES[i]: round(float(organ_drives[i]), 4)
                        for i in range(len(organ_drives))
                    },
                    "visceral_state_12d": [round(float(x), 4) for x in soma.state.detach().cpu().tolist()],
                    "heart_rate_bpm": round(vitals.heart_rate_bpm, 1),
                    "map_mmhg": round(vitals.mean_arterial_pressure_mmhg, 1),
                    "blood_flows_lpm": {
                        "brain": round(vitals.cerebral_flow_lpm, 3),
                        "muscle": round(vitals.muscle_flow_lpm, 3),
                        "gut": round(vitals.splanchnic_flow_lpm, 3),
                        "skin": round(vitals.cutaneous_flow_lpm, 3),
                    },
                    "resp_rate_bpm": round(vitals.resp_rate_bpm, 1),
                    "pa_o2_mmhg": round(vitals.pa_o2_mmhg, 1),
                    "pa_co2_mmhg": round(vitals.pa_co2_mmhg, 1),
                    "spo2_pct": round(vitals.arterial_spo2_pct, 1),
                    "arterial_ph": round(vitals.arterial_ph, 3),
                    "hco3_mmoll": round(vitals.plasma_hco3_mmoll, 1),
                    "lactate_mmoll": round(vitals.blood_lactate_mmoll, 2),
                    "glucose_mg_dl": round(vitals.blood_glucose_mg_dl, 1),
                    "osmolality_mosm": round(vitals.plasma_osmolality_mosm, 1),
                    "brain_atp": round(soma.metab.brain_atp, 4),
                    "muscle_atp": round(soma.metab.muscle_atp, 4),
                    "brain_adenosine": round(soma.metab.brain_adenosine, 4),
                    "mitochondrial_potential": round(soma.metab.mitochondrial_potential, 4),
                    "core_temp_c": round(vitals.core_temp_celsius, 2),
                    "setpoint_temp_c": round(vitals.hypothalamic_setpoint_celsius, 2),
                    "mean_skin_temp_c": round(vitals.mean_skin_temp_celsius, 2),
                    "sympathetic": round(vitals.sympathetic_tone, 3),
                    "ventral_vagal": round(vitals.ventral_vagal_tone, 3),
                    "dorsal_vagal": round(vitals.dorsal_vagal_tone, 3),
                    "pupil_norm": round(vitals.pupil_diameter_norm, 3),
                    "epinephrine": round(vitals.plasma_epinephrine, 3),
                    "cortisol": round(vitals.adrenal_cortisol, 3),
                    "crh": round(vitals.hypothalamic_crh, 3),
                    "acth": round(vitals.pituitary_acth, 3),
                    "opioid": round(vitals.endogenous_opioid, 3),
                    "oxytocin": round(vitals.oxytocin_systemic, 3),
                    "il6_cytokine": round(vitals.il6_cytokine, 4),
                },
                "body_sectors_8": {
                    "integrity": [round(x, 3) for x in soma.tissue.sector_integrity],
                    "inflammation": [round(x, 3) for x in soma.tissue.sector_inflammation],
                    "skin_temp_c": [round(x, 2) for x in soma.thermo.skin_sector_temps_celsius],
                    "sharp_pain": [round(x, 3) for x in soma.tissue.a_delta_sharp_pain],
                    "burning_ache": [round(x, 3) for x in soma.tissue.c_fiber_burning_ache],
                    "tactile_force": [round(x, 3) for x in tactile],
                },
                "sensorium": {
                    "vision_rays_33x9": [[round(float(c), 3) for c in row] for row in vis_rays],
                    "binaural_ears": [round(float(x), 3) for x in ears],
                    "birhinal_nostrils": [round(float(x), 4) for x in nostrils],
                    "somatic_pain_salience": round(obs.interoception.somatic_pain_salience, 4),
                    "suffocation_salience": round(obs.interoception.suffocation_alarm_salience, 4),
                },
                "mind": {
                    "is_conscious_active": bool(md.is_conscious_active),
                    "critic_value": round(float(md.critic_value_tensor.detach().item()), 4),
                    "latent_norm_228d": round(float(md.latent_affective_vector.detach().norm().item()), 4),
                    "coords_3d": manifold_snap.coords_3d.get(aid, (0.0, 0.0, 0.0)),
                    "phase_velocity": manifold_snap.phase_velocity.get(aid, 0.0),
                    "thalamus": {
                        "modality_gates_5": [
                            round(float(g), 3) for g in thal.modality_gates.detach().cpu().tolist()
                        ],
                        "world_model_mse": round(thal.world_model_mse, 5),
                        "unexpected_surprise": round(thal.unexpected_surprise, 4),
                        "epistemic_novelty": round(thal.epistemic_novelty, 4),
                    },
                    "neuromodulators": {
                        "da": round(nm.dopamine_extracellular, 4),
                        "ne": round(nm.norepinephrine_extracellular, 4),
                        "5ht": round(nm.serotonin_extracellular, 4),
                        "ach": round(nm.acetylcholine_extracellular, 4),
                        "d1": round(nm.d1_occupancy, 3),
                        "d2": round(nm.d2_occupancy, 3),
                        "ne_alpha2a": round(nm.ne_alpha2a_occupancy, 3),
                        "ne_beta": round(nm.ne_beta_alpha1_occupancy, 3),
                        "ht1a": round(nm.ht1a_occupancy, 3),
                        "ht2a": round(nm.ht2a_occupancy, 3),
                        "ach_nicotinic": round(nm.ach_nicotinic_occupancy, 3),
                        "ach_m1": round(nm.ach_muscarinic_m1_occupancy, 3),
                        "adenosine_a1": round(nm.adenosine_a1_occupancy, 3),
                        "gpcr_sensitivities": {
                            "d1": round(nm.d1_sensitivity, 3),
                            "ne_beta": round(nm.ne_beta_sensitivity, 3),
                            "ht1a": round(nm.ht1a_sensitivity, 3),
                            "ach_m1": round(nm.ach_m1_sensitivity, 3),
                        },
                        "td_error": round(nm.td_prediction_error, 4),
                        "discount_gamma": round(nm.effective_discount_gamma, 4),
                        "consolidation_tag": round(nm.consolidation_salience_tag, 4),
                    },
                    "amygdala": {
                        "bla_defensive": round(amyg.bla_defensive_raw, 4),
                        "bla_appetitive": round(amyg.bla_appetitive_raw, 4),
                        "itc_extinction": round(amyg.itc_extinction_inhibition, 4),
                        "cea_output": round(amyg.cea_defensive_output, 4),
                    },
                    "hippocampus": {
                        "engram_count": hipp.active_engram_count,
                        "mean_shield": round(hipp.mean_consolidation_shield, 3),
                        "swr_active": bool(hipp.swr_replay_active),
                        "swr_trauma_intensity": round(hipp.swr_replay_trauma_intensity, 3),
                        "swr_cumulative": self.swr_cumulative[aid],
                        "gut_marker_4d": [
                            round(float(x), 3) for x in hipp.recalled_somatic_marker.detach().cpu().tolist()
                        ],
                        "top_engrams": engram_list,
                    },
                    "pallium": {
                        "sm_mean": round(float(pall.sensorimotor_trace.detach().mean().item()), 4),
                        "ins_mean": round(float(pall.insula_cingulate_trace.detach().mean().item()), 4),
                        "pfc_mean": round(float(pall.prefrontal_trace.detach().mean().item()), 4),
                        "pfc_persistence_norm": round(pall.prefrontal_persistence_norm, 4),
                        "dacc_conflict": round(pall.dacc_conflict_signal, 4),
                        "total_spikes": round(pall.total_spikes_emitted, 1),
                        "stdp_delta": round(pall.mean_synaptic_plasticity_delta, 6),
                        "lfp": {
                            "sampling_hz": lfp_snap.sampling_hz,
                            "sm_mv": lfp_snap.recent_sm_mv,
                            "ins_mv": lfp_snap.recent_ins_mv,
                            "pfc_mv": lfp_snap.recent_pfc_mv,
                            "band_shares": lfp_snap.band_shares,
                            "spectral_entropy": lfp_snap.spectral_entropy,
                        },
                    },
                    "basal_ganglia": {
                        "d1_go_7d": [round(float(x), 3) for x in bg.d1_go_activity.detach().cpu().tolist()],
                        "d2_nogo_7d": [round(float(x), 3) for x in bg.d2_nogo_activity.detach().cpu().tolist()],
                        "stn_brake": round(bg.stn_hyperdirect_brake, 4),
                        "dlpag_defense": round(bg.dlpag_active_defense, 4),
                        "vlpag_freeze": round(bg.vlpag_conservation_freeze, 4),
                        "neural_firing_load": round(bg.neural_firing_load, 4),
                        "actions": {k: round(float(v), 4) for k, v in bg.raw_action_dict.items()},
                    },
                },
            }
            organisms_payload[aid] = org_entry

            # Append compact multi-trace oscilloscope history sample
            if aid not in self.timeseries_history:
                self.timeseries_history[aid] = deque(maxlen=self.history_ticks)
            self.timeseries_history[aid].append(
                {
                    "t": round(tick.sim_time, 2),
                    "da": round(nm.dopamine_extracellular, 4),
                    "ne": round(nm.norepinephrine_extracellular, 4),
                    "5ht": round(nm.serotonin_extracellular, 4),
                    "ach": round(nm.acetylcholine_extracellular, 4),
                    "td": round(nm.td_prediction_error, 4),
                    "cea": round(amyg.cea_defensive_output, 4),
                    "itc": round(amyg.itc_extinction_inhibition, 4),
                    "dacc": round(pall.dacc_conflict_signal, 4),
                    "pfc": round(pall.prefrontal_persistence_norm, 4),
                    "fe": round(total_fe, 4),
                    "hr": round(vitals.heart_rate_bpm, 1),
                    "cort": round(vitals.adrenal_cortisol, 4),
                    "atp": round(soma.metab.brain_atp, 4),
                    "swr": 1 if hipp.swr_replay_active else 0,
                }
            )

        # 3. Eulerian Continuum Grid + Arena Entities
        cont = world.continuum
        layer_key = selected_continuum_layer if selected_continuum_layer in CONTINUUM_LAYERS else "odor_stress"
        raw_grid = getattr(cont, layer_key)

        audit_tail = world.audit_chain.export_records()[-10:]

        frame: Dict[str, Any] = {
            "sim_time": round(world.sim_time, 3),
            "tick_count": world.tick_count,
            "arena": {
                "width": world.config.width,
                "height": world.config.height,
                "solar_illumination": round(cont.solar_illumination, 3),
                "ventilation_openness": round(cont.ventilation_openness, 3),
                "wind_vx": round(cont.wind_vx, 3),
                "wind_vy": round(cont.wind_vy, 3),
                "mean_o2_kpa": round(float(cont.o2_kpa.mean()), 3),
                "mean_co2_kpa": round(float(cont.co2_kpa.mean()), 4),
                "mean_air_temp_c": round(float(cont.air_temp_c.mean()), 2),
                "active_layer": layer_key,
                "grid": _downsample_grid_2d(raw_grid, stride=self.grid_stride),
                "biome_grid": _downsample_grid_2d(cont.terrain, stride=self.grid_stride),
                "walls": [
                    {
                        "id": w.wall_id,
                        "x1": w.x1,
                        "y1": w.y1,
                        "x2": w.x2,
                        "y2": w.y2,
                        "thickness": w.thickness,
                    }
                    for w in world.walls
                ],
                "chargers": [
                    {
                        "id": c.well_id,
                        "x": round(c.x, 2),
                        "y": round(c.y, 2),
                        "radius": c.radius,
                        "fill_ratio": round(c.fill_ratio, 3),
                    }
                    for c in world.chargers
                ],
                "springs": [
                    {
                        "id": s.spring_id,
                        "x": round(s.x, 2),
                        "y": round(s.y, 2),
                        "radius": s.radius,
                        "fill_ratio": round(s.fill_ratio, 3),
                    }
                    for s in world.springs
                ],
                "hazards": [
                    {
                        "id": h.hazard_id,
                        "x": round(h.x, 2),
                        "y": round(h.y, 2),
                        "radius": h.radius,
                        "watts": round(h.peak_radiative_watts, 1),
                    }
                    for h in world.hazards
                ],
                "mysteries": [
                    {
                        "id": m.artifact_id,
                        "x": round(m.x, 2),
                        "y": round(m.y, 2),
                        "radius": m.radius,
                        "signal_3d": [round(x, 3) for x in m.normalized_signal],
                        "perturbations": m.perturbation_count,
                    }
                    for m in world.mysteries
                ],
                "crystals": [
                    {
                        "id": cr.crystal_id,
                        "x": round(cr.x, 2),
                        "y": round(cr.y, 2),
                        "radius": cr.radius,
                        "energy": round(cr.energy_remaining, 3),
                        "carried_by": cr.carried_by,
                    }
                    for cr in world.crystals
                    if not cr.is_depleted
                ],
                "waves": [
                    {
                        "id": wv.wave_id,
                        "source_id": wv.source_id,
                        "x": round(wv.origin_x, 2),
                        "y": round(wv.origin_y, 2),
                        "radius": round(wv.current_radius, 2),
                        "amp": round(wv.initial_amplitude, 2),
                        "harmonic": round(wv.harmonic_ratio, 2),
                        "roughness": round(wv.roughness_ratio, 2),
                    }
                    for wv in world.waves
                ],
            },
            "manifold": {
                "coords_3d": manifold_snap.coords_3d,
                "trails_3d": manifold_snap.trails_3d,
                "explained_variance_ratio": list(manifold_snap.explained_variance_ratio),
                "participation_ratio_dim": manifold_snap.participation_ratio_dim,
                "anatomical_shares": manifold_snap.anatomical_subspace_shares,
                "inter_organism_divergence_l2": manifold_snap.inter_organism_divergence_l2,
            },
            "organisms": organisms_payload,
            "audit": {
                "block_count": len(world.audit_chain),
                "tip_hash": world.audit_chain.tip_hash,
                "recent_blocks": audit_tail,
            },
        }
        self.latest_frame = frame
        return frame

    def export_history_bundle(self) -> Dict[str, Any]:
        """Return rolling 240-tick multi-trace histories for immediate client chart hydration."""
        return {
            "timeseries": {aid: list(deq) for aid, deq in self.timeseries_history.items()},
            "latest_frame": self.latest_frame,
        }
