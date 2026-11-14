"""Autonomous Closed-Loop Mind-Body-World Runner ('EmbodiedMindArena').

Couples each ``EmbodiedOrganism`` inside a ``TrumanDomeWorld`` to its own independent,
continuous-time ``LimbicBrain`` without any circular imports or omniscient state leaks:
    414D Sensory Nerves -> LimbicBrain -> 7D Muscles + Cerebral ATP Load -> TrumanDomeWorld
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from limbic_zero.mind.brain import BrainConfig, LimbicBrain, MindDiagnostics
from limbic_zero.sandbox.presets import build_cain_and_abel_arena
from limbic_zero.sandbox.world import OrganismObservation, TrumanDomeWorld


@dataclass
class ClosedLoopTickResult:
    """Synchronized physical-physiological observations and intracranial mind diagnostics."""

    sim_time: float
    observations: Dict[str, OrganismObservation]
    mind_diagnostics: Dict[str, MindDiagnostics]
    executed_actions: Dict[str, Dict[str, float]]


class EmbodiedMindArena:
    """Runs multiple autonomous mind-equipped organisms inside an air-gapped ``TrumanDomeWorld``."""

    def __init__(
        self,
        world: TrumanDomeWorld,
        brain_config: BrainConfig | None = None,
        base_seed: int = 1000,
    ) -> None:
        self.world = world
        self.brain_config = brain_config or BrainConfig()
        self.brains: Dict[str, LimbicBrain] = {}

        for idx, agent_id in enumerate(self.world.organisms.keys()):
            self.brains[agent_id] = LimbicBrain(
                config=self.brain_config,
                seed=base_seed + idx * 137,
            )

        self._latest_obs: Dict[str, OrganismObservation] = self.world.get_observations()
        self.latest_mind_diag: Dict[str, MindDiagnostics] = {}

    @classmethod
    def create_cain_and_abel(
        cls,
        seed: int = 42,
        deterministic_sensors: bool = False,
        brain_config: BrainConfig | None = None,
    ) -> "EmbodiedMindArena":
        """Build the canonical two-creature ('ADAM-01' & 'ABEL-02') autonomous mind-body arena."""
        world = build_cain_and_abel_arena(seed=seed, deterministic_sensors=deterministic_sensors)
        return cls(world=world, brain_config=brain_config, base_seed=seed * 31 + 7)

    def ensure_brain(self, agent_id: str, seed: int = 2026) -> LimbicBrain:
        """Attach a ``LimbicBrain`` if a new organism was spawned dynamically."""
        if agent_id not in self.brains:
            self.brains[agent_id] = LimbicBrain(config=self.brain_config, seed=seed)
            self._latest_obs = self.world.get_observations()
        return self.brains[agent_id]

    def step(
        self,
        dt: float = 0.05,
        *,
        deterministic_mind: bool = False,
        enable_grad: bool = False,
        action_overrides: Dict[str, Dict[str, Any]] | None = None,
    ) -> ClosedLoopTickResult:
        """Advance all brains and the physical-physiological world by ``dt`` seconds."""
        import torch

        if set(self.world.organisms.keys()) != set(self.brains.keys()):
            self._latest_obs = self.world.get_observations(dt=dt)
            for idx, aid in enumerate(self.world.organisms.keys()):
                if aid not in self.brains:
                    self.brains[aid] = LimbicBrain(config=self.brain_config, seed=9000 + idx)

        actions: Dict[str, Dict[str, float]] = {}
        mind_diags: Dict[str, MindDiagnostics] = {}

        with torch.set_grad_enabled(enable_grad):
            for agent_id, org in self.world.organisms.items():
                obs = self._latest_obs[agent_id]
                brain = self.brains[agent_id]
                act_dict, m_diag = brain.step(
                    obs=obs,
                    soma=org.soma,
                    dt=dt,
                    deterministic=deterministic_mind,
                )
                if not enable_grad:
                    brain.detach_state_()
                if action_overrides and agent_id in action_overrides:
                    merged = dict(act_dict)
                    merged.update(action_overrides[agent_id])
                    actions[agent_id] = merged
                else:
                    actions[agent_id] = act_dict
                mind_diags[agent_id] = m_diag

        next_obs = self.world.step(raw_actions=actions, dt=dt)
        self._latest_obs = next_obs
        self.latest_mind_diag = mind_diags

        # Periodically seal compact neuro-telemetry into the SHA-256 immutable audit ledger
        if self.world.tick_count % self.world.config.telemetry_snapshot_interval == 0:
            neuro_summary = {
                aid: {
                    "da": round(md.neuromodulators.dopamine_extracellular, 4),
                    "ne": round(md.neuromodulators.norepinephrine_extracellular, 4),
                    "5ht": round(md.neuromodulators.serotonin_extracellular, 4),
                    "ach": round(md.neuromodulators.acetylcholine_extracellular, 4),
                    "cea": round(md.amygdala.cea_defensive_output, 4),
                    "pfc_norm": round(md.pallium.prefrontal_persistence_norm, 4),
                    "engrams": md.hippocampus.active_engram_count,
                }
                for aid, md in mind_diags.items()
            }
            self.world.audit_chain.append(
                self.world.sim_time,
                "mind_neuro_telemetry_snapshot",
                neuro_summary,
            )

        return ClosedLoopTickResult(
            sim_time=self.world.sim_time,
            observations=next_obs,
            mind_diagnostics=mind_diags,
            executed_actions=actions,
        )
