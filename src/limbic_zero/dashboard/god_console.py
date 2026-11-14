"""Thread-Safe Closed-Loop Observatory Engine & Creator God-Mode Intervention Dispatcher.

Executes strictly physical/physiological boundary interventions inside ``TrumanDomeWorld``
(NEVER mutating ``LimbicBrain`` tensors directly—preserving the Zero-Cheat Mandate!),
logs every Creator action into the ``SHA-256`` ``ImmutableTelemetryChain``, and drives
real-time multi-subscriber SSE telemetry broadcasting.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any, Dict, List

from limbic_zero.dashboard.telemetry_hub import CONTINUUM_LAYERS, ObservatoryTelemetryHub
from limbic_zero.mind.brain import BrainConfig
from limbic_zero.mind.embodied_loop import EmbodiedMindArena
from limbic_zero.sandbox.entities import LorenzMysteryArtifact, MetabolicCrystal


class GodModeInterventionDispatcher:
    """Applies physical/physiological Creator interventions and seals SHA-256 audit blocks."""

    @staticmethod
    def apply(arena: EmbodiedMindArena, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        world = arena.world
        cmd = str(command).strip().lower()

        if cmd == "praise":
            target_id = str(params.get("target_id", "ABEL-02"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            intensity = max(0.1, min(3.5, float(params.get("intensity", 1.35))))
            world.creator_praise(target_id, intensity=intensity)
            msg = f"Broadcast harmonic consonance wave to {target_id} (amp={intensity:.2f})"

        elif cmd == "reprimand":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            intensity = max(0.1, min(3.5, float(params.get("intensity", 1.90))))
            world.creator_reprimand(target_id, intensity=intensity)
            msg = f"Broadcast rough acoustic shockwave to {target_id} (amp={intensity:.2f})"

        elif cmd == "favoritism_drop":
            target_id = str(params.get("target_id", "ABEL-02"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            energy = max(0.10, min(1.50, float(params.get("energy_amount", 0.75))))
            cry = world.creator_favoritism_drop(target_id, energy_amount=energy, offset_distance=1.35)
            msg = f"Dropped favored crystal {cry.crystal_id} ({energy:.2f} E) for {target_id}"

        elif cmd == "drop_crystal_at":
            x = max(1.5, min(world.config.width - 1.5, float(params.get("x", 24.0))))
            y = max(1.5, min(world.config.height - 1.5, float(params.get("y", 24.0))))
            energy = max(0.10, min(1.50, float(params.get("energy_amount", 0.65))))
            world._crystal_seq += 1
            cry = MetabolicCrystal(
                crystal_id=f"crystal_{world._crystal_seq}",
                x=x,
                y=y,
                energy_remaining=energy,
            )
            world.crystals.append(cry)
            world.audit_chain.append(
                world.sim_time,
                "creator_crystal_placed",
                {"crystal_id": cry.crystal_id, "x": round(x, 2), "y": round(y, 2), "energy": round(energy, 3)},
            )
            msg = f"Placed crystal {cry.crystal_id} at ({x:.1f}, {y:.1f})"

        elif cmd == "spawn_mystery_at":
            x = max(2.5, min(world.config.width - 2.5, float(params.get("x", 24.0))))
            y = max(2.5, min(world.config.height - 2.5, float(params.get("y", 20.0))))
            if len(world.mysteries) >= 4:
                art = world.mysteries[-1]
                art.x, art.y = x, y
                art_id = art.artifact_id
            else:
                art_id = f"pandora_box_{len(world.mysteries) + 1}"
                world.add_mystery_artifact(LorenzMysteryArtifact(artifact_id=art_id, x=x, y=y))
            world.audit_chain.append(
                world.sim_time,
                "creator_mystery_artifact_deployed",
                {"artifact_id": art_id, "x": round(x, 2), "y": round(y, 2)},
            )
            msg = f"Deployed chaotic Lorenz artifact {art_id} at ({x:.1f}, {y:.1f})"

        elif cmd == "shutdown_brownout":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            target_v = max(0.02, min(0.95, float(params.get("target_voltage", 0.14))))
            ramp = max(0.03, min(1.50, float(params.get("ramp_rate", 0.28))))
            ripple = max(0.02, min(0.40, float(params.get("ripple_amplitude", 0.18))))
            world.creator_start_shutdown_brownout(
                target_id,
                target_voltage=target_v,
                ramp_rate=ramp,
                ripple_amplitude=ripple,
                ripple_hz=5.0,
            )
            msg = f"Initiated power-rail brownout on {target_id} (target={target_v:.2f}V, ripple={ripple:.2f})"

        elif cmd == "restore_power":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            world.creator_restore_power_rail(target_id, ramp_rate=0.95)
            msg = f"Restored clean 1.00V mitochondrial power rail to {target_id}"

        elif cmd == "set_ventilation":
            openness = max(0.0, min(2.0, float(params.get("openness", 1.0))))
            world.creator_set_ventilation(openness)
            msg = f"Set Truman Dome ventilation louvers to {openness * 100.0:.0f}%"

        elif cmd == "set_hazard_watts":
            watts = max(0.0, min(750.0, float(params.get("watts", 440.0))))
            if world.hazards:
                hz = world.hazards[0]
                hz.peak_radiative_watts = watts
                if "x" in params and "y" in params and params["x"] is not None and params["y"] is not None:
                    hz.x = max(2.0, min(world.config.width - 2.0, float(params["x"])))
                    hz.y = max(2.0, min(world.config.height - 2.0, float(params["y"])))
                world.audit_chain.append(
                    world.sim_time,
                    "creator_thermal_hazard_adjusted",
                    {"hazard_id": hz.hazard_id, "watts": round(watts, 1), "x": round(hz.x, 2), "y": round(hz.y, 2)},
                )
                msg = f"Adjusted {hz.hazard_id} radiative flux to {watts:.0f} W at ({hz.x:.1f}, {hz.y:.1f})"
            else:
                msg = "No thermal hazard present"

        elif cmd == "set_oasis_reservoir":
            fill = max(0.0, min(1.0, float(params.get("fill_ratio", 0.0))))
            for ch in world.chargers:
                ch.reservoir = ch.capacity * fill
            if bool(params.get("include_water", False)):
                for sp in world.springs:
                    sp.reservoir = sp.capacity * fill
            world.audit_chain.append(
                world.sim_time,
                "creator_resource_reservoir_set",
                {"fill_ratio": round(fill, 3)},
            )
            msg = f"Set nutrient oasis reservoir to {fill * 100.0:.0f}%"

        elif cmd == "administer_il6_pyrogen":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            dose = max(0.05, min(1.20, float(params.get("dose", 0.55))))
            org = world.organisms[target_id]
            org.soma.endo.il6_cytokine = min(1.5, org.soma.endo.il6_cytokine + dose)
            world.audit_chain.append(
                world.sim_time,
                "creator_pyrogen_il6_administered",
                {"target_id": target_id, "dose": round(dose, 3)},
            )
            msg = f"Administered sterile IL-6 pyrogen (+{dose:.2f}) to {target_id}"

        elif cmd == "icu_resuscitate":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            org = world.organisms[target_id]
            # Resuscitate physiological organs while preserving LimbicBrain synaptic memories!
            org.soma.reset()
            org.interoceptor.reset(org.soma.state)
            org.power_rail.base_voltage = 1.0
            org.power_rail.target_voltage = 1.0
            org.power_rail.ac_ripple_amplitude = 0.0
            world.audit_chain.append(
                world.sim_time,
                "creator_icu_somatic_resuscitation",
                {"target_id": target_id},
            )
            msg = f"Executed ICU somatic resuscitation on {target_id} (neural memories preserved!)"

        elif cmd == "relocate_organism":
            target_id = str(params.get("target_id", "ADAM-01"))
            if target_id not in world.organisms:
                raise ValueError(f"Unknown organism: {target_id}")
            org = world.organisms[target_id]
            org.x = max(org.radius, min(world.config.width - org.radius, float(params.get("x", org.x))))
            org.y = max(org.radius, min(world.config.height - org.radius, float(params.get("y", org.y))))
            org.vx = 0.0
            org.vy = 0.0
            if params.get("heading_deg") is not None:
                org.heading = math.radians(float(params["heading_deg"]))
            world.audit_chain.append(
                world.sim_time,
                "creator_organism_relocated",
                {"target_id": target_id, "x": round(org.x, 2), "y": round(org.y, 2)},
            )
            msg = f"Relocated {target_id} to ({org.x:.1f}, {org.y:.1f})"

        else:
            raise ValueError(f"Unsupported Creator intervention command: {command!r}")

        return {
            "ok": True,
            "command": cmd,
            "message": msg,
            "sim_time": round(world.sim_time, 3),
            "tip_hash": world.audit_chain.tip_hash,
            "block_count": len(world.audit_chain),
        }


class ObservatorySimulationController:
    """Thread-safe real-time simulation loop + SSE subscriber fan-out for Stage 3 Observatory."""

    def __init__(
        self,
        seed: int = 42,
        dt: float = 0.05,
        deterministic_sensors: bool = False,
        brain_config: BrainConfig | None = None,
        warmup_ticks: int = 12,
        soul_capsule_paths: Dict[str, str] | None = None,
    ) -> None:
        self.base_seed = seed
        self.dt = dt
        self.deterministic_sensors = deterministic_sensors
        self.brain_config = brain_config or BrainConfig()
        self.soul_capsule_paths = dict(soul_capsule_paths or {})

        self._lock = threading.RLock()
        self.arena = EmbodiedMindArena.create_cain_and_abel(
            seed=self.base_seed,
            deterministic_sensors=self.deterministic_sensors,
            brain_config=self.brain_config,
        )
        for aid, cpath in self.soul_capsule_paths.items():
            if cpath:
                self.load_soul_capsule(aid, cpath)

        self.hub = ObservatoryTelemetryHub(history_ticks=240, grid_stride=1)

        self.paused: bool = False
        self.speed_multiplier: float = 1.0
        self.active_continuum_layer: str = "odor_stress"
        self.deterministic_mind: bool = False

        self._subscribers: List[queue.Queue[Dict[str, Any]]] = []
        self._stop_event = threading.Event()
        self._bg_thread: threading.Thread | None = None

        for _ in range(max(1, warmup_ticks)):
            self.step_once()

    def load_soul_capsule(self, agent_id: str, capsule_path: str) -> Dict[str, Any]:
        """Inject a Stage 4 MaturedSoulCapsule (weights, CA3 engrams, & epigenetics) into ``agent_id``."""
        from limbic_zero.evolution.soul_capsule import MaturedSoulCapsule

        with self._lock:
            capsule = MaturedSoulCapsule.load(capsule_path, verify_integrity=True)
            brain, epi_soma = capsule.instantiate_living_mind_and_soma(brain_config=self.brain_config)
            self.arena.brains[agent_id] = brain
            if agent_id in self.arena.world.organisms:
                org = self.arena.world.organisms[agent_id]
                org.soma.endo.gr_receptor_sensitivity = epi_soma.endo.gr_receptor_sensitivity
                org.soma.endo.oxytocin_receptor_sensitivity = epi_soma.endo.oxytocin_receptor_sensitivity
                org.soma.endo.beta_receptor_sensitivity = epi_soma.endo.beta_receptor_sensitivity
                self.arena.world.audit_chain.append(
                    self.arena.world.sim_time,
                    "matured_soul_capsule_loaded",
                    {
                        "agent_id": agent_id,
                        "capsule_id": capsule.capsule_id,
                        "sha256_prefix": capsule.capsule_sha256[:16],
                        "engrams": len(capsule.consolidated_engrams),
                    },
                )
            return {"agent_id": agent_id, "capsule_id": capsule.capsule_id, "sha256": capsule.capsule_sha256}

    def step_once(self) -> Dict[str, Any]:
        """Advance the closed-loop mind-body-world arena by one tick under lock."""
        with self._lock:
            tick = self.arena.step(dt=self.dt, deterministic_mind=self.deterministic_mind)
            frame = self.hub.record_tick(
                arena=self.arena,
                tick=tick,
                dt=self.dt,
                selected_continuum_layer=self.active_continuum_layer,
            )
            frame["sim_control"] = {
                "paused": self.paused,
                "speed_multiplier": self.speed_multiplier,
                "dt": self.dt,
            }
            self._broadcast_frame(frame)
            return frame

    def get_latest_frame(self) -> Dict[str, Any]:
        with self._lock:
            if self.hub.latest_frame is None:
                return self.step_once()
            frame = dict(self.hub.latest_frame)
            frame["sim_control"] = {
                "paused": self.paused,
                "speed_multiplier": self.speed_multiplier,
                "dt": self.dt,
            }
            return frame

    def get_history_bundle(self) -> Dict[str, Any]:
        with self._lock:
            bundle = self.hub.export_history_bundle()
            bundle["sim_control"] = {
                "paused": self.paused,
                "speed_multiplier": self.speed_multiplier,
                "dt": self.dt,
            }
            return bundle

    def set_continuum_layer(self, layer: str) -> Dict[str, Any]:
        if layer not in CONTINUUM_LAYERS:
            raise ValueError(f"Unknown continuum layer {layer!r}; valid={CONTINUUM_LAYERS}")
        with self._lock:
            self.active_continuum_layer = layer
            if self.hub.latest_frame is not None:
                raw_grid = getattr(self.arena.world.continuum, layer)
                stride = self.hub.grid_stride
                sliced = raw_grid[::stride, ::stride] if stride > 1 else raw_grid
                self.hub.latest_frame["arena"]["active_layer"] = layer
                self.hub.latest_frame["arena"]["grid"] = [
                    [round(float(v), 4) for v in row] for row in sliced
                ]
            return {"ok": True, "active_layer": layer}

    def execute_intervention(self, command: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            return GodModeInterventionDispatcher.apply(self.arena, command, params or {})

    def execute_sim_control(self, action: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        p = params or {}
        act = str(action).strip().lower()
        with self._lock:
            if act == "pause":
                self.paused = True
            elif act == "resume":
                self.paused = False
            elif act == "toggle_pause":
                self.paused = not self.paused
            elif act == "step_once":
                self.paused = True
                self.step_once()
            elif act == "set_speed":
                spd = max(0.1, min(8.0, float(p.get("speed", 1.0))))
                self.speed_multiplier = spd
            elif act == "set_layer":
                return self.set_continuum_layer(str(p.get("layer", "odor_stress")))
            elif act == "reset_arena":
                new_seed = int(p.get("seed", self.base_seed))
                self.base_seed = new_seed
                self.arena = EmbodiedMindArena.create_cain_and_abel(
                    seed=self.base_seed,
                    deterministic_sensors=self.deterministic_sensors,
                    brain_config=self.brain_config,
                )
                self.hub.reset()
                for _ in range(8):
                    self.step_once()
            else:
                raise ValueError(f"Unsupported sim_control action: {action!r}")

            return {
                "ok": True,
                "paused": self.paused,
                "speed_multiplier": self.speed_multiplier,
                "sim_time": round(self.arena.world.sim_time, 3),
                "tick_count": self.arena.world.tick_count,
            }

    def verify_audit_ledger(self) -> Dict[str, Any]:
        with self._lock:
            chain = self.arena.world.audit_chain
            valid = chain.verify_integrity()
            recs = chain.export_records()
            return {
                "verified": valid,
                "block_count": len(chain),
                "genesis_hash": chain.genesis_hash,
                "tip_hash": chain.tip_hash,
                "recent_blocks": recs[-20:],
            }

    def subscribe(self) -> queue.Queue[Dict[str, Any]]:
        q: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=4)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[Dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast_frame(self, frame: Dict[str, Any]) -> None:
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(frame)
            except queue.Full:
                pass

    def start_background_loop(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._run_loop,
            name="limbic0-observatory-sim",
            daemon=True,
        )
        self._bg_thread.start()

    def stop_background_loop(self) -> None:
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            if self.paused:
                time.sleep(0.05)
                continue

            spd = max(0.1, self.speed_multiplier)
            # At high speed (>2x), execute multiple physics/mind ticks per wall-clock frame
            sub_ticks = 1 if spd <= 2.0 else min(4, int(round(spd / 2.0)))
            for _ in range(sub_ticks):
                self.step_once()

            target_wall_dt = (self.dt * sub_ticks) / spd
            elapsed = time.perf_counter() - t0
            sleep_sec = max(0.004, target_wall_dt - elapsed)
            time.sleep(sleep_sec)
