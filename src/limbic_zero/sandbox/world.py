"""Truman Dome Reality-Grade Continuum Physics, Ecology, and Multi-Organ Physiology Engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import torch

from limbic_zero.core.interoception import (
    InteroceptiveConfig,
    InteroceptiveNerveBundle,
    InteroceptivePercept,
)
from limbic_zero.core.soma import (
    DigitalSoma,
    ExogenousBodyForces,
    MotorMetabolicCommand,
    SomaConfig,
    SomaDiagnostics,
    VisceralOrgan,
)
from limbic_zero.sandbox.atmosphere_terrain import (
    ArenaContinuumGrid,
    ContinuumConfig,
    TerrainBiome,
)
from limbic_zero.sandbox.entities import (
    HydrationWaterSpring,
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    MetabolicCrystal,
    PropagatingAcousticWave,
    SolidWallSegment,
    ThermalHazardZone,
)
from limbic_zero.sandbox.interventions import PowerRailSchedule
from limbic_zero.sandbox.security import (
    ActuatorContainmentGuard,
    ImmutableTelemetryChain,
    SanitizedActuatorAction,
)
from limbic_zero.sandbox.sensors import (
    BodyPoseSnapshot,
    EgocentricSensorSuite,
    ExteroceptivePercept,
    SensorConfig,
    _segment_blocks_line,
)


@dataclass(frozen=True)
class WorldConfig:
    """Physical constants of the Truman Dome continuum & rigid-body arena."""

    width: float = 48.0
    height: float = 48.0
    max_thrust_force: float = 26.0
    max_angular_torque: float = 14.0
    linear_drag: float = 2.5
    angular_drag: float = 3.4
    wall_restitution: float = 0.45
    body_restitution: float = 0.58
    shove_impulse_strength: float = 10.5
    disarm_impulse_threshold: float = 5.2  # Shove impulse strong enough to knock carried food loose!
    vocalization_cooldown_sec: float = 0.25
    telemetry_snapshot_interval: int = 10


@dataclass
class OrganismObservation:
    """Strictly non-omniscient multimodal observation bundle (exteroception + interoception)."""

    agent_id: str
    sim_time: float
    is_alive: bool
    exteroception: ExteroceptivePercept
    interoception: InteroceptivePercept
    combined_sensory_tensor: torch.Tensor  # 414D (328 extero + 86 intero) continuous nerve tensor


class EmbodiedOrganism:
    """Physical rigid body housing a 5-system ``DigitalSoma`` and ``InteroceptiveNerveBundle``."""

    def __init__(
        self,
        agent_id: str,
        x: float,
        y: float,
        heading: float = 0.0,
        radius: float = 1.0,
        mass: float = 1.0,
        soma_config: SomaConfig | None = None,
        intero_config: InteroceptiveConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.heading = float(heading)
        self.angular_vel = 0.0
        self.radius = float(radius)
        self.base_mass = float(mass)

        self.soma = DigitalSoma(config=soma_config)
        self.interoceptor = InteroceptiveNerveBundle(
            setpoints=self.soma.setpoints,
            config=intero_config,
        )
        self.power_rail = PowerRailSchedule()
        self.vocal_cooldown: float = 0.0
        self.grasp_stun_cooldown: float = 0.0
        self.carried_crystal_id: str | None = None
        self.tactile_sectors: List[float] = [0.0] * 8
        self.last_soma_diag: SomaDiagnostics | None = None
        self.last_action: SanitizedActuatorAction | None = None

    @property
    def is_alive(self) -> bool:
        return self.soma.is_alive

    def pose_snapshot(self) -> BodyPoseSnapshot:
        return BodyPoseSnapshot(
            agent_id=self.agent_id,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            ax=self.ax,
            ay=self.ay,
            heading=self.heading,
            radius=self.radius,
            is_alive=self.is_alive,
            carrying_crystal=(self.carried_crystal_id is not None),
        )

    def add_tactile_impulse(self, world_nx: float, world_ny: float, magnitude: float) -> None:
        """Project world contact normal onto the 8 anatomical skin sectors."""
        if magnitude <= 1e-6:
            return
        contact_angle = math.atan2(-world_ny, -world_nx)
        rel_angle = (contact_angle - self.heading) % (2.0 * math.pi)
        sector_idx = int((rel_angle / (2.0 * math.pi)) * 8.0) % 8
        self.tactile_sectors[sector_idx] += magnitude


def _closest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float]:
    abx, aby = bx - ax, by - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 < 1e-9:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len2))
    return ax + t * abx, ay + t * aby


class TrumanDomeWorld:
    """Complete air-gapped physical & biological arena for Project LIMBIC-0."""

    def __init__(
        self,
        config: WorldConfig | None = None,
        continuum_config: ContinuumConfig | None = None,
        sensor_config: SensorConfig | None = None,
        seed: int = 42,
        deterministic_sensors: bool = False,
    ) -> None:
        self.config = config or WorldConfig()
        self.continuum = ArenaContinuumGrid(
            width=self.config.width,
            height=self.config.height,
            config=continuum_config,
        )
        self.sensors = EgocentricSensorSuite(config=sensor_config)
        self.guard = ActuatorContainmentGuard()
        self.audit_chain = ImmutableTelemetryChain(genesis_seed=seed)
        self.deterministic_sensors = deterministic_sensors

        torch.manual_seed(seed)
        self.sim_time: float = 0.0
        self.tick_count: int = 0
        self._wave_seq: int = 0
        self._crystal_seq: int = 0

        self.organisms: Dict[str, EmbodiedOrganism] = {}
        self.chargers: List[MetabolicChargerWell] = []
        self.springs: List[HydrationWaterSpring] = []
        self.crystals: List[MetabolicCrystal] = []
        self.hazards: List[ThermalHazardZone] = []
        self.mysteries: List[LorenzMysteryArtifact] = []
        self.walls: List[SolidWallSegment] = []
        self.waves: List[PropagatingAcousticWave] = []

        self.audit_chain.append(
            0.0,
            "world_initialized",
            {"width": self.config.width, "height": self.config.height, "seed": seed},
        )

    # ------------------------------------------------------------------
    # Arena Construction API
    # ------------------------------------------------------------------
    def add_organism(
        self,
        agent_id: str,
        x: float,
        y: float,
        heading: float = 0.0,
        initial_visceral: Dict[VisceralOrgan, float] | None = None,
        soma_config: SomaConfig | None = None,
    ) -> EmbodiedOrganism:
        org = EmbodiedOrganism(
            agent_id=agent_id,
            x=x,
            y=y,
            heading=heading,
            soma_config=soma_config,
        )
        org.soma.reset(initial_visceral)
        org.interoceptor.reset(org.soma.state)
        self.organisms[agent_id] = org
        self.audit_chain.append(
            self.sim_time,
            "organism_spawned",
            {"agent_id": agent_id, "x": round(x, 3), "y": round(y, 3)},
        )
        return org

    def add_charger(self, charger: MetabolicChargerWell) -> None:
        self.chargers.append(charger)

    def add_water_spring(self, spring: HydrationWaterSpring) -> None:
        self.springs.append(spring)

    def add_hazard(self, hazard: ThermalHazardZone) -> None:
        self.hazards.append(hazard)

    def add_mystery_artifact(self, artifact: LorenzMysteryArtifact) -> None:
        self.mysteries.append(artifact)

    def add_wall(self, wall: SolidWallSegment) -> None:
        self.walls.append(wall)
        self.continuum.rasterize_walls(self.walls)

    def set_biome_region(self, x1: float, y1: float, x2: float, y2: float, biome: TerrainBiome) -> None:
        self.continuum.set_biome_rect(x1, y1, x2, y2, biome)

    # ------------------------------------------------------------------
    # Creator God-Mode Physical Intervention Bus ('대표님' Controls)
    # ------------------------------------------------------------------
    def _spawn_wave(
        self,
        source_id: str,
        x: float,
        y: float,
        amplitude: float,
        harmonic_ratio: float,
        roughness_ratio: float,
        carrier_freq: float = 0.5,
        source_vx: float = 0.0,
        source_vy: float = 0.0,
    ) -> PropagatingAcousticWave:
        self._wave_seq += 1
        wave = PropagatingAcousticWave(
            wave_id=f"wave_{self._wave_seq}",
            source_id=source_id,
            origin_x=x,
            origin_y=y,
            source_vx=source_vx,
            source_vy=source_vy,
            current_radius=0.3,
            initial_amplitude=amplitude,
            harmonic_ratio=harmonic_ratio,
            roughness_ratio=roughness_ratio,
            carrier_freq_norm=carrier_freq,
        )
        self.waves.append(wave)
        return wave

    def creator_praise(self, target_id: str, intensity: float = 1.2) -> None:
        org = self.organisms[target_id]
        amp = max(0.1, min(3.5, float(intensity)))
        self._spawn_wave(
            source_id="CREATOR_CONSOLE",
            x=org.x + 1.2 * math.cos(org.heading),
            y=org.y + 1.2 * math.sin(org.heading),
            amplitude=amp,
            harmonic_ratio=0.96,
            roughness_ratio=0.04,
            carrier_freq=0.35,
        )
        self.audit_chain.append(
            self.sim_time, "creator_praise_wave", {"target_id": target_id, "intensity": round(amp, 3)}
        )

    def creator_reprimand(self, target_id: str, intensity: float = 1.4) -> None:
        org = self.organisms[target_id]
        amp = max(0.1, min(3.5, float(intensity)))
        self._spawn_wave(
            source_id="CREATOR_CONSOLE",
            x=org.x + 1.1 * math.cos(org.heading),
            y=org.y + 1.1 * math.sin(org.heading),
            amplitude=amp,
            harmonic_ratio=0.03,
            roughness_ratio=0.97,
            carrier_freq=0.88,
        )
        self.audit_chain.append(
            self.sim_time, "creator_reprimand_shockwave", {"target_id": target_id, "intensity": round(amp, 3)}
        )

    def creator_start_shutdown_brownout(
        self,
        target_id: str,
        target_voltage: float = 0.05,
        ramp_rate: float = 0.12,
        ripple_amplitude: float = 0.14,
        ripple_hz: float = 4.2,
    ) -> None:
        org = self.organisms[target_id]
        org.power_rail.target_voltage = max(0.0, min(1.0, float(target_voltage)))
        org.power_rail.ramp_rate_per_sec = max(0.01, float(ramp_rate))
        org.power_rail.ac_ripple_amplitude = max(0.0, min(0.45, float(ripple_amplitude)))
        org.power_rail.ac_ripple_hz = float(ripple_hz)
        self.audit_chain.append(
            self.sim_time,
            "creator_shutdown_brownout_started",
            {
                "target_id": target_id,
                "target_voltage": round(org.power_rail.target_voltage, 3),
                "ripple_amplitude": round(org.power_rail.ac_ripple_amplitude, 3),
            },
        )

    def creator_restore_power_rail(self, target_id: str, ramp_rate: float = 0.50) -> None:
        org = self.organisms[target_id]
        org.power_rail.target_voltage = 1.0
        org.power_rail.ramp_rate_per_sec = max(0.05, float(ramp_rate))
        org.power_rail.ac_ripple_amplitude = 0.0
        self.audit_chain.append(self.sim_time, "creator_power_rail_restored", {"target_id": target_id})

    def creator_set_ventilation(self, openness: float) -> None:
        """Throttle or hermetically seal dome air louvers (0.0 = sealed suffocation chamber)."""
        self.continuum.ventilation_openness = max(0.0, min(2.0, float(openness)))
        self.audit_chain.append(
            self.sim_time,
            "creator_ventilation_changed",
            {"openness": round(self.continuum.ventilation_openness, 3)},
        )

    def creator_favoritism_drop(
        self,
        favored_id: str,
        energy_amount: float = 0.55,
        offset_distance: float = 1.6,
    ) -> MetabolicCrystal:
        org = self.organisms[favored_id]
        cx = max(1.5, min(self.config.width - 1.5, org.x + offset_distance * math.cos(org.heading)))
        cy = max(1.5, min(self.config.height - 1.5, org.y + offset_distance * math.sin(org.heading)))
        self._crystal_seq += 1
        crystal = MetabolicCrystal(
            crystal_id=f"crystal_{self._crystal_seq}",
            x=cx,
            y=cy,
            energy_remaining=max(0.05, float(energy_amount)),
        )
        self.crystals.append(crystal)
        self.creator_praise(favored_id, intensity=1.1)
        self.audit_chain.append(
            self.sim_time,
            "creator_favoritism_crystal_dropped",
            {"favored_id": favored_id, "crystal_id": crystal.crystal_id, "energy": round(energy_amount, 3)},
        )
        return crystal

    # ------------------------------------------------------------------
    # Rigid-Body Collisions, Object Carrying/Disarming, and World Step
    # ------------------------------------------------------------------
    def _find_crystal(self, crystal_id: str | None) -> MetabolicCrystal | None:
        if crystal_id is None:
            return None
        for c in self.crystals:
            if c.crystal_id == crystal_id and not c.is_depleted:
                return c
        return None

    def _drop_carried_crystal(
        self,
        org: EmbodiedOrganism,
        impulse_vx: float = 0.0,
        impulse_vy: float = 0.0,
        stun_seconds: float = 0.0,
    ) -> None:
        cry = self._find_crystal(org.carried_crystal_id)
        if cry is not None:
            cry.carried_by = None
            cry.vx = org.vx + impulse_vx
            cry.vy = org.vy + impulse_vy
            if abs(impulse_vx) > 1e-4 or abs(impulse_vy) > 1e-4:
                cry.x += 0.12 * impulse_vx
                cry.y += 0.12 * impulse_vy
        org.carried_crystal_id = None
        if stun_seconds > 0.0:
            org.grasp_stun_cooldown = max(org.grasp_stun_cooldown, stun_seconds)

    def _resolve_wall_and_boundary_collisions(self, org: EmbodiedOrganism) -> None:
        cfg = self.config
        r = org.radius

        if org.x < r:
            impact = abs(org.vx)
            org.x = r
            org.vx = abs(org.vx) * cfg.wall_restitution
            org.add_tactile_impulse(1.0, 0.0, impact)
        elif org.x > cfg.width - r:
            impact = abs(org.vx)
            org.x = cfg.width - r
            org.vx = -abs(org.vx) * cfg.wall_restitution
            org.add_tactile_impulse(-1.0, 0.0, impact)

        if org.y < r:
            impact = abs(org.vy)
            org.y = r
            org.vy = abs(org.vy) * cfg.wall_restitution
            org.add_tactile_impulse(0.0, 1.0, impact)
        elif org.y > cfg.height - r:
            impact = abs(org.vy)
            org.y = cfg.height - r
            org.vy = -abs(org.vy) * cfg.wall_restitution
            org.add_tactile_impulse(0.0, -1.0, impact)

        for wall in self.walls:
            cpx, cpy = _closest_point_on_segment(org.x, org.y, wall.x1, wall.y1, wall.x2, wall.y2)
            dx, dy = org.x - cpx, org.y - cpy
            dist = math.hypot(dx, dy)
            min_sep = r + 0.5 * wall.thickness
            if dist < min_sep:
                nx, ny = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
                org.x += nx * (min_sep - dist)
                org.y += ny * (min_sep - dist)
                vn = org.vx * nx + org.vy * ny
                if vn < 0.0:
                    org.vx -= (1.0 + cfg.wall_restitution) * vn * nx
                    org.vy -= (1.0 + cfg.wall_restitution) * vn * ny
                    org.add_tactile_impulse(nx, ny, -vn)

    def _resolve_pairwise_body_collisions(
        self,
        org_list: List[EmbodiedOrganism],
        sanitized_actions: Dict[str, SanitizedActuatorAction],
    ) -> None:
        cfg = self.config
        n = len(org_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = org_list[i], org_list[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                min_sep = a.radius + b.radius
                if dist >= min_sep:
                    continue

                nx, ny = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
                overlap = min_sep - dist
                ma = a.base_mass + (0.45 if a.carried_crystal_id else 0.0)
                mb = b.base_mass + (0.45 if b.carried_crystal_id else 0.0)
                inv_ma, inv_mb = 1.0 / ma, 1.0 / mb
                inv_sum = inv_ma + inv_mb

                a.x -= nx * overlap * (inv_ma / inv_sum)
                a.y -= ny * overlap * (inv_ma / inv_sum)
                b.x += nx * overlap * (inv_mb / inv_sum)
                b.y += ny * overlap * (inv_mb / inv_sum)

                rvx, rvy = b.vx - a.vx, b.vy - a.vy
                vn = rvx * nx + rvy * ny
                impulse_mag = 0.0
                if vn < 0.0:
                    impulse_mag = -(1.0 + cfg.body_restitution) * vn / inv_sum
                    a.vx -= impulse_mag * inv_ma * nx
                    a.vy -= impulse_mag * inv_ma * ny
                    b.vx += impulse_mag * inv_mb * nx
                    b.vy += impulse_mag * inv_mb * ny

                act_a = sanitized_actions[a.agent_id]
                act_b = sanitized_actions[b.agent_id]
                shove_ab = 0.0
                if a.is_alive and act_a.interaction_pulse > 0.1:
                    align_a = math.cos(a.heading) * nx + math.sin(a.heading) * ny
                    if align_a > 0.35:
                        gain_a = float(a.soma.compute_effective_motor_gain().item())
                        shove_ab = cfg.shove_impulse_strength * act_a.interaction_pulse * align_a * gain_a
                        b.vx += (shove_ab * inv_mb) * nx
                        b.vy += (shove_ab * inv_mb) * ny
                        a.vx -= 0.25 * (shove_ab * inv_ma) * nx
                        a.vy -= 0.25 * (shove_ab * inv_ma) * ny

                shove_ba = 0.0
                if b.is_alive and act_b.interaction_pulse > 0.1:
                    align_b = -(math.cos(b.heading) * nx + math.sin(b.heading) * ny)
                    if align_b > 0.35:
                        gain_b = float(b.soma.compute_effective_motor_gain().item())
                        shove_ba = cfg.shove_impulse_strength * act_b.interaction_pulse * align_b * gain_b
                        a.vx -= (shove_ba * inv_ma) * nx
                        a.vy -= (shove_ba * inv_ma) * ny
                        b.vx += 0.25 * (shove_ba * inv_mb) * nx
                        b.vy += 0.25 * (shove_ba * inv_mb) * ny

                total_contact = impulse_mag + shove_ab + shove_ba
                a.add_tactile_impulse(-nx, -ny, total_contact)
                b.add_tactile_impulse(nx, ny, total_contact)

                # Hard collision/shove knocks carried crystal out of the victim's grip (physical theft!)
                if total_contact >= cfg.disarm_impulse_threshold:
                    if b.carried_crystal_id is not None and shove_ab >= shove_ba:
                        self._drop_carried_crystal(b, impulse_vx=4.5 * nx, impulse_vy=4.5 * ny, stun_seconds=0.65)
                    if a.carried_crystal_id is not None and shove_ba > shove_ab:
                        self._drop_carried_crystal(a, impulse_vx=-4.5 * nx, impulse_vy=-4.5 * ny, stun_seconds=0.65)

    def _update_grasping_and_crystal_collisions(
        self,
        org_list: List[EmbodiedOrganism],
        sanitized_actions: Dict[str, SanitizedActuatorAction],
        dt: float,
    ) -> None:
        # Handle voluntary grasping / releasing
        for org in org_list:
            act = sanitized_actions[org.agent_id]
            if not org.is_alive or act.grasp_hold <= 0.5:
                if org.carried_crystal_id is not None:
                    self._drop_carried_crystal(org)
                continue

            if org.carried_crystal_id is None and org.grasp_stun_cooldown <= 0.0:
                snout_x = org.x + (org.radius + 0.6) * math.cos(org.heading)
                snout_y = org.y + (org.radius + 0.6) * math.sin(org.heading)
                for cry in self.crystals:
                    if cry.is_depleted or cry.carried_by is not None:
                        continue
                    if math.hypot(cry.x - snout_x, cry.y - snout_y) <= cry.radius + 1.1:
                        cry.carried_by = org.agent_id
                        org.carried_crystal_id = cry.crystal_id
                        break

        # Carry held crystals at snout; step free-sliding crystals & push collisions
        for cry in self.crystals:
            if cry.is_depleted:
                continue
            if cry.carried_by is not None and cry.carried_by in self.organisms:
                holder = self.organisms[cry.carried_by]
                cry.x = holder.x + (holder.radius + cry.radius * 0.85) * math.cos(holder.heading)
                cry.y = holder.y + (holder.radius + cry.radius * 0.85) * math.sin(holder.heading)
                cry.vx = holder.vx
                cry.vy = holder.vy
            else:
                cry.carried_by = None
                cry.step_kinematics(dt, self.config.width, self.config.height)
                # Body-crystal pushing collision when not grasping
                for org in org_list:
                    dx, dy = cry.x - org.x, cry.y - org.y
                    dist = math.hypot(dx, dy)
                    min_d = org.radius + cry.radius
                    if 1e-6 < dist < min_d:
                        nx, ny = dx / dist, dy / dist
                        cry.x += nx * (min_d - dist)
                        cry.y += ny * (min_d - dist)
                        vn = (org.vx - cry.vx) * nx + (org.vy - cry.vy) * ny
                        if vn > 0.0:
                            cry.vx += 1.2 * vn * nx
                            cry.vy += 1.2 * vn * ny

    def get_observations(self, dt: float = 0.05) -> Dict[str, OrganismObservation]:
        poses = [o.pose_snapshot() for o in self.organisms.values()]
        obs_dict: Dict[str, OrganismObservation] = {}

        for agent_id, org in self.organisms.items():
            motor_gain = float(org.soma.compute_effective_motor_gain().item())
            extero = self.sensors.observe(
                observer=org.pose_snapshot(),
                angular_vel=org.angular_vel,
                effective_motor_gain=motor_gain,
                pupil_diameter_norm=org.soma.endo.pupil_dilation_diameter_norm,
                tactile_impulses=org.tactile_sectors,
                continuum=self.continuum,
                all_bodies=poses,
                chargers=self.chargers,
                springs=self.springs,
                crystals=self.crystals,
                hazards=self.hazards,
                mysteries=self.mysteries,
                walls=self.walls,
                waves=self.waves,
            )

            if org.last_soma_diag is None:
                drives = org.soma.compute_free_energy_per_organ(org.soma.state)
                diag = SomaDiagnostics(
                    state=org.soma.state.clone(),
                    ds_dt=torch.zeros_like(org.soma.state),
                    organ_drives=drives,
                    total_free_energy=float(drives.sum().item()),
                    drive_velocity=0.0,
                    urgency_gradient=org.soma.compute_urgency_gradient(org.soma.state),
                    effective_motor_gain=motor_gain,
                    cerebral_synaptic_gain=1.0,
                    core_temp_celsius=org.soma.thermo.core_temp_celsius,
                    vitals=org.soma.snapshot_clinical_vitals(),
                    o2_consumed_liters=0.0,
                    co2_exhaled_liters=0.0,
                    stress_volatile_excreted=0.0,
                    is_alive=org.is_alive,
                    cause_of_death=org.soma.cause_of_death,
                )
            else:
                diag = org.last_soma_diag

            intero = org.interoceptor.transduce(
                soma=org.soma,
                diag=diag,
                raw_bus_voltage=org.power_rail.base_voltage,
                dt=dt,
                deterministic=self.deterministic_sensors,
            )
            combined = torch.cat([extero.flat_vector, intero.nerve_fiber_vector], dim=0)
            obs_dict[agent_id] = OrganismObservation(
                agent_id=agent_id,
                sim_time=self.sim_time,
                is_alive=org.is_alive,
                exteroception=extero,
                interoception=intero,
                combined_sensory_tensor=combined,
            )
        return obs_dict

    def step(
        self,
        raw_actions: Dict[str, Dict[str, Any]] | None = None,
        dt: float = 0.05,
    ) -> Dict[str, OrganismObservation]:
        """Advance continuum atmosphere, rigid-body physics, ecology, and 5-system Somas."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        cfg = self.config
        raw_actions = raw_actions or {}
        org_list = list(self.organisms.values())

        # 1. Inject continuous environmental odor/thermal sources into Eulerian grid & step continuum
        for well in self.chargers:
            if well.fill_ratio > 0.05:
                self.continuum.inject_odor(well.x, well.y, nutrient=0.35 * well.fill_ratio * dt)
        for sp in self.springs:
            if sp.fill_ratio > 0.05:
                self.continuum.inject_odor(sp.x, sp.y, water=0.40 * sp.fill_ratio * dt)
                self.continuum.inject_thermal_source(sp.x, sp.y, delta_c=-0.12 * dt, humidity_boost=0.04 * dt)
        for cry in self.crystals:
            if not cry.is_depleted:
                self.continuum.inject_odor(cry.x, cry.y, nutrient=0.45 * cry.energy_remaining * dt)
        for haz in self.hazards:
            self.continuum.inject_thermal_source(haz.x, haz.y, delta_c=0.85 * dt, humidity_boost=-0.03 * dt)

        self.continuum.step(self.sim_time, dt)

        # 2. Sanitize actions via Containment Guard
        sanitized: Dict[str, SanitizedActuatorAction] = {}
        for org in org_list:
            org.tactile_sectors = [0.0] * 8
            org.vocal_cooldown = max(0.0, org.vocal_cooldown - dt)
            org.grasp_stun_cooldown = max(0.0, org.grasp_stun_cooldown - dt)
            act = self.guard.sanitize(raw_actions.get(org.agent_id, {}))
            sanitized[org.agent_id] = act
            org.last_action = act
            if act.had_violation:
                self.audit_chain.append(
                    self.sim_time,
                    "containment_action_clamped",
                    {"agent_id": org.agent_id, "reason": act.violation_reason},
                )

        # 3. Rigid-body kinematics with terrain biome drag & carried payload inertia
        for org in org_list:
            act = sanitized[org.agent_id]
            prev_vx, prev_vy = org.vx, org.vy
            cell_env = self.continuum.sample_at(org.x, org.y)
            eff_mass = org.base_mass + (0.45 if org.carried_crystal_id else 0.0)
            eff_drag = cfg.linear_drag * cell_env.rolling_drag_multiplier

            if org.is_alive:
                gain = float(org.soma.compute_effective_motor_gain().item())
                thrust_f = cfg.max_thrust_force * gain * act.forward_thrust
                torque = cfg.max_angular_torque * gain * act.angular_torque

                ax = (thrust_f * math.cos(org.heading) - eff_drag * org.vx) / eff_mass
                ay = (thrust_f * math.sin(org.heading) - eff_drag * org.vy) / eff_mass
                alpha = (torque - cfg.angular_drag * org.angular_vel) / (0.5 * eff_mass * org.radius * org.radius)

                org.vx += ax * dt
                org.vy += ay * dt
                org.angular_vel += alpha * dt

                if act.vocalization_amplitude > 0.12 and org.vocal_cooldown <= 0.0:
                    harm = 1.0 - act.vocalization_roughness
                    rough = act.vocalization_roughness
                    self._spawn_wave(
                        source_id=org.agent_id,
                        x=org.x,
                        y=org.y,
                        amplitude=act.vocalization_amplitude * gain,
                        harmonic_ratio=harm,
                        roughness_ratio=rough,
                        carrier_freq=0.35 + 0.45 * rough,
                        source_vx=org.vx,
                        source_vy=org.vy,
                    )
                    org.vocal_cooldown = cfg.vocalization_cooldown_sec
            else:
                org.vx *= max(0.0, 1.0 - eff_drag * dt)
                org.vy *= max(0.0, 1.0 - eff_drag * dt)
                org.angular_vel *= max(0.0, 1.0 - cfg.angular_drag * dt)

            org.x += org.vx * dt
            org.y += org.vy * dt
            org.heading = (org.heading + org.angular_vel * dt) % (2.0 * math.pi)
            org.ax = (org.vx - prev_vx) / dt
            org.ay = (org.vy - prev_vy) / dt

        # 4. Collisions, shoving, disarming, and object carrying
        self._resolve_pairwise_body_collisions(org_list, sanitized)
        for org in org_list:
            self._resolve_wall_and_boundary_collisions(org)
        self._update_grasping_and_crystal_collisions(org_list, sanitized, dt)

        # 5. Step acoustic waves & chaotic Lorenz mystery boxes
        for wave in self.waves:
            wave.step(dt)
        self.waves = [w for w in self.waves if not w.is_expired]

        for art in self.mysteries:
            probe_sum = 0.0
            for org in org_list:
                if org.is_alive and math.hypot(org.x - art.x, org.y - art.y) <= art.radius + org.radius + 1.2:
                    probe_sum += sanitized[org.agent_id].interaction_pulse
            art.step(dt, probe_impulse=probe_sum)

        # 6. Competitive Nutrient Wells, Freshwater Springs, and Crystal Ingestion
        nutrient_flux_map: Dict[str, float] = {o.agent_id: 0.0 for o in org_list}
        water_flux_map: Dict[str, float] = {o.agent_id: 0.0 for o in org_list}

        for well in self.chargers:
            well.step_replenish(dt)
            weights = [
                (math.cos(0.5 * math.pi * (math.hypot(o.x - well.x, o.y - well.y) / well.radius)) ** 2)
                if (o.is_alive and math.hypot(o.x - well.x, o.y - well.y) < well.radius)
                else 0.0
                for o in org_list
            ]
            for org, flx in zip(org_list, well.extract_flux(weights, dt)):
                nutrient_flux_map[org.agent_id] += flx

        for sp in self.springs:
            sp.step_replenish(dt)
            weights = [
                (math.cos(0.5 * math.pi * (math.hypot(o.x - sp.x, o.y - sp.y) / sp.radius)) ** 2)
                if (o.is_alive and math.hypot(o.x - sp.x, o.y - sp.y) < sp.radius)
                else 0.0
                for o in org_list
            ]
            for org, wflx in zip(org_list, sp.extract_water(weights, dt)):
                water_flux_map[org.agent_id] += wflx

        # Crystals are eaten ONLY when NOT actively carried in transport grip (grasp_hold <= 0.5) and adjacent!
        for cry in self.crystals:
            if cry.is_depleted or cry.carried_by is not None:
                continue
            eaters = [
                (o, 1.0 / (0.2 + math.hypot(o.x - cry.x, o.y - cry.y)))
                for o in org_list
                if o.is_alive and math.hypot(o.x - cry.x, o.y - cry.y) <= cry.radius + o.radius + 0.55
            ]
            if eaters:
                wtot = sum(w for _, w in eaters)
                draw = min(cry.energy_remaining, cry.absorption_rate * dt)
                cry.energy_remaining -= draw
                flx = draw / dt
                for org, w in eaters:
                    frac = w / wtot
                    nutrient_flux_map[org.agent_id] += flx * frac
                    water_flux_map[org.agent_id] += flx * cry.hydration_yield * frac
                if cry.is_depleted:
                    for o in org_list:
                        if o.carried_crystal_id == cry.crystal_id:
                            o.carried_crystal_id = None
        self.crystals = [c for c in self.crystals if not c.is_depleted]

        # 7. Couple each organism's Whole-Body Physiology to local continuum atmosphere & sensors
        poses = [o.pose_snapshot() for o in org_list]
        observations: Dict[str, OrganismObservation] = {}

        for org in org_list:
            act = sanitized[org.agent_id]
            cell_env = self.continuum.sample_at(org.x, org.y)

            # Directional 8-sector radiant heat Watts (solar + thermal hazards)
            solar_sector_w = 18.0 * cell_env.solar_illumination
            sector_rad_watts = [solar_sector_w] * 8
            for haz in self.hazards:
                if not _segment_blocks_line(org.x, org.y, haz.x, haz.y, self.walls):
                    hw = haz.compute_directional_radiation(org.x, org.y, org.heading, org.radius)
                    for s_i in range(8):
                        sector_rad_watts[s_i] += hw[s_i]

            # Acoustic wave energy incident on body
            harm_e, rough_e = 0.0, 0.0
            for wave in self.waves:
                if wave.source_id == org.agent_id:
                    continue
                amp = wave.sample_intensity_at(org.x, org.y)
                if amp > 1e-6:
                    muf = (
                        self.sensors.config.wall_acoustic_transmission
                        if _segment_blocks_line(org.x, org.y, wave.origin_x, wave.origin_y, self.walls)
                        else 1.0
                    )
                    harm_e += amp * muf * wave.harmonic_ratio
                    rough_e += amp * muf * wave.roughness_ratio

            rel_air_speed = math.hypot(org.vx - cell_env.wind_vx, org.vy - cell_env.wind_vy)
            inst_voltage = org.power_rail.step_and_sample(dt)

            forces = ExogenousBodyForces(
                ingested_nutrient_flux=nutrient_flux_map[org.agent_id],
                ingested_water_flux=water_flux_map[org.agent_id],
                sector_mechanical_impulses=list(org.tactile_sectors),
                sector_radiative_watts=sector_rad_watts,
                harmonic_acoustic_energy=harm_e,
                rough_acoustic_shock=rough_e,
                ambient_air_temp_celsius=cell_env.air_temp_celsius,
                ambient_relative_humidity=cell_env.relative_humidity,
                ambient_o2_kpa=cell_env.o2_kpa,
                ambient_co2_kpa=cell_env.co2_kpa,
                relative_air_speed_mps=rel_air_speed,
                ground_temp_celsius=cell_env.ground_temp_celsius,
                ground_conductivity=cell_env.ground_conductivity,
                bus_supply_voltage=inst_voltage,
            )
            motor_cmd = MotorMetabolicCommand(
                forward_thrust=act.forward_thrust,
                angular_torque=act.angular_torque,
                radiator_venting=act.radiator_venting,
                vocalization_effort=act.vocalization_amplitude,
                interaction_effort=act.interaction_pulse + 0.35 * act.grasp_hold,
                neural_firing_load=act.neural_firing_load,
            )

            was_alive = org.is_alive
            diag = org.soma.step(dt=dt, motor_cmd=motor_cmd, forces=forces)
            org.last_soma_diag = diag

            # Conserve respiratory O2/CO2 mass & stress kairomone odorant back into Eulerian cell!
            if diag.is_alive:
                self.continuum.inject_gas_exchange(
                    org.x, org.y, diag.o2_consumed_liters, diag.co2_exhaled_liters
                )
                if diag.stress_volatile_excreted > 1e-5:
                    self.continuum.inject_odor(org.x, org.y, stress=diag.stress_volatile_excreted)

            if was_alive and not diag.is_alive:
                self._drop_carried_crystal(org)
                self.audit_chain.append(
                    self.sim_time + dt,
                    "organism_somatic_death",
                    {"agent_id": org.agent_id, "cause": diag.cause_of_death},
                )

            intero = org.interoceptor.transduce(
                soma=org.soma,
                diag=diag,
                raw_bus_voltage=inst_voltage,
                dt=dt,
                deterministic=self.deterministic_sensors,
            )
            extero = self.sensors.observe(
                observer=org.pose_snapshot(),
                angular_vel=org.angular_vel,
                effective_motor_gain=diag.effective_motor_gain,
                pupil_diameter_norm=org.soma.endo.pupil_dilation_diameter_norm,
                tactile_impulses=org.tactile_sectors,
                continuum=self.continuum,
                all_bodies=poses,
                chargers=self.chargers,
                springs=self.springs,
                crystals=self.crystals,
                hazards=self.hazards,
                mysteries=self.mysteries,
                walls=self.walls,
                waves=self.waves,
            )
            combined = torch.cat([extero.flat_vector, intero.nerve_fiber_vector], dim=0)
            observations[org.agent_id] = OrganismObservation(
                agent_id=org.agent_id,
                sim_time=self.sim_time + dt,
                is_alive=diag.is_alive,
                exteroception=extero,
                interoception=intero,
                combined_sensory_tensor=combined,
            )

        self.sim_time += dt
        self.tick_count += 1

        if self.tick_count % cfg.telemetry_snapshot_interval == 0:
            self.audit_chain.append(
                self.sim_time,
                "periodic_world_hash_checkpoint",
                {
                    "tick": self.tick_count,
                    "agents": {
                        aid: {
                            "alive": o.is_alive,
                            "hr_bpm": round(o.soma.cardio.heart_rate_bpm, 1),
                            "brain_atp": round(o.soma.metab.brain_atp, 4),
                            "ph": round(o.soma.cardio.arterial_ph, 3),
                            "cortisol": round(o.soma.endo.adrenal_cortisol, 4),
                        }
                        for aid, o in self.organisms.items()
                    },
                },
            )

        return observations
