"""Anatomical Exteroceptive Sensory Organs: Foveated Retina, Doppler Cochlea, Birhinal Nostrils, Vestibular & Skin.

Strict Anti-Omniscience & Biophysical Realism:
    1. Foveated 32-Ray Retina (dense high-acuity central fovea + wide peripheral rods):
       Effective visual range & contrast depend on ``solar_illumination`` AND the
       organism's autonomic ``pupil_diameter_norm`` (sympathetic mydriasis boosts
       night vision in darkness while causing slight glare scatter at noon!).
       Exact analytical wall/body occlusion prevents seeing through barriers.
    2. Binaural Doppler Cochlea (Left & Right ears on opposite sides of the head):
       Measures Interaural Level Difference (head-shadowing ILD), wall transmission
       muffling, harmonic vs rough spectral energy, and radial **Doppler pitch shift**!
    3. Birhinal Stereo Olfactory Nostrils (Left & Right nostrils on snout):
       Samples the 2D Eulerian continuum grid for ``[nutrient, water, stress_kairomone, co2]``
       so organisms can smell food, water, suffocation pockets, AND another agent's
       sweated adrenaline fear trail curving around walls!
    4. Vestibular Labyrinth + Carried-Load Proprioception:
       Otolith linear acceleration, semicircular canal angular rate, body-frame velocity,
       muscle contractility gain, and jaw/grip payload sensation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import torch

from limbic_zero.sandbox.atmosphere_terrain import ArenaContinuumGrid
from limbic_zero.sandbox.entities import (
    HydrationWaterSpring,
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    MetabolicCrystal,
    PropagatingAcousticWave,
    SolidWallSegment,
    ThermalHazardZone,
)


RAY_CHANNELS: int = 9
# Per-ray channels:
#   0: opaque_proximity       (1 - hit_dist/eff_range)
#   1: wall_signature         (1 if boundary/wall)
#   2: rival_signature        (1 if alive rival, 0.35 if corpse; +0.25 if rival carries crystal)
#   3: crystal_signature      (energy remaining if movable nutrient object)
#   4: mystery_u_sig          (Lorenz chaotic channel u)
#   5: mystery_v_sig          (Lorenz chaotic channel v)
#   6: charger_glow           (visible nutrient oasis glow before opaque hit)
#   7: water_spring_shimmer   (visible freshwater pool shimmer before opaque hit)
#   8: hazard_glow            (visible thermal vent glow before opaque hit)


@dataclass(frozen=True)
class SensorConfig:
    """Geometry and biophysical constants of the exteroceptive sense organs."""

    num_vision_rays: int = 33
    vision_fov_radians: float = math.radians(300.0)
    vision_nominal_range: float = 34.0
    num_tactile_sectors: int = 8
    nostril_separation_rad: float = math.radians(32.0)
    wall_acoustic_transmission: float = 0.30


@dataclass(frozen=True)
class BodyPoseSnapshot:
    """Physical pose & carried-object state of an organism for multi-body optics."""

    agent_id: str
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    heading: float
    radius: float
    is_alive: bool
    carrying_crystal: bool = False


@dataclass
class ExteroceptivePercept:
    """Complete body-centric sensory bundle delivered to the neural substrate."""

    vision_rays: torch.Tensor        # (33, 9) foveated retinal array
    binaural_ears: torch.Tensor      # (8,) [L_amp, L_harm, L_rough, L_doppler_freq, R_amp, R_harm, R_rough, R_doppler_freq]
    birhinal_nostrils: torch.Tensor  # (8,) [L_nutr, L_water, L_stress_fear, L_co2, R_nutr, R_water, R_stress_fear, R_co2]
    tactile_ring: torch.Tensor       # (8,) cutaneous mechanical pressure sectors
    vestibular_proprio: torch.Tensor # (7,) [fwd_vel, lat_vel, fwd_accel, lat_accel, ang_vel, motor_gain, carrying_flag]
    flat_vector: torch.Tensor        # Concatenated 328D 1D exteroceptive tensor


def _ray_circle_earliest_hit(
    ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, radius: float
) -> float | None:
    mx = ox - cx
    my = oy - cy
    b = mx * dx + my * dy
    c = mx * mx + my * my - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    t1 = -b - sqrt_d
    if t1 > 1e-5:
        return t1
    t2 = -b + sqrt_d
    if t2 > 1e-5 and c <= 0.0:
        return 0.0
    return None


def _ray_segment_hit(
    ox: float, oy: float, dx: float, dy: float, ax: float, ay: float, bx: float, by: float
) -> float | None:
    v1x, v1y = ox - ax, oy - ay
    v2x, v2y = bx - ax, by - ay
    denom = -v2x * dy + v2y * dx
    if abs(denom) < 1e-9:
        return None
    t = (v2x * v1y - v2y * v1x) / denom
    u = (-v1x * dy + v1y * dx) / denom
    if t > 1e-5 and 0.0 <= u <= 1.0:
        return t
    return None


def _ray_capsule_wall_hit(
    ox: float, oy: float, dx: float, dy: float, wall: SolidWallSegment
) -> float | None:
    ux = wall.x2 - wall.x1
    uy = wall.y2 - wall.y1
    seg_len = math.hypot(ux, uy)
    half_r = 0.5 * wall.thickness
    candidates: List[float] = []
    if seg_len > 1e-6:
        nx = -uy / seg_len * half_r
        ny = ux / seg_len * half_r
        t_top = _ray_segment_hit(ox, oy, dx, dy, wall.x1 + nx, wall.y1 + ny, wall.x2 + nx, wall.y2 + ny)
        t_bot = _ray_segment_hit(ox, oy, dx, dy, wall.x1 - nx, wall.y1 - ny, wall.x2 - nx, wall.y2 - ny)
        if t_top is not None:
            candidates.append(t_top)
        if t_bot is not None:
            candidates.append(t_bot)
    t_c1 = _ray_circle_earliest_hit(ox, oy, dx, dy, wall.x1, wall.y1, half_r)
    t_c2 = _ray_circle_earliest_hit(ox, oy, dx, dy, wall.x2, wall.y2, half_r)
    if t_c1 is not None:
        candidates.append(t_c1)
    if t_c2 is not None:
        candidates.append(t_c2)
    return min(candidates) if candidates else None


def _segment_blocks_line(
    px: float, py: float, qx: float, qy: float, walls: Sequence[SolidWallSegment]
) -> bool:
    ux, uy = qx - px, qy - py
    dist = math.hypot(ux, uy)
    if dist < 1e-6:
        return False
    dx, dy = ux / dist, uy / dist
    for wall in walls:
        hit = _ray_capsule_wall_hit(px, py, dx, dy, wall)
        if hit is not None and hit < dist:
            return True
    return False


class EgocentricSensorSuite:
    """Computes strictly local, anatomically grounded multimodal sensory observations."""

    def __init__(self, config: SensorConfig | None = None, dtype: torch.dtype = torch.float32) -> None:
        self.config = config or SensorConfig()
        self.dtype = dtype

        # Build non-uniform Foveated Ray Distribution: dense in central fovea, wider in periphery!
        n = self.config.num_vision_rays
        if n == 1:
            self._rel_angles = [0.0]
        else:
            half = 0.5 * self.config.vision_fov_radians
            self._rel_angles = []
            for i in range(n):
                u = -1.0 + (2.0 * i) / (n - 1)  # [-1, +1]
                # Cubic compression packs ~2x higher angular resolution near u=0 (fovea centralis!)
                warped = 0.45 * u + 0.55 * (u ** 3)
                self._rel_angles.append(half * warped)

    @property
    def flat_dim(self) -> int:
        return self.config.num_vision_rays * RAY_CHANNELS + 8 + 8 + self.config.num_tactile_sectors + 7

    @staticmethod
    def _compute_arena_boundary_hit(
        ox: float, oy: float, dx: float, dy: float, width: float, height: float, max_r: float
    ) -> float:
        best = max_r + 1.0
        if dx > 1e-8:
            best = min(best, (width - ox) / dx)
        elif dx < -1e-8:
            best = min(best, (0.0 - ox) / dx)
        if dy > 1e-8:
            best = min(best, (height - oy) / dy)
        elif dy < -1e-8:
            best = min(best, (0.0 - oy) / dy)
        return max(0.0, best)

    def observe(
        self,
        observer: BodyPoseSnapshot,
        angular_vel: float,
        effective_motor_gain: float,
        pupil_diameter_norm: float,
        tactile_impulses: Sequence[float],
        continuum: ArenaContinuumGrid,
        all_bodies: Sequence[BodyPoseSnapshot],
        chargers: Sequence[MetabolicChargerWell],
        springs: Sequence[HydrationWaterSpring],
        crystals: Sequence[MetabolicCrystal],
        hazards: Sequence[ThermalHazardZone],
        mysteries: Sequence[LorenzMysteryArtifact],
        walls: Sequence[SolidWallSegment],
        waves: Sequence[PropagatingAcousticWave],
    ) -> ExteroceptivePercept:
        cfg = self.config
        ox, oy = observer.x, observer.y
        theta = observer.heading

        # Pupil Aperture & Photopic/Scotopic Retinal Range Physics:
        # In dark night (solar~0.06), wide pupil mydriasis drastically expands scotopic photon capture;
        # in bright daylight, excessive pupil dilation incurs optical glare scatter.
        local_env = continuum.sample_at(ox, oy)
        aperture_gain = 0.32 + 1.45 * (max(0.02, pupil_diameter_norm) ** 1.25)
        ambient_lux_factor = 0.36 + 0.72 * math.sqrt(max(0.0, local_env.solar_illumination))
        photon_capture = min(1.12, ambient_lux_factor * aperture_gain)
        glare_penalty = max(0.0, local_env.solar_illumination * pupil_diameter_norm - 0.65) * 0.28
        eff_max_r = cfg.vision_nominal_range * max(0.22, photon_capture - glare_penalty)

        # 1. Foveated 33-Ray Retina with Exact Occlusion
        ray_data = torch.zeros((cfg.num_vision_rays, RAY_CHANNELS), dtype=self.dtype)

        for r_idx, rel_ang in enumerate(self._rel_angles):
            ray_ang = theta + rel_ang
            dx, dy = math.cos(ray_ang), math.sin(ray_ang)
            eye_x = ox + observer.radius * dx
            eye_y = oy + observer.radius * dy

            best_t = eff_max_r
            hit_kind = "none"
            hit_payload: Tuple[float, float] = (0.0, 0.0)

            t_bound = self._compute_arena_boundary_hit(
                eye_x, eye_y, dx, dy, continuum.width, continuum.height, eff_max_r
            )
            if t_bound < best_t:
                best_t = t_bound
                hit_kind = "wall"

            for wall in walls:
                t_w = _ray_capsule_wall_hit(eye_x, eye_y, dx, dy, wall)
                if t_w is not None and t_w < best_t:
                    best_t = t_w
                    hit_kind = "wall"

            for other in all_bodies:
                if other.agent_id == observer.agent_id:
                    continue
                t_o = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, other.x, other.y, other.radius)
                if t_o is not None and t_o < best_t:
                    best_t = t_o
                    hit_kind = "rival"
                    sig = (1.0 if other.is_alive else 0.35) + (0.25 if other.carrying_crystal else 0.0)
                    hit_payload = (min(1.25, sig), 0.0)

            for cry in crystals:
                if cry.is_depleted or cry.carried_by == observer.agent_id:
                    continue
                t_c = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, cry.x, cry.y, cry.radius)
                if t_c is not None and t_c < best_t:
                    best_t = t_c
                    hit_kind = "crystal"
                    hit_payload = (min(1.0, cry.energy_remaining * 1.8), 0.0)

            for art in mysteries:
                t_m = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, art.x, art.y, art.radius)
                if t_m is not None and t_m < best_t:
                    best_t = t_m
                    hit_kind = "mystery"
                    sig3 = art.normalized_signal
                    hit_payload = (sig3[0], sig3[1])

            if hit_kind != "none":
                ray_data[r_idx, 0] = max(0.0, 1.0 - (best_t / eff_max_r))
                if hit_kind == "wall":
                    ray_data[r_idx, 1] = 1.0
                elif hit_kind == "rival":
                    ray_data[r_idx, 2] = hit_payload[0]
                elif hit_kind == "crystal":
                    ray_data[r_idx, 3] = hit_payload[0]
                elif hit_kind == "mystery":
                    ray_data[r_idx, 4] = hit_payload[0]
                    ray_data[r_idx, 5] = hit_payload[1]

            # Floor pools/vents visible up to first opaque obstruction best_t
            c_glow = 0.0
            for well in chargers:
                tw = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, well.x, well.y, well.radius)
                if tw is not None and tw < best_t:
                    c_glow = max(c_glow, well.fill_ratio * (1.0 - tw / eff_max_r))
            ray_data[r_idx, 6] = c_glow

            w_shimmer = 0.0
            for sp in springs:
                ts = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, sp.x, sp.y, sp.radius)
                if ts is not None and ts < best_t:
                    w_shimmer = max(w_shimmer, sp.fill_ratio * (1.0 - ts / eff_max_r))
            ray_data[r_idx, 7] = w_shimmer

            h_glow = 0.0
            for haz in hazards:
                th = _ray_circle_earliest_hit(eye_x, eye_y, dx, dy, haz.x, haz.y, haz.radius)
                if th is not None and th < best_t:
                    h_glow = max(h_glow, 1.0 - th / eff_max_r)
            ray_data[r_idx, 8] = h_glow

        # 2. Binaural Cochlear Ears with Head-Shadowing, Wall Muffling, and Doppler Frequency Shift
        left_nx, left_ny = -math.sin(theta), math.cos(theta)
        ear_coords = [
            (ox + observer.radius * left_nx, oy + observer.radius * left_ny, left_nx, left_ny),
            (ox - observer.radius * left_nx, oy - observer.radius * left_ny, -left_nx, -left_ny),
        ]
        ear_accum = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

        for wave in waves:
            if wave.source_id == observer.agent_id:
                continue
            for e_idx, (ex, ey, enx, eny) in enumerate(ear_coords):
                base_amp = wave.sample_intensity_at(ex, ey)
                if base_amp <= 1e-6:
                    continue
                sx, sy = wave.origin_x - ex, wave.origin_y - ey
                s_dist = math.hypot(sx, sy)
                head_shadow = (0.55 + 0.45 * ((sx / s_dist) * enx + (sy / s_dist) * eny)) if s_dist > 1e-6 else 1.0
                wall_factor = (
                    cfg.wall_acoustic_transmission
                    if _segment_blocks_line(ex, ey, wave.origin_x, wave.origin_y, walls)
                    else 1.0
                )
                eff_amp = base_amp * head_shadow * wall_factor
                doppler_f = wave.compute_doppler_freq_at(ex, ey, observer.vx, observer.vy)
                ear_accum[e_idx][0] += eff_amp
                ear_accum[e_idx][1] += eff_amp * wave.harmonic_ratio
                ear_accum[e_idx][2] += eff_amp * wave.roughness_ratio
                ear_accum[e_idx][3] += eff_amp * doppler_f

        binaural_vals: List[float] = []
        for acc in ear_accum:
            tot = acc[0]
            if tot > 1e-6:
                binaural_vals.extend([math.tanh(tot), acc[1] / tot, acc[2] / tot, acc[3] / tot])
            else:
                binaural_vals.extend([0.0, 0.0, 0.0, 0.0])
        binaural_tensor = torch.tensor(binaural_vals, dtype=self.dtype)

        # 3. Birhinal Stereo Olfactory Nostrils sampling Eulerian Advection-Diffusion Plumes!
        snout_r = observer.radius * 1.15
        ang_l = theta + cfg.nostril_separation_rad
        ang_r = theta - cfg.nostril_separation_rad
        env_l = continuum.sample_at(ox + snout_r * math.cos(ang_l), oy + snout_r * math.sin(ang_l))
        env_r = continuum.sample_at(ox + snout_r * math.cos(ang_r), oy + snout_r * math.sin(ang_r))

        nostril_tensor = torch.tensor(
            [
                math.tanh(env_l.odor_nutrient),
                math.tanh(env_l.odor_water),
                math.tanh(env_l.odor_stress_kairomone * 2.0),
                math.tanh((env_l.co2_kpa - 0.04) / 2.5),
                math.tanh(env_r.odor_nutrient),
                math.tanh(env_r.odor_water),
                math.tanh(env_r.odor_stress_kairomone * 2.0),
                math.tanh((env_r.co2_kpa - 0.04) / 2.5),
            ],
            dtype=self.dtype,
        )

        # 4. 8-Sector Tactile Ring
        tactile_tensor = torch.tensor(
            [math.tanh(float(v) / 4.0) for v in tactile_impulses[: cfg.num_tactile_sectors]],
            dtype=self.dtype,
        )

        # 5. Vestibular Otoliths (Acceleration), Semicircular Canals (Angular Vel), & Grip Proprioception
        fwd_x, fwd_y = math.cos(theta), math.sin(theta)
        fwd_vel = observer.vx * fwd_x + observer.vy * fwd_y
        lat_vel = observer.vx * left_nx + observer.vy * left_ny
        fwd_acc = observer.ax * fwd_x + observer.ay * fwd_y
        lat_acc = observer.ax * left_nx + observer.ay * left_ny

        vestibular_tensor = torch.tensor(
            [
                math.tanh(fwd_vel / 12.0),
                math.tanh(lat_vel / 12.0),
                math.tanh(fwd_acc / 25.0),
                math.tanh(lat_acc / 25.0),
                math.tanh(angular_vel / 6.0),
                float(effective_motor_gain),
                1.0 if observer.carrying_crystal else 0.0,
            ],
            dtype=self.dtype,
        )

        flat = torch.cat(
            [ray_data.reshape(-1), binaural_tensor, nostril_tensor, tactile_tensor, vestibular_tensor],
            dim=0,
        )

        return ExteroceptivePercept(
            vision_rays=ray_data,
            binaural_ears=binaural_tensor,
            birhinal_nostrils=nostril_tensor,
            tactile_ring=tactile_tensor,
            vestibular_proprio=vestibular_tensor,
            flat_vector=flat,
        )
