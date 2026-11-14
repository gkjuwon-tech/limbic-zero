"""Physical, Thermodynamic, Hydrological, Carryable, and Chaotic Entities Inside Truman Dome.

Includes:
    - ``MetabolicChargerWell``: Finite-reservoir nutrient oasis with competitive depletion.
    - ``HydrationWaterSpring``: Finite-reservoir freshwater pool replenishing body water
      and evaporating local humidity.
    - ``MovableNutrientFruit`` / ``MetabolicCrystal``: Rigid-body movable/carryable
      nutrient objects with mass, sliding friction, grasping/carrying, hoarding, and theft!
    - ``ThermalHazardZone``: Directional radiant heat (Watts) and thermal hazard vent.
    - ``LorenzMysteryArtifact``: Zero-calorie 3D chaotic Lorenz oscillator responding to
      tactile probing via phase perturbation.
    - ``SolidWallSegment``: Impermeable stone barrier blocking movement, sight, and gas/odor diffusion.
    - ``PropagatingAcousticWave``: Finite-speed pressure wave with source velocity for
      real radial Doppler frequency shift!
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple


@dataclass
class MetabolicChargerWell:
    """Finite-capacity thermodynamic nutrient spring."""

    well_id: str
    x: float
    y: float
    radius: float = 4.5
    capacity: float = 1.0
    reservoir: float = 1.0
    recharge_rate: float = 0.055
    max_output_flux: float = 0.26

    @property
    def fill_ratio(self) -> float:
        return max(0.0, min(1.0, self.reservoir / max(1e-6, self.capacity)))

    def step_replenish(self, dt: float) -> None:
        self.reservoir = min(
            self.capacity,
            self.reservoir + self.recharge_rate * (self.capacity - self.reservoir) * dt,
        )

    def extract_flux(self, demand_weights: List[float], dt: float) -> List[float]:
        if not demand_weights or self.reservoir <= 1e-6:
            return [0.0 for _ in demand_weights]
        total_demand = sum(max(0.0, w) for w in demand_weights)
        if total_demand <= 1e-8:
            return [0.0 for _ in demand_weights]

        pressure = math.sqrt(self.fill_ratio)
        desired_total = self.max_output_flux * pressure * min(1.35, total_demand)
        max_sustainable = (self.reservoir * 0.90) / max(1e-5, dt)
        actual_total = min(desired_total, max_sustainable)
        self.reservoir = max(0.0, self.reservoir - actual_total * dt)
        return [actual_total * (max(0.0, w) / total_demand) for w in demand_weights]


@dataclass
class HydrationWaterSpring:
    """Freshwater pool providing hydration intake and local evaporative cooling."""

    spring_id: str
    x: float
    y: float
    radius: float = 4.0
    capacity: float = 1.0
    reservoir: float = 1.0
    recharge_rate: float = 0.065
    max_drink_flux: float = 0.30

    @property
    def fill_ratio(self) -> float:
        return max(0.0, min(1.0, self.reservoir / max(1e-6, self.capacity)))

    def step_replenish(self, dt: float) -> None:
        self.reservoir = min(
            self.capacity,
            self.reservoir + self.recharge_rate * (self.capacity - self.reservoir) * dt,
        )

    def extract_water(self, demand_weights: List[float], dt: float) -> List[float]:
        if not demand_weights or self.reservoir <= 1e-6:
            return [0.0 for _ in demand_weights]
        total_demand = sum(max(0.0, w) for w in demand_weights)
        if total_demand <= 1e-8:
            return [0.0 for _ in demand_weights]

        actual_total = min(
            self.max_drink_flux * math.sqrt(self.fill_ratio) * min(1.3, total_demand),
            (self.reservoir * 0.90) / max(1e-5, dt),
        )
        self.reservoir = max(0.0, self.reservoir - actual_total * dt)
        return [actual_total * (max(0.0, w) / total_demand) for w in demand_weights]


@dataclass
class MetabolicCrystal:
    """Movable, pushable, and carryable physical nutrient object (hoardable/stealable!).

    Has real Newtonian velocity ``(vx, vy)``, mass, sliding friction, and ``carried_by``
    ownership state so organisms can physically pick up resources, hide them behind
    baffles, or knock them loose from a rival!
    """

    crystal_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 1.1
    mass: float = 0.45
    sliding_drag: float = 3.8
    energy_remaining: float = 0.55
    hydration_yield: float = 0.12
    absorption_rate: float = 0.38
    carried_by: str | None = None

    @property
    def is_depleted(self) -> bool:
        return self.energy_remaining <= 1e-4

    def step_kinematics(self, dt: float, arena_w: float, arena_h: float) -> None:
        """Update free-sliding rigid-body motion when not held in an organism's jaws/grip."""
        if self.carried_by is not None:
            return
        damp = max(0.0, 1.0 - self.sliding_drag * dt)
        self.vx *= damp
        self.vy *= damp
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x < self.radius:
            self.x = self.radius
            self.vx = abs(self.vx) * 0.5
        elif self.x > arena_w - self.radius:
            self.x = arena_w - self.radius
            self.vx = -abs(self.vx) * 0.5

        if self.y < self.radius:
            self.y = self.radius
            self.vy = abs(self.vy) * 0.5
        elif self.y > arena_h - self.radius:
            self.y = arena_h - self.radius
            self.vy = -abs(self.vy) * 0.5


@dataclass
class ThermalHazardZone:
    """Radiative thermal vent heating surrounding air and burning nearby skin sectors."""

    hazard_id: str
    x: float
    y: float
    radius: float = 5.5
    peak_radiative_watts: float = 420.0  # Strong radiant flux raising skin >43C TRPV1 threshold!

    def compute_directional_radiation(
        self, ox: float, oy: float, heading: float, body_radius: float
    ) -> List[float]:
        """Compute radiant heat (Watts) absorbed by each of the 8 body-centric skin sectors."""
        dx = self.x - ox
        dy = self.y - oy
        dist = math.hypot(dx, dy)
        if dist >= self.radius * 1.6:
            return [0.0] * 8

        closeness = max(0.0, 1.0 - (dist / (self.radius * 1.6)))
        base_watts = self.peak_radiative_watts * (closeness ** 2)

        ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
        sector_watts: List[float] = []
        for s_idx in range(8):
            sec_ang = heading + (2.0 * math.pi * (s_idx + 0.5)) / 8.0
            nx, ny = math.cos(sec_ang), math.sin(sec_ang)
            # Lambert's cosine law on skin sector normal facing the radiant source!
            incidence = max(0.08, nx * ux + ny * uy)
            sector_watts.append(base_watts * incidence * 0.32)
        return sector_watts


@dataclass
class LorenzMysteryArtifact:
    """Zero-calorie 3D chaotic Lorenz strange-attractor puzzle object ('Pandora's Box')."""

    artifact_id: str
    x: float
    y: float
    radius: float = 2.2
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    u: float = 1.1
    v: float = -1.5
    w: float = 24.0
    time_scale: float = 1.4
    perturbation_count: int = 0

    def _lorenz_rhs(self, u: float, v: float, w: float, probe_impulse: float) -> Tuple[float, float, float]:
        du = self.sigma * (v - u) + 3.5 * probe_impulse
        dv = u * (self.rho - w) - v
        dw = u * v - self.beta * w
        return du, dv, dw

    def step(self, dt: float, probe_impulse: float = 0.0) -> None:
        if probe_impulse > 0.05:
            self.perturbation_count += 1
        h = dt * self.time_scale
        u, v, w = self.u, self.v, self.w
        k1 = self._lorenz_rhs(u, v, w, probe_impulse)
        k2 = self._lorenz_rhs(u + 0.5 * h * k1[0], v + 0.5 * h * k1[1], w + 0.5 * h * k1[2], probe_impulse)
        k3 = self._lorenz_rhs(u + 0.5 * h * k2[0], v + 0.5 * h * k2[1], w + 0.5 * h * k2[2], probe_impulse)
        k4 = self._lorenz_rhs(u + h * k3[0], v + h * k3[1], w + h * k3[2], probe_impulse)
        self.u = max(-50.0, min(50.0, u + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])))
        self.v = max(-50.0, min(50.0, v + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])))
        self.w = max(0.0, min(65.0, w + (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])))

    @property
    def normalized_signal(self) -> Tuple[float, float, float]:
        return (
            math.tanh(self.u / 18.0),
            math.tanh(self.v / 22.0),
            math.tanh((self.w - 25.0) / 16.0),
        )


@dataclass(frozen=True)
class SolidWallSegment:
    """Rigid stone barrier blocking physical motion, vision, sound, and gas diffusion."""

    wall_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 0.9


@dataclass
class PropagatingAcousticWave:
    """Expanding 2D pressure wave supporting real radial Doppler shift & spectral bands."""

    wave_id: str
    source_id: str
    origin_x: float
    origin_y: float
    source_vx: float = 0.0
    source_vy: float = 0.0
    current_radius: float = 0.2
    propagation_speed: float = 30.0
    max_radius: float = 56.0
    initial_amplitude: float = 1.0
    harmonic_ratio: float = 0.8
    roughness_ratio: float = 0.2
    carrier_freq_norm: float = 0.5
    shell_thickness: float = 4.4

    @property
    def is_expired(self) -> bool:
        return self.current_radius > self.max_radius

    def step(self, dt: float) -> None:
        self.current_radius += self.propagation_speed * dt

    def sample_intensity_at(self, px: float, py: float) -> float:
        dist = math.hypot(px - self.origin_x, py - self.origin_y)
        radial_offset = abs(dist - self.current_radius)
        if radial_offset > self.shell_thickness:
            return 0.0
        envelope = 0.5 * (1.0 + math.cos(math.pi * radial_offset / self.shell_thickness))
        attenuation = 1.0 / (1.0 + 0.032 * dist * dist)
        return self.initial_amplitude * envelope * attenuation

    def compute_doppler_freq_at(self, px: float, py: float, listener_vx: float, listener_vy: float) -> float:
        """Classical acoustic Doppler shift from source & listener velocities along line-of-propagation."""
        dx = px - self.origin_x
        dy = py - self.origin_y
        dist = math.hypot(dx, dy)
        if dist < 1e-5:
            return self.carrier_freq_norm
        ux, uy = dx / dist, dy / dist
        c = self.propagation_speed
        v_src_along = max(-0.6 * c, min(0.6 * c, self.source_vx * ux + self.source_vy * uy))
        v_lis_along = max(-0.6 * c, min(0.6 * c, listener_vx * ux + listener_vy * uy))
        doppler_ratio = (c - v_lis_along) / max(5.0, c - v_src_along)
        return max(0.05, min(1.0, self.carrier_freq_norm * doppler_ratio))
