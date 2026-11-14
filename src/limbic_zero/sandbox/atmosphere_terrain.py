"""2D Eulerian Continuum Grid: Atmospheric O2/CO2, Thermal/Humidity, Odor Plumes, & Terrain Biomes.

Implements real finite-difference advection-diffusion continuum physics on a 2D
spatial grid covering the Truman Dome, with impermeable zero-flux Neumann boundary
conditions across solid wall segments so gases, heat, and odor plumes physically
flow AROUND walls rather than leaking through solid stone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence, Tuple

import numpy as np

from limbic_zero.sandbox.entities import SolidWallSegment


class TerrainBiome(IntEnum):
    """Physical surface biomes with distinct drag, thermal conductivity, and canopy shade."""

    LOAM_SOIL = 0          # Standard traction (drag 1.0x, ground cond 1.0x, solar 1.0x)
    VISCOUS_MUD = 1        # High mechanical drag (drag 2.4x, ground cond 1.5x, moist)
    CONDUCTIVE_METAL = 2   # High conductive heat sink/source (drag 0.85x, ground cond 4.2x)
    SHADE_CANOPY = 3       # Overhead roof blocking direct solar radiation (solar 0.08x)


@dataclass(frozen=True)
class ContinuumConfig:
    """Grid resolution and physical transport diffusivities."""

    grid_nx: int = 24
    grid_ny: int = 24
    base_air_temp_celsius: float = 23.0
    solar_peak_heating_c_per_sec: float = 0.06
    day_cycle_period_sec: float = 120.0      # Full day-night cycle every 120 sim seconds
    gas_diffusivity: float = 2.4             # m^2/s turbulent eddy diffusivity for O2/CO2
    thermal_diffusivity: float = 1.8         # m^2/s air thermal eddy diffusivity
    odor_diffusivity: float = 1.6            # m^2/s volatile odorant diffusivity
    odor_decay_rate: float = 0.055           # Natural volatile oxidation/settling per sec
    nominal_ventilation_rate: float = 0.045  # Louver exchange rate with exterior atmosphere
    cell_headspace_liters: float = 85.0      # Effective breathable air volume per 2m x 2m cell
    ecological_gas_exchange_scale: float = 60.0  # Compressed arena respiratory turnover matching 120s day cycle


@dataclass(frozen=True)
class LocalCellEnvironment:
    """Sampled physical continuum conditions at an organism's exact world position."""

    air_temp_celsius: float
    ground_temp_celsius: float
    ground_conductivity: float
    rolling_drag_multiplier: float
    relative_humidity: float
    o2_kpa: float
    co2_kpa: float
    solar_illumination: float
    wind_vx: float
    wind_vy: float
    odor_nutrient: float
    odor_water: float
    odor_stress_kairomone: float


class ArenaContinuumGrid:
    """2D finite-volume Eulerian atmosphere, odorant, solar, and terrain simulator."""

    def __init__(self, width: float, height: float, config: ContinuumConfig | None = None) -> None:
        self.width = float(width)
        self.height = float(height)
        self.config = config or ContinuumConfig()

        nx, ny = self.config.grid_nx, self.config.grid_ny
        self.dx = self.width / nx
        self.dy = self.height / ny

        # Spatial fields (shape: ny, nx)
        self.o2_kpa = np.full((ny, nx), 21.30, dtype=np.float64)
        self.co2_kpa = np.full((ny, nx), 0.04, dtype=np.float64)
        self.air_temp_c = np.full((ny, nx), self.config.base_air_temp_celsius, dtype=np.float64)
        self.ground_temp_c = np.full((ny, nx), self.config.base_air_temp_celsius - 1.0, dtype=np.float64)
        self.humidity_rel = np.full((ny, nx), 0.45, dtype=np.float64)

        self.odor_nutrient = np.zeros((ny, nx), dtype=np.float64)
        self.odor_water = np.zeros((ny, nx), dtype=np.float64)
        self.odor_stress = np.zeros((ny, nx), dtype=np.float64)

        self.terrain = np.full((ny, nx), int(TerrainBiome.LOAM_SOIL), dtype=np.int32)
        self.wall_mask = np.zeros((ny, nx), dtype=bool)  # True if solid wall occupies cell center

        # Controllable arena atmospheric boundary conditions
        self.ventilation_openness: float = 1.0  # [0.0 = hermetically sealed dome, 1.0 = normal vents]
        self.wind_vx: float = 0.65              # Gentle prevailing breeze (m/s)
        self.wind_vy: float = 0.20
        self.solar_illumination: float = 0.80   # [0.05 night, 1.00 noon]

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(max(0, min(self.config.grid_nx - 1, math.floor(x / self.dx))))
        gy = int(max(0, min(self.config.grid_ny - 1, math.floor(y / self.dy))))
        return gx, gy

    def set_biome_rect(self, x1: float, y1: float, x2: float, y2: float, biome: TerrainBiome) -> None:
        gx1, gy1 = self.world_to_cell(min(x1, x2), min(y1, y2))
        gx2, gy2 = self.world_to_cell(max(x1, x2), max(y1, y2))
        self.terrain[gy1 : gy2 + 1, gx1 : gx2 + 1] = int(biome)

    def rasterize_walls(self, walls: Sequence[SolidWallSegment]) -> None:
        """Rasterize solid wall segments onto ``wall_mask`` to block diffusion across walls."""
        self.wall_mask.fill(False)
        for wall in walls:
            length = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)
            steps = max(2, int(math.ceil(length / (0.35 * min(self.dx, self.dy)))))
            for s in range(steps + 1):
                u = s / float(steps)
                wx = wall.x1 + u * (wall.x2 - wall.x1)
                wy = wall.y1 + u * (wall.y2 - wall.y1)
                gx, gy = self.world_to_cell(wx, wy)
                self.wall_mask[gy, gx] = True

    def _diffuse_and_advect(
        self,
        field: np.ndarray,
        diffusivity: float,
        dt: float,
        advect: bool = True,
    ) -> np.ndarray:
        """Explicit conservative 2D diffusion + upwind wind advection with zero-flux walls."""
        h2 = min(self.dx, self.dy) ** 2
        # Stability clamp on Fourier number
        alpha = min(0.20, diffusivity * dt / max(1e-4, h2))
        open_mask = ~self.wall_mask

        # Shifted neighbors with Neumann outer edge padding
        up = np.vstack([field[0:1, :], field[:-1, :]])
        down = np.vstack([field[1:, :], field[-1:, :]])
        left = np.hstack([field[:, 0:1], field[:, :-1]])
        right = np.hstack([field[:, 1:], field[:, -1:]])

        up_open = np.vstack([open_mask[0:1, :], open_mask[:-1, :]])
        down_open = np.vstack([open_mask[1:, :], open_mask[-1:, :]])
        left_open = np.hstack([open_mask[:, 0:1], open_mask[:, :-1]])
        right_open = np.hstack([open_mask[:, 1:], open_mask[:, -1:]])

        # Zero-flux across wall faces: if neighbor is a wall, reflect center value
        f_up = np.where(up_open, up, field)
        f_down = np.where(down_open, down, field)
        f_left = np.where(left_open, left, field)
        f_right = np.where(right_open, right, field)

        laplacian = (f_up + f_down + f_left + f_right) - 4.0 * field
        updated = field + alpha * laplacian

        eff_wvx = self.wind_vx * min(1.0, max(0.0, self.ventilation_openness))
        eff_wvy = self.wind_vy * min(1.0, max(0.0, self.ventilation_openness))
        if advect and (abs(eff_wvx) > 1e-4 or abs(eff_wvy) > 1e-4):
            cx = min(0.22, abs(eff_wvx) * dt / self.dx)
            cy = min(0.22, abs(eff_wvy) * dt / self.dy)
            grad_x = (updated - f_left) if eff_wvx > 0.0 else (f_right - updated)
            grad_y = (updated - f_up) if eff_wvy > 0.0 else (f_down - updated)
            updated = updated - cx * grad_x - cy * grad_y

        return np.where(open_mask, updated, field)

    def inject_gas_exchange(self, x: float, y: float, o2_consumed_l: float, co2_exhaled_l: float) -> None:
        """Conserve respiratory O2 consumption & CO2 exhalation in local cell air."""
        gx, gy = self.world_to_cell(x, y)
        vol = self.config.cell_headspace_liters
        scale = self.config.ecological_gas_exchange_scale
        # 101.3 kPa atmospheric pressure * ecological volume fraction change
        do2_kpa = ((o2_consumed_l * scale) / vol) * 101.325
        dco2_kpa = ((co2_exhaled_l * scale) / vol) * 101.325
        self.o2_kpa[gy, gx] = max(2.0, self.o2_kpa[gy, gx] - do2_kpa)
        self.co2_kpa[gy, gx] = min(25.0, self.co2_kpa[gy, gx] + dco2_kpa)

    def inject_odor(self, x: float, y: float, nutrient: float = 0.0, water: float = 0.0, stress: float = 0.0) -> None:
        gx, gy = self.world_to_cell(x, y)
        if not self.wall_mask[gy, gx]:
            self.odor_nutrient[gy, gx] = min(5.0, self.odor_nutrient[gy, gx] + nutrient)
            self.odor_water[gy, gx] = min(5.0, self.odor_water[gy, gx] + water)
            self.odor_stress[gy, gx] = min(5.0, self.odor_stress[gy, gx] + stress)

    def inject_thermal_source(self, x: float, y: float, delta_c: float, humidity_boost: float = 0.0) -> None:
        gx, gy = self.world_to_cell(x, y)
        if not self.wall_mask[gy, gx]:
            self.air_temp_c[gy, gx] = max(-5.0, min(65.0, self.air_temp_c[gy, gx] + delta_c))
            if humidity_boost != 0.0:
                self.humidity_rel[gy, gx] = max(0.05, min(0.99, self.humidity_rel[gy, gx] + humidity_boost))

    def step(self, sim_time: float, dt: float) -> None:
        """Advance atmospheric diffusion, solar heating cycle, and ventilation exchange."""
        cfg = self.config

        # 1. Day-Night circadian solar irradiance cycle in [0.06, 1.00]
        solar_phase = 2.0 * math.pi * (sim_time / cfg.day_cycle_period_sec)
        self.solar_illumination = 0.53 + 0.47 * math.sin(solar_phase)

        # 2. Diffuse & advect O2, CO2, Air Temp, Humidity, and 3 Odorant fields
        self.o2_kpa = self._diffuse_and_advect(self.o2_kpa, cfg.gas_diffusivity, dt, advect=True)
        self.co2_kpa = self._diffuse_and_advect(self.co2_kpa, cfg.gas_diffusivity, dt, advect=True)
        self.air_temp_c = self._diffuse_and_advect(self.air_temp_c, cfg.thermal_diffusivity, dt, advect=True)
        self.humidity_rel = self._diffuse_and_advect(self.humidity_rel, cfg.odor_diffusivity, dt, advect=True)

        self.odor_nutrient = self._diffuse_and_advect(self.odor_nutrient, cfg.odor_diffusivity, dt, advect=True)
        self.odor_water = self._diffuse_and_advect(self.odor_water, cfg.odor_diffusivity, dt, advect=True)
        self.odor_stress = self._diffuse_and_advect(self.odor_stress, cfg.odor_diffusivity, dt, advect=True)

        # Natural volatile decay
        decay = max(0.0, 1.0 - cfg.odor_decay_rate * dt)
        self.odor_nutrient *= decay
        self.odor_water *= decay
        self.odor_stress *= decay

        # 3. Ventilation louver gas/thermal relaxation (zero when hermetically sealed!)
        vent = cfg.nominal_ventilation_rate * max(0.0, min(2.0, self.ventilation_openness)) * dt
        if vent > 0.0:
            self.o2_kpa += vent * (21.30 - self.o2_kpa)
            self.co2_kpa += vent * (0.04 - self.co2_kpa)
            self.humidity_rel += vent * (0.45 - self.humidity_rel)

        # 4. Solar ground heating (blocked in SHADE_CANOPY cells!) & air-ground coupling
        shade_factor = np.where(self.terrain == int(TerrainBiome.SHADE_CANOPY), 0.08, 1.0)
        solar_target_temp = cfg.base_air_temp_celsius - 2.5 + 8.5 * self.solar_illumination * shade_factor
        self.ground_temp_c += (solar_target_temp - self.ground_temp_c) * min(1.0, dt * 0.08)
        self.air_temp_c += (self.ground_temp_c - self.air_temp_c) * min(1.0, dt * 0.05)

    def sample_at(self, x: float, y: float) -> LocalCellEnvironment:
        """Bilinearly / cell-interpolated local environment at ``(x, y)``."""
        # Continuous grid coordinates centered at half-cells
        fx = max(0.0, min(self.config.grid_nx - 1.001, (x / self.dx) - 0.5))
        fy = max(0.0, min(self.config.grid_ny - 1.001, (y / self.dy) - 0.5))
        ix = int(math.floor(fx))
        iy = int(math.floor(fy))
        tx = fx - ix
        ty = fy - iy
        ix1 = min(self.config.grid_nx - 1, ix + 1)
        iy1 = min(self.config.grid_ny - 1, iy + 1)

        def bilerp(arr: np.ndarray) -> float:
            v00 = float(arr[iy, ix])
            v10 = float(arr[iy, ix1])
            v01 = float(arr[iy1, ix])
            v11 = float(arr[iy1, ix1])
            return (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) + ty * ((1.0 - tx) * v01 + tx * v11)

        gx, gy = self.world_to_cell(x, y)
        biome = TerrainBiome(int(self.terrain[gy, gx]))
        if biome == TerrainBiome.VISCOUS_MUD:
            drag_mult, cond_mult = 2.40, 1.50
        elif biome == TerrainBiome.CONDUCTIVE_METAL:
            drag_mult, cond_mult = 0.85, 4.20
        elif biome == TerrainBiome.SHADE_CANOPY:
            drag_mult, cond_mult = 1.00, 1.05
        else:
            drag_mult, cond_mult = 1.00, 1.00

        local_solar = self.solar_illumination * (0.12 if biome == TerrainBiome.SHADE_CANOPY else 1.0)

        return LocalCellEnvironment(
            air_temp_celsius=bilerp(self.air_temp_c),
            ground_temp_celsius=bilerp(self.ground_temp_c),
            ground_conductivity=cond_mult,
            rolling_drag_multiplier=drag_mult,
            relative_humidity=max(0.05, min(0.99, bilerp(self.humidity_rel))),
            o2_kpa=max(1.0, bilerp(self.o2_kpa)),
            co2_kpa=max(0.01, bilerp(self.co2_kpa)),
            solar_illumination=local_solar,
            wind_vx=self.wind_vx,
            wind_vy=self.wind_vy,
            odor_nutrient=max(0.0, bilerp(self.odor_nutrient)),
            odor_water=max(0.0, bilerp(self.odor_water)),
            odor_stress_kairomone=max(0.0, bilerp(self.odor_stress)),
        )
