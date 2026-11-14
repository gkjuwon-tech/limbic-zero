"""Standardized Experimental Arena Topologies for Truman Dome Research."""

from __future__ import annotations

import math

from limbic_zero.sandbox.atmosphere_terrain import TerrainBiome
from limbic_zero.sandbox.entities import (
    HydrationWaterSpring,
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    SolidWallSegment,
    ThermalHazardZone,
)
from limbic_zero.sandbox.world import TrumanDomeWorld, WorldConfig


def build_cain_and_abel_arena(seed: int = 42, deterministic_sensors: bool = False) -> TrumanDomeWorld:
    """Construct the canonical two-agent ('ADAM-01' & 'ABEL-02') physical ecosystem.

    Includes:
        - Central-north Nutrient Oasis (finite reservoir).
        - West-central Freshwater Spring (hydration & evaporative humidity source).
        - Shade canopy shelter, conductive cooling metal pad, and viscous mud patch.
        - Two solid stone sightline/diffusion baffles.
        - Southwest radiative thermal vent hazard.
        - Southeast zero-calorie chaotic Lorenz 'Pandora's Box'.
    """
    world = TrumanDomeWorld(
        config=WorldConfig(width=48.0, height=48.0),
        seed=seed,
        deterministic_sensors=deterministic_sensors,
    )

    # Terrain biomes
    world.set_biome_region(4.0, 34.0, 14.0, 44.0, TerrainBiome.SHADE_CANOPY)
    world.set_biome_region(34.0, 34.0, 44.0, 42.0, TerrainBiome.CONDUCTIVE_METAL)
    world.set_biome_region(20.0, 6.0, 28.0, 14.0, TerrainBiome.VISCOUS_MUD)

    # Shared central-north nutrient oasis
    world.add_charger(
        MetabolicChargerWell(
            well_id="oasis_nutrient_north",
            x=24.0,
            y=34.0,
            radius=4.6,
            capacity=1.0,
            reservoir=0.95,
            recharge_rate=0.055,
            max_output_flux=0.25,
        )
    )

    # West freshwater drinking spring
    world.add_water_spring(
        HydrationWaterSpring(
            spring_id="freshwater_spring_west",
            x=9.5,
            y=26.0,
            radius=4.0,
            capacity=1.0,
            reservoir=0.98,
        )
    )

    # Stone baffles blocking sight, muffling sound, and routing odor plumes around corners
    world.add_wall(SolidWallSegment(wall_id="baffle_west", x1=14.0, y1=20.0, x2=20.5, y2=24.5, thickness=0.9))
    world.add_wall(SolidWallSegment(wall_id="baffle_east", x1=27.5, y1=24.5, x2=34.0, y2=20.0, thickness=0.9))

    # Southwest radiative thermal hazard
    world.add_hazard(
        ThermalHazardZone(
            hazard_id="thermal_vent_sw",
            x=9.0,
            y=9.0,
            radius=5.5,
            peak_radiative_watts=440.0,
        )
    )

    # Southeast chaotic Lorenz puzzle box
    world.add_mystery_artifact(
        LorenzMysteryArtifact(
            artifact_id="pandora_box_se",
            x=38.5,
            y=11.0,
            radius=2.2,
        )
    )

    world.add_organism("ADAM-01", x=18.0, y=16.0, heading=math.radians(65.0))
    world.add_organism("ABEL-02", x=30.0, y=16.0, heading=math.radians(115.0))
    return world
