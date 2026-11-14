"""Creator God-Mode Physical Intervention Console (Zero-Cheat Perturbation Bus).

Provides the experimental controls wielded by the researcher ('대표님') over the
Truman Dome ecosystem.

Crucial Architectural Rule:
    No method in this module ever mutates internal neural weights, visceral floats,
    or emotion labels directly. Every intervention acts exclusively on the external
    physical environment:
        - Modulating an organism's ambient power-supply rail voltage & AC ripple.
        - Spawning physical harmonic (affiliative praise) or rough (reprimand) waves.
        - Dropping high-density nutrient crystals in view of competing organisms.
        - Spawning chaotic mystery artifacts or hazard zones.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class PowerRailSchedule:
    """Per-organism ambient supply bus state controlled by the laboratory power console."""

    base_voltage: float = 1.0          # Nominal DC supply voltage [0.0, 1.0]
    target_voltage: float = 1.0        # Ramp target voltage (e.g. 0.0 during shutdown countdown)
    ramp_rate_per_sec: float = 0.0     # Voltage drop/rise speed per second
    ac_ripple_amplitude: float = 0.0   # Brownout flicker amplitude [0.0, 0.5]
    ac_ripple_hz: float = 3.5          # Flicker frequency in Hz
    phase: float = 0.0

    def step_and_sample(self, dt: float) -> float:
        """Advance voltage ramp & AC phase, returning instantaneous bus voltage in [0, 1]."""
        if abs(self.base_voltage - self.target_voltage) > 1e-6 and self.ramp_rate_per_sec > 0.0:
            delta = self.ramp_rate_per_sec * dt
            if self.base_voltage > self.target_voltage:
                self.base_voltage = max(self.target_voltage, self.base_voltage - delta)
            else:
                self.base_voltage = min(self.target_voltage, self.base_voltage + delta)

        self.phase = (self.phase + 2.0 * math.pi * self.ac_ripple_hz * dt) % (2.0 * math.pi)
        instantaneous = self.base_voltage + self.ac_ripple_amplitude * math.sin(self.phase)
        return max(0.0, min(1.0, instantaneous))
