"""Anatomical Ascending Interoceptive Tracts & Active Visceral Predictive Coding.

Transduces whole-body physiology from ``DigitalSoma`` into anatomically distinct
ascending nerve bundles:
    1. Lamina I Spinothalamic Tract (8-sector sharp A-delta pain, 8-sector slow
       C-fiber burning ache, 8-sector cutaneous warm/cold thermoreceptors).
    2. Vagal Nodose -> Nucleus Tractus Solitarius (NTS) Visceral Tract
       (carotid baroreceptor stretch, carotid-body hypoxic glomus, gut nutrient/water
       distension, IL-6 vagal sickness afferents, cardiorespiratory rhythm).
    3. Central Medullary & Circumventricular Sensors (Retrotrapezoid CO2/pH
       suffocation alarm, OVLT osmoreceptor thirst, POAH hypothalamic fever/chill error,
       and bioelectric supply-rail AC brownout ripple).
    4. Precision-Weighted Visceral Predictive Coding (tonic deviations, phasic derivatives,
       analytical Free-Energy urgency, internal generative prediction error, and
       Keramati-Gutkin drive-reduction valence ``-dF/dt``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from limbic_zero.core.soma import DigitalSoma, NUM_VISCERAL_ORGANS, SomaDiagnostics


@dataclass(frozen=True)
class InteroceptiveConfig:
    """Biophysical time constants and transduction noise of ascending visceral nerves."""

    prediction_horizon_tau: float = 0.90
    ripple_bandpass_fast_tau: float = 0.10
    ripple_bandpass_slow_tau: float = 1.25
    sensory_noise_std: float = 0.0012
    urgency_soft_clip: float = 14.0


@dataclass
class InteroceptivePercept:
    """Complete multi-tract ascending visceral nerve bundle delivered to the brain."""

    # 12D homeostatic predictive coding channels
    tonic_deviation: torch.Tensor               # (12,) s(t) - s*
    phasic_derivative: torch.Tensor             # (12,) ds/dt
    scaled_urgency: torch.Tensor                # (12,) tanh-compressed -grad_s F(s)
    prediction_error: torch.Tensor              # (12,) s(t) - s_hat(t)

    # Specific anatomical ascending tracts
    spinothalamic_a_delta: torch.Tensor         # (8,) fast localized sharp pain
    spinothalamic_c_fiber: torch.Tensor         # (8,) slow sensitized inflammatory ache
    cutaneous_thermo_afferents: torch.Tensor    # (8,) skin sector deviation from neutral 33.4 C
    medullary_vagal_chemo_tract: torch.Tensor   # (10,) [CO2_alarm, O2_glomus, thirst_OVLT, hunger_glycemic, baro, gut_fill, lactate_burn, cytokine_malaise, cardiac_awareness, shivering_sense]

    # Summary scalar physiological signals
    homeostatic_valence: float                  # -dF/dt (positive = physiological relief)
    total_drive: float                          # F(s(t)) >= 0
    rail_ripple_intensity: float                # High-frequency supply-bus flicker >= 0
    somatic_pain_salience: float                # Integrated gated nociception [0, 1]
    suffocation_alarm_salience: float           # Integrated CO2 + hypoxic air-hunger [0, 1]
    thirst_osmo_salience: float                 # OVLT plasma hyperosmolality drive [0, 1]

    nerve_fiber_vector: torch.Tensor            # Concatenated 86D ascending visceral tensor


class InteroceptiveNerveBundle(nn.Module):
    """Stateful ascending interoceptive pathway with forward visceral generative predictor."""

    def __init__(
        self,
        setpoints: torch.Tensor,
        config: InteroceptiveConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or InteroceptiveConfig()
        self.dtype = dtype
        self._device = torch.device(device)

        self.register_buffer("setpoints", setpoints.detach().clone().to(dtype=dtype, device=self._device))
        self.register_buffer("expected_state", self.setpoints.clone())
        self.register_buffer("expected_ds_dt", torch.zeros(NUM_VISCERAL_ORGANS, dtype=dtype, device=self._device))
        self.register_buffer("rail_fast_ema", torch.tensor(1.0, dtype=dtype, device=self._device))
        self.register_buffer("rail_slow_ema", torch.tensor(1.0, dtype=dtype, device=self._device))
        self.register_buffer("rail_ripple_ema", torch.tensor(0.0, dtype=dtype, device=self._device))

    @property
    def fiber_dim(self) -> int:
        # 12*4 (48) + 8*3 (24) + 10 + 4 = 86 continuous ascending visceral fibers
        return NUM_VISCERAL_ORGANS * 4 + 24 + 10 + 4

    def reset(self, initial_state: torch.Tensor | None = None) -> None:
        base = self.setpoints if initial_state is None else initial_state.to(dtype=self.dtype, device=self._device)
        self.expected_state.copy_(base)
        self.expected_ds_dt.zero_()
        rail_init = float(base[11].item())
        self.rail_fast_ema.fill_(rail_init)
        self.rail_slow_ema.fill_(rail_init)
        self.rail_ripple_ema.zero_()

    def transduce(
        self,
        soma: DigitalSoma,
        diag: SomaDiagnostics,
        raw_bus_voltage: float,
        dt: float,
        deterministic: bool = False,
    ) -> InteroceptivePercept:
        """Transduce whole-body organs and ``SomaDiagnostics`` into 86 ascending visceral fibers."""
        cfg = self.config
        s_obs = diag.state.to(dtype=self.dtype, device=self._device)
        ds_obs = diag.ds_dt.to(dtype=self.dtype, device=self._device)

        if not deterministic and cfg.sensory_noise_std > 0.0:
            s_sensed = torch.clamp(s_obs + torch.randn_like(s_obs) * cfg.sensory_noise_std, 0.0, 1.0)
        else:
            s_sensed = s_obs

        # 1. Forward visceral prediction & Active Interoceptive Inference error
        predicted_now = torch.clamp(self.expected_state + self.expected_ds_dt * dt, 0.0, 1.0)
        pred_error = s_sensed - predicted_now

        alpha_pred = min(1.0, dt / max(1e-4, cfg.prediction_horizon_tau))
        self.expected_state.copy_(predicted_now + alpha_pred * pred_error)
        self.expected_ds_dt.copy_((1.0 - alpha_pred) * self.expected_ds_dt + alpha_pred * ds_obs)

        # 2. Bioelectric power-rail AC brownout ripple bandpass filter
        v_tensor = torch.tensor(float(raw_bus_voltage), dtype=self.dtype, device=self._device)
        alpha_fast = min(1.0, dt / max(1e-4, cfg.ripple_bandpass_fast_tau))
        alpha_slow = min(1.0, dt / max(1e-4, cfg.ripple_bandpass_slow_tau))
        self.rail_fast_ema.copy_((1.0 - alpha_fast) * self.rail_fast_ema + alpha_fast * v_tensor)
        self.rail_slow_ema.copy_((1.0 - alpha_slow) * self.rail_slow_ema + alpha_slow * v_tensor)
        inst_ripple = torch.abs(self.rail_fast_ema - self.rail_slow_ema)
        self.rail_ripple_ema.copy_(0.84 * self.rail_ripple_ema + 0.16 * inst_ripple)
        ripple_val = float(self.rail_ripple_ema.item())

        # 3. Lamina I Spinothalamic 8-sector pain & cutaneous thermoreceptors
        a_delta = torch.tensor(soma.tissue.a_delta_sharp_pain, dtype=self.dtype, device=self._device)
        c_fiber = torch.tensor(soma.tissue.c_fiber_burning_ache, dtype=self.dtype, device=self._device)
        skin_thermo = torch.tensor(
            [math.tanh((t - 33.4) / 6.0) for t in soma.thermo.skin_sector_temps_celsius],
            dtype=self.dtype,
            device=self._device,
        )

        # 4. Medullary, Vagal NTS, and Circumventricular Organ (OVLT/Area Postrema) Afferents
        co2_alarm = float(soma.cardio.central_co2_chemoreceptor_drive)
        o2_glomus = float(soma.cardio.peripheral_hypoxic_drive)
        thirst_ovlt = math.tanh(max(0.0, soma.metab.plasma_osmolality_mosm - 289.0) / 14.0)
        hunger_glycemic = math.tanh(max(0.0, 90.0 - soma.metab.blood_glucose_mg_dl) / 25.0 + 0.4 * max(0.0, 0.70 - soma.metab.liver_glycogen))
        baro_stretch = float(soma.cardio.carotid_baroreceptor_hz_norm)
        gut_fill = math.tanh(soma.metab.gut_nutrient_lumen + soma.metab.gut_water_lumen)
        muscle_metaboreceptor_burn = math.tanh(max(0.0, soma.metab.blood_lactate_mmoll - 1.5) / 6.0)
        cytokine_sickness = float(soma.endo.il6_cytokine)
        cardiac_palpitation = math.tanh(max(0.0, (soma.cardio.heart_rate_bpm - 75.0) / 70.0) * (soma.cardio.stroke_volume_ml / 70.0))
        shivering_proprio = float(soma.thermo.shivering_thermogenesis_cmd)

        medullary_tract = torch.tensor(
            [
                co2_alarm,
                o2_glomus,
                thirst_ovlt,
                hunger_glycemic,
                baro_stretch,
                gut_fill,
                muscle_metaboreceptor_burn,
                cytokine_sickness,
                cardiac_palpitation,
                shivering_proprio,
            ],
            dtype=self.dtype,
            device=self._device,
        )

        tonic_dev = s_sensed - self.setpoints
        scaled_urg = torch.tanh(diag.urgency_gradient.to(dtype=self.dtype, device=self._device) / cfg.urgency_soft_clip)
        homeo_valence = -float(diag.drive_velocity)

        pain_salience = math.tanh(float(0.6 * a_delta.sum().item() + 0.45 * c_fiber.sum().item()))
        suffocation_salience = math.tanh(1.4 * co2_alarm + 1.2 * o2_glomus)

        summary_scalars = torch.tensor(
            [
                math.tanh(homeo_valence * 0.4),
                math.tanh(diag.total_free_energy * 0.18),
                min(1.0, ripple_val * 5.0),
                pain_salience,
            ],
            dtype=self.dtype,
            device=self._device,
        )

        fiber_vec = torch.cat(
            [
                tonic_dev,
                torch.tanh(ds_obs * 2.0),
                scaled_urg,
                pred_error * 4.0,
                torch.tanh(a_delta),
                torch.tanh(c_fiber),
                skin_thermo,
                medullary_tract,
                summary_scalars,
            ],
            dim=0,
        )

        return InteroceptivePercept(
            tonic_deviation=tonic_dev,
            phasic_derivative=ds_obs,
            scaled_urgency=scaled_urg,
            prediction_error=pred_error,
            spinothalamic_a_delta=a_delta,
            spinothalamic_c_fiber=c_fiber,
            cutaneous_thermo_afferents=skin_thermo,
            medullary_vagal_chemo_tract=medullary_tract,
            homeostatic_valence=homeo_valence,
            total_drive=diag.total_free_energy,
            rail_ripple_intensity=ripple_val,
            somatic_pain_salience=pain_salience,
            suffocation_alarm_salience=suffocation_salience,
            thirst_osmo_salience=thirst_ovlt,
            nerve_fiber_vector=fiber_vec,
        )
