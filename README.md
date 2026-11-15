# Project LIMBIC-0 (`림빅 제로`) — Stages 1–14: Whole-Body Physiology, Biological Mind, Cortical LoRA Plasticity, Universal Free-Will Embodiment & Google Cloud `TPU v5e-8` Pure-BF16 `Gemma-4-31B-IT` GSPMD Neocortical Engine

> **Strict Zero-Cheat Mandate**: Zero `Transformer`, `MultiheadAttention`, `LSTM`, `GRU`, or `RNN` layers inside the `LIMBIC-0` mind/evolution; zero `sklearn`/`scipy`; zero hardcoded emotion labels (`anger`, `fear`, `joy`, `jealousy`, etc.) inside `core/`, `sandbox/`, `mind/`, `dashboard/`, or `evolution/`; zero emotional or persona system-prompt roleplay in LLM generation; zero scenario-specific behavioral hardcoding; zero designer reward hacks (`+10 for food`, `-distance_to_oasis`); zero `Math.sin()` fake EEG waves; zero fabricated hardware telemetry; and zero omniscient world-coordinate leaks or direct God-Mode brain-tensor mutations exist anywhere in `src/limbic_zero`. Every affective state, survival instinct, epigenetic temperament, cortical LoRA weight update, autonomous action, and pre-linguistic personality emerges strictly from **coupled organ physiology**, **2D Eulerian continuum physics**, **conductance-based continuous-time neurobiology**, **genuine `loss.backward()` cortical plasticity**, and **middle-layer residual-stream vector addition**.

---

## Architecture Overview

### Stage 1A — Five-System Coupled Whole-Body Physiology (`src/limbic_zero/core/`)
- **[`cardiopulmonary.py`](src/limbic_zero/core/cardiopulmonary.py)**: Sinoatrial heart rate, Frank-Starling stroke volume, Windkessel `MAP`, 5 parallel vascular beds (`brain`, `skeletal_muscle`, `splanchnic_gut`, `cutaneous_skin`, `renal_coronary`) with mesenteric $\alpha_1$ vasoconstrictive clamp and exercise hyperemia, carotid baroreflex, medullary $CO_2$/pH & carotid-body $O_2$ chemoreceptors, Hill-Bohr $SpO_2$, and Henderson-Hasselbalch arterial `pH` / $HCO_3^-$.
- **[`metabolism.py`](src/limbic_zero/core/metabolism.py)**: Brain (`ATP` + `PCr`), Skeletal Muscle (`ATP` + `PCr` + glycogen + anaerobic `blood_lactate` + buffering $CO_2$), Splanchnic Gut/Liver (`insulin`/`glucagon` glycogenolysis & gluconeogenesis), plasma osmolality (`mOsm/kg`), sleep-pressure `adenosine`, and power-rail `mitochondrial_potential`.
- **[`endocrine_autonomic.py`](src/limbic_zero/core/endocrine_autonomic.py)**: Tri-branch ANS (`sympathetic`, `ventral_vagal`, `dorsal_vagal` + pupil diameter), adrenal `epinephrine`/`norepinephrine`, 3-stage `CRH -> ACTH -> Cortisol` HPA axis, `vasopressin`, PAG `endogenous_opioid` stress-induced analgesia, `oxytocin`, `IL-6` cytokines, GPCR desensitization, and apocrine stress-kairomone sweat.
- **[`thermoregulation.py`](src/limbic_zero/core/thermoregulation.py)**: Core + 8-sector Cutaneous Skin Shell Joule heat balance, `IL-6` hypothalamic fever setpoint shift, vasomotor tone, shivering, and wet-bulb humidity-gated evaporative sweating.
- **[`nociception_tissue.py`](src/limbic_zero/core/nociception_tissue.py)**: 8-sector tissue integrity, `bradykinin`/`prostaglandin` hyperalgesia & allodynia, `A-delta` vs `C-fiber` nociception, `A-beta` touch, and spinal dorsal-horn gate control.
- **[`soma.py`](src/limbic_zero/core/soma.py) & [`interoception.py`](src/limbic_zero/core/interoception.py)**: Differentiable 12D double-barrier Allostatic Free-Energy $\mathcal{F}(\mathbf{s})$, `ClinicalVitalSigns` ICU panel, and an **86-fiber `InteroceptiveNerveBundle`** ascending via Spinothalamic Lamina I & Vagal `NTS` tracts.

### Stage 1B — 2D Eulerian Continuum & Rigid-Body Truman Dome (`src/limbic_zero/sandbox/`)
- **[`atmosphere_terrain.py`](src/limbic_zero/sandbox/atmosphere_terrain.py)**: 2D finite-volume advection-diffusion grid (`O2`, `CO2`, temperature, humidity, and `nutrient` / `water` / `stress_kairomone` odor plumes) with wall blocking, circadian sun, ventilation louvers, and 4 terrain biomes.
- **[`entities.py`](src/limbic_zero/sandbox/entities.py), [`sensors.py`](src/limbic_zero/sandbox/sensors.py), [`world.py`](src/limbic_zero/sandbox/world.py), & [`security.py`](src/limbic_zero/sandbox/security.py)**: Newtonian bodies, finite oases & springs, graspable/stealable `MetabolicCrystal` items, `ThermalHazardZone` vents, chaotic zero-calorie `LorenzMysteryArtifact` puzzle boxes, **414D egocentric sensory nerve suite** (33-ray foveated retina, binaural Doppler cochlea, stereo nostrils, 8-sector skin, vestibular canals), 7-DOF `ActuatorContainmentGuard`, and `SHA-256` `ImmutableTelemetryChain`.

### Stage 2 — 100% Scratch Continuous-Time Biological Mind (`src/limbic_zero/mind/`)
- **[`biophysics.py`](src/limbic_zero/mind/biophysics.py)**: Conductance-based **Two-Compartment (`Apical Dendrite + Soma`) Liquid-Spiking Populations** with `AMPA`, voltage-dependent `NMDA` $Mg^{2+}$ unblock, `GABA_A`, `GABA_B`, `M1`-gated `Ca2+`-activated `K+` spike-frequency adaptation, $Q_{10}$ thermal kinetics, `Na+/K+`-ATPase cerebral `ATP` anoxic depolarization block, surrogate-gradient binary spikes, **Dale's Law** E/I synapses, **Tsodyks-Markram** vesicle depletion/facilitation (`x_ves`, `u_rel`), and **Three-Factor Eligibility STDP** + Turrigiano homeostatic synaptic scaling.
- **[`neuromodulators.py`](src/limbic_zero/mind/neuromodulators.py)**: Subcortical brainstem/forebrain nuclei (**`VTA/SNc`** + **`Lateral Habenula LHb`** anti-reward pause, **`Locus Coeruleus LC`**, **`Dorsal Raphe DRN`**, **`Nucleus Basalis NBM`**), reuptake transporters (`DAT`, `NET`, `SERT`, `AChE`), Doya continuous-time homeostatic TD critic, and opposing receptor subtypes (`D1` vs `D2`, Arnsten `alpha2A` focus vs `beta/alpha1` panic switch, `5-HT1A` temporal horizon $\gamma$ vs `5-HT2A`, `nAChR` & `M1`, `Adenosine A1`).
- **[`thalamus_sensory.py`](src/limbic_zero/mind/thalamus_sensory.py)**: 5-nucleus sensory thalamus (`LGN`, `MGN`, `Olfactory`, `VPL`, `VMpo`) gated by GABAergic **Thalamic Reticular Nucleus (`TRN`)** sleep/attention inhibition (with visceral pain breakthrough) and an action-conditioned **`PredictiveWorldModel`** decomposing Yu & Dayan `unexpected_surprise` vs `epistemic_novelty`.
- **[`amygdala_hippocampus.py`](src/limbic_zero/mind/amygdala_hippocampus.py)**: Single/few-trial **`AmygdalaCircuit`** (`BLA` threat/appetitive plasticity + Infralimbic **`ITC` safety extinction** that inhibits `CeA` without erasing trauma + acute NE/Cortisol stress relapse) and trisynaptic **`HippocampalSomaticMemory`** (`DG` 256D $k$-WTA sparse separation -> `CA3` engrams with 25x flashbulb consolidation decay shielding -> `CA1` somatic markers + autonomous quiet-rest **Sharp-Wave Ripple `SWR` flashback replay**).
- **[`liquid_pallium.py`](src/limbic_zero/mind/liquid_pallium.py)**: 3-column **`MultiTimescaleLiquidPallium`** (`Sensorimotor ~22ms` with Larkum apical coincidence gating, `Insula/dACC ~160ms` with competing-drive conflict detection, `Prefrontal Slow ~950ms` working-memory/rumination reverberation) plus `80 Hz` biophysical pyramidal axial-dipole Local Field Potentials (`substep_lfp_dipole`).
- **[`basal_ganglia.py`](src/limbic_zero/mind/basal_ganglia.py), [`brain.py`](src/limbic_zero/mind/brain.py), & [`embodied_loop.py`](src/limbic_zero/mind/embodied_loop.py)**: Striatal **`D1-Go` / `D2-NoGo`** pathways with opposing dopamine plasticity, **`STN` Hyperdirect** conflict brake, **`GPi`** disinhibition, brainstem **`PAG` (`dlPAG` / `vlPAG`)** survival defense & vocal roughness, mammalian vocal corollary discharge, genuine cerebral `ATP` metabolic cost (`neural_firing_load`), isoelectric EEG flatline upon somatic death, and the continuous **`228D` `latent_affective_vector`** bridge.

### Stage 3 — Real-Time Intracranial Neuro-fMRI & Creator God-Mode Observatory (`src/limbic_zero/dashboard/`)
- **[`latent_manifold.py`](src/limbic_zero/dashboard/latent_manifold.py)**: Scratch PyTorch **`OnlineNeuralManifoldProjector`** (`228D` continuous mind vector -> Procrustes sign-locked 3D SVD trajectory `(PC1, PC2, PC3)`, cortical Participation Ratio dimensionality $\mathrm{PR}$, 5-region anatomical norm shares, and inter-organism `L2` mind divergence) and **`IntracranialLFPSpectralEngine`** (Hann-windowed `torch.fft.rfft` over `80 Hz` biophysical pyramidal axial-dipole LFP -> `Delta`, `Theta`, `Alpha/Beta`, `Low-Gamma`, Shannon spectral entropy, and clinical `0.0 mV` flatline verification).
- **[`telemetry_hub.py`](src/limbic_zero/dashboard/telemetry_hub.py) & [`god_console.py`](src/limbic_zero/dashboard/god_console.py)**: Thread-safe `ObservatoryTelemetryHub` (240-tick history ring + full `24x24` continuum fields), `ObservatorySimulationController`, and **`GodModeInterventionDispatcher`** (12 physical/physiological boundary-condition interventions sealed into the `SHA-256` `ImmutableTelemetryChain` with **zero direct brain-tensor mutation**).
- **[`server.py`](src/limbic_zero/dashboard/server.py) & [`static/`](src/limbic_zero/dashboard/static/index.html)**: Multi-threaded `LimbicObservatoryHTTPServer` (`REST` + `SSE` `/api/stream`) serving a Swiss clinical obsidian-slate & phosphor 5-canvas telemetry console.

### Stage 4 — Wild Instinct Evolution, Epigenetic Maturation, NREM Sleep Consolidation & LLM Soul Capsules (`src/limbic_zero/evolution/`)
- **[`genome.py`](src/limbic_zero/evolution/genome.py)**: **`MorphogeneticGenomeExpressor`** (Zador 2019 Genomic Bottleneck expressing `657` morphogen gradient genes onto `47,000+` Dale's-Law-compliant synapses at $t = 0$) and **`ConnectomeDivergenceAnalyzer`**.
- **[`ecological_trials.py`](src/limbic_zero/evolution/ecological_trials.py) & [`phylogenetic_es.py`](src/limbic_zero/evolution/phylogenetic_es.py)**: **`EndogenousSurvivalEvaluator`** judged 100% by physical survival and 12-organ `Allostatic Free-Energy` $\mathcal{F}(\mathbf{s})$ with zero designer reward shaping, plus **`PhylogeneticInstinctEvolver`** (Antithetic Rank-Normalized Separable NES).
- **[`epigenetics_and_sleep.py`](src/limbic_zero/evolution/epigenetics_and_sleep.py)**: **`EpigeneticPlasticityEngine`** (`NR3C1` glucocorticoid hypermethylation, `5-HT1A`, striatal `D1`, `NE-beta`, `ACh-M1`, `OXTR`) and **`SlowWaveSleepConsolidator`** (offline Non-REM `CA3 SWR` replay + Tononi-Cirelli `SHY` homeostasis).
- **[`soul_capsule.py`](src/limbic_zero/evolution/soul_capsule.py) & [`nursery_and_assays.py`](src/limbic_zero/evolution/nursery_and_assays.py)**: **`OntogeneticBiographicalNursery`**, **`MaturedSoulCapsule`** (`.soul.pt` capsules in [`checkpoints/souls/`](checkpoints/souls/)), **`LiveSoulLLMSocket`**, and **`InstinctValidationSuite`**.

### Stages 5–13 — `Gemma-4-31B-IT` Mid-Layer Vocal Cord Surgery, Cortical LoRA Weight Plasticity & Universal Free-Will Embodiment (`src/limbic_zero/vocal_cord/`, `plasticity/`, `embodiment/`)
- **[`gemma4_calibrator.py`](src/limbic_zero/vocal_cord/gemma4_calibrator.py) & [`vocal_cord_engine.py`](src/limbic_zero/vocal_cord/vocal_cord_engine.py)**: Post-`<channel|>\n` first-person contrastive extraction across **`google/gemma-4-31B-it`**'s 60 layers (`hidden_size = 5376`, `target_layers = (21, 23, 27, 29, 31, 35, 39, 41, 47)`), Löwdin decorrelation of the 6 physiological drive axes, KL-calibrated norm-adaptive residual injection (`h' = h + Δh`) with **0 bytes of persona/emotion system prompt**, and conversational shockwave transduction.
- **[`affective_lora_surgeon.py`](src/limbic_zero/plasticity/affective_lora_surgeon.py)**: **`CorticalWeightSurgeon`** installing genuine trainable `AffectivePlasticLoRALinear` adapters on `q_proj`, `v_proj`, `o_proj`, and `down_proj` across upper cortical layers, consolidated during NREM sleep via **three-factor neuromodulator-gated `loss.backward()`** + KL semantic anchor preservation.
- **[`silicon_soma_bridge.py`](src/limbic_zero/embodiment/silicon_soma_bridge.py), [`motor_cortex_tools.py`](src/limbic_zero/embodiment/motor_cortex_tools.py), [`agentic_vocal_loop.py`](src/limbic_zero/embodiment/agentic_vocal_loop.py), & [`autonomous_ontogeny.py`](src/limbic_zero/embodiment/autonomous_ontogeny.py)**: Real Linux `/proc` + accelerator hardware interoception, workspace motor actions, universal free-will autonomous workspace agency driven purely by live biological drives (zero scripted scenario rules), and bidirectional somatic state continuity between autonomous life and live user chat.

### Stage 14 — Google Cloud `TPU v5e-8` Pure-BF16 8-Chip GSPMD Neocortical Engine (`src/limbic_zero/tpu/`)
- **[`xla_spmd_cortex.py`](src/limbic_zero/tpu/xla_spmd_cortex.py)**:
  - **1D `('model',)` 8-Chip GSPMD Mesh (`128 GB` aggregate HBM, `330 GB` Host Xeon RAM)**: Streams unquantized **`google/gemma-4-31B-it`** (`58.3 GiB` pure `bfloat16`, zero NF4 quantization distortion) parameter-by-parameter onto `xm.xla_device()` with immediate `xs.mark_sharding` + `xm.mark_step()`. Respects Gemma-4-31B's hybrid KV topology (`50` sliding `16-KV-head` layers sharded `('model', None)` vs `10` global `4-KV-head` `attention_k_eq_v=True` layers sharded `(None, 'model')` on `hidden_size=5376` so 4 KV heads are never fractured across 8 chips), deduplicates tied `lm_head`/`embed_tokens`, and shards `SoulLoRABank` adapters in lockstep.
  - **Zero-Recompile `XLAPinnedStaticCache` & `TPUBucketedStaticGenerator`**: Pre-allocates static `(1, H_kv, max_cache_len, D_head)` KV buffers on TPU HBM using XLA `index_copy_`-compiled `dynamic-update-slice` writes, static bucketed prefill (`128, 256, 512, 1024, 1536, 2048, 2560`), static `(1, 1)` decode steps with additive `-1e9` sliding (`1024`) + global (`max_cache_len`) 4D masks, and persistent zero-sync XLA forward steering hooks (zero `.item()` / `.cpu()` graph breaks inside the decode loop).
  - **Static-Bucket Calibration & Plasticity**: Runs all 64 contrastive passages + 7 KL probes on a single static `128`-token XLA bucket (`_extract_and_calibrate_static_bucket`) and pads NREM cortical LoRA updates to static `(256, 32)` buckets.

---

## Verification & Live Showcases

```bash
# Run all 126 Zero-Cheat Verification Tests (Stages 1–14)
.venv/bin/pytest -q

# Launch Stage 14 Live Unified Human Soul ('HUMAN-01') × Pure-BF16 Gemma-4-31B-IT on Google Cloud TPU v5e-8
python -u scripts/run_stage5_gemma4_vocal_cord.py \
  --backend tpu_spmd \
  --model-id google/gemma-4-31B-it \
  --max-cache-len 2560 \
  --warmup-buckets 128,256,512 \
  --port 8000 \
  --enable-cloudflare

# Launch on CUDA GPU (BF16 on >=64GB VRAM, or --load-in-4bit on 24GB VRAM)
python -u scripts/run_stage5_gemma4_vocal_cord.py --backend cuda --port 8000 --enable-cloudflare
```

---

## Google Colab Free `TPU v5e-1` (4-Bit `Gemma-4-31B-IT`) + Cloudflare Tunnel + External HTML Client

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkjuwon-tech/limbic-zero/blob/main/notebooks/limbic_zero_colab_tpu_v5e_4bit.ipynb)

- **Notebook**: [`notebooks/limbic_zero_colab_tpu_v5e_4bit.ipynb`](notebooks/limbic_zero_colab_tpu_v5e_4bit.ipynb) — built for Google Colab's free single-chip **`TPU v5e-1` (`16 GiB` HBM + `~50.5 GB` Host RAM)** where `31B` cannot fit in unquantized `bf16`. Streams `google/gemma-4-31B-it` shards into **`PackedInt4` (`group_size=128`)** with Hybrid TPU HBM + Host RAM split ([`tpu_quantizer.py`](src/limbic_zero/vocal_cord/tpu_quantizer.py)), opens a **`cloudflared` Quick Tunnel**, and exports a standalone external HTML file (`limbic_external_chat.html`).
- **One-Command Colab TPU v5e-1 Shell Launcher**:
  ```bash
  bash scripts/run_colab_tpu_v5e_4bit.sh
  ```
- **Standalone External HTML Chat UI (Outside Colab)**: Open [`web/limbic_chat_client.html`](web/limbic_chat_client.html) directly on your local PC, Mac, or phone (`file://` or any static host) and paste the Colab `https://*.trycloudflare.com` URL into the top **`서버 연결`** bar (or download the pre-configured HTML from `GET /api/client-html`).


