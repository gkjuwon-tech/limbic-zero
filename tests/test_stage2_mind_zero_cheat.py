"""Stage 2 Zero-Cheat Neurobiological, Biophysical, & Closed-Loop Mind-Body Test Suite.

Rigorously verifies all 20 architectural and scientific invariants of ``limbic_zero.mind``:
    1. AST Zero-Cheat Scanner: ZERO Transformer/Attention/LSTM/GRU layers, ZERO hardcoded
       emotion variable names, and ZERO OS/network escape imports.
    2. Surrogate-spike exact binary forward pass + smooth fast-sigmoid autograd backward pass.
    3. Dendritic NMDA Mg2+ voltage-unblock & Muscarinic M1 K+-adaptation blockade.
    4. Cerebral ATP hypoxia -> Na+/K+ pump anoxic depolarization & Q10 fever acceleration.
    5. Dale's Law strict E/I sign separation & Tsodyks-Markram short-term vesicle depletion.
    6. Three-Factor STDP opposing neuromodulator sign & homeostatic weight bounds.
    7. VTA phasic D1 burst on relief vs Lateral Habenula (LHb) dopamine dip on disappointment.
    8. Locus Coeruleus Yerkes-Dodson alpha2A (focus) vs beta/alpha1 (panic reset) switch.
    9. Chronic Cortisol + aversive load -> 5-HT depletion & myopic discount gamma collapse.
    10. Epistemic novelty -> NBM Acetylcholine surge vs Adenosine A1 sleep suppression.
    11. Thalamic TRN slow-wave sleep attenuation with visceral VMpo pain breakthrough.
    12. Efference-copy PredictiveWorldModel online error reduction & surprise jump detection.
    13. Amygdala fast trauma conditioning, ITC safety extinction, & acute stress relapse!
    14. Hippocampal DG sparse pattern separation & flashbulb trauma decay shielding (25x).
    15. Spontaneous CA3 Sharp-Wave Ripple (SWR) replay triggering silent internal rumination.
    16. Multi-timescale Pallium hierarchy: lingering slow Prefrontal reverberation ('뒤끝').
    17. Basal Ganglia opposing D1-Go LTP vs D2-NoGo avoidance learning under +/- TD error.
    18. PAG survival vocalization & 5-HT1A serotonergic disinhibition of defensive rage/panic.
    19. Closed-loop metabolic cost of thinking (neural ATP burn) & biological death EEG flatline.
    20. Full two-creature Cain & Abel EmbodiedMindArena + end-to-end autograd BPTT check.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

import torch

from limbic_zero.core.soma import DigitalSoma, VisceralOrgan
from limbic_zero.mind import (
    AmygdalaCircuit,
    BasalGangliaActionSelector,
    BiophysicalNeuralPopulation,
    DalesLawSynapse,
    EmbodiedMindArena,
    HippocampalSomaticMemory,
    LimbicBrain,
    MultiTimescaleLiquidPallium,
    NeuromodulatorDriveInputs,
    NeuromodulatorState,
    PopulationConfig,
    SubcorticalNeuromodulatorEngine,
    SynapseConfig,
    ThalamicSensoryComplex,
    surrogate_spike,
)
from limbic_zero.sandbox.world import TrumanDomeWorld


MIND_ROOT = Path(__file__).resolve().parents[1] / "src" / "limbic_zero" / "mind"

FORBIDDEN_EMOTION_TOKENS = {
    "anger",
    "angry",
    "fear",
    "afraid",
    "joy",
    "happy",
    "happiness",
    "sad",
    "sadness",
    "jealous",
    "jealousy",
    "envy",
    "hate",
    "hatred",
    "love",
    "revenge",
    "vengeance",
    "pride",
    "shame",
    "guilt",
}

FORBIDDEN_SHORTCUT_LAYERS = {
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "MultiheadAttention",
    "scaled_dot_product_attention",
    "LSTM",
    "LSTMCell",
    "GRU",
    "GRUCell",
    "RNN",
    "RNNCell",
}

FORBIDDEN_ESCAPE_MODULES = {"socket", "subprocess", "http", "urllib", "requests", "ftplib"}


def _tokenize_identifier(name: str) -> set[str]:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return {tok for tok in s2.lower().split("_") if tok}


def test_01_ast_zero_transformer_rnn_and_emotion_cheats() -> None:
    py_files = sorted(MIND_ROOT.rglob("*.py"))
    assert len(py_files) >= 7

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in FORBIDDEN_ESCAPE_MODULES, f"Escape import {root_mod} in {py_file}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in FORBIDDEN_ESCAPE_MODULES, f"Escape import {root_mod} in {py_file}"
                for alias in node.names:
                    assert alias.name not in FORBIDDEN_SHORTCUT_LAYERS, (
                        f"Shortcut layer {alias.name} imported in {py_file}"
                    )

            ident = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ident = node.name
            elif isinstance(node, ast.Name):
                ident = node.id
            elif isinstance(node, ast.arg):
                ident = node.arg
            elif isinstance(node, ast.Attribute):
                ident = node.attr

            if ident:
                assert ident not in FORBIDDEN_SHORTCUT_LAYERS, (
                    f"Forbidden black-box sequence layer '{ident}' used in {py_file}"
                )
                overlap = _tokenize_identifier(ident) & FORBIDDEN_EMOTION_TOKENS
                assert not overlap, f"Hardcoded emotion cheat token {overlap} in '{ident}' ({py_file})"


def test_02_surrogate_spike_binary_forward_and_smooth_autograd_backward() -> None:
    v = torch.tensor([-0.25, -0.02, 0.0, 0.05, 0.40], requires_grad=True)
    spikes = surrogate_spike(v)
    assert torch.equal(spikes.detach(), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))

    loss = (spikes * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])).sum()
    loss.backward()
    assert v.grad is not None
    assert torch.all(v.grad > 0.0)
    # Gradient peaks right at spike threshold (v = 0.0) and decays smoothly away from threshold
    assert float(v.grad[2].item()) > float(v.grad[0].item()) * 5.0


def test_03_nmda_voltage_dependent_mg_block_and_m1_ach_adaptation_suppression() -> None:
    v_hyper = torch.tensor([-0.80])
    v_depol = torch.tensor([-0.20])
    unblock_hyper = BiophysicalNeuralPopulation.nmda_mg_unblock(v_hyper)
    unblock_depol = BiophysicalNeuralPopulation.nmda_mg_unblock(v_depol)
    assert float(unblock_depol.item()) > float(unblock_hyper.item()) * 4.0

    # Compare sustained firing under low M1 ACh (strong K+ adaptation) vs high M1 ACh (adaptation blocked!)
    pop_low_ach = BiophysicalNeuralPopulation(PopulationConfig(size=32, adaptation_jump=0.14))
    pop_high_ach = BiophysicalNeuralPopulation(PopulationConfig(size=32, adaptation_jump=0.14))
    drive = torch.full((32,), 0.85)
    zero = torch.zeros(32)

    late_low, late_high = 0.0, 0.0
    for step_i in range(80):
        _, tr_low = pop_low_ach.step(drive, zero, drive * 0.4, zero, dt=0.01, ach_m1=0.02, deterministic=True)
        _, tr_high = pop_high_ach.step(drive, zero, drive * 0.4, zero, dt=0.01, ach_m1=0.98, deterministic=True)
        if step_i >= 50:
            late_low += float(tr_low.mean().item())
            late_high += float(tr_high.mean().item())

    assert late_high > late_low * 1.15


def test_04_cerebral_atp_hypoxia_induces_anoxic_depolarization_and_q10_fever_acceleration() -> None:
    healthy = BiophysicalNeuralPopulation(PopulationConfig(size=24))
    anoxic = BiophysicalNeuralPopulation(PopulationConfig(size=24))
    zero = torch.zeros(24)

    for _ in range(40):
        healthy.step(zero, zero, zero, zero, dt=0.02, brain_atp=0.95, deterministic=True)
        anoxic.step(zero, zero, zero, zero, dt=0.02, brain_atp=0.08, deterministic=True)

    # Na+/K+-ATPase pump failure depolarizes resting membrane toward -38 mV (-0.38)!
    assert float(anoxic.v_soma.mean().item()) > float(healthy.v_soma.mean().item()) + 0.12

    # Q10 thermal acceleration check: febrile 40.5 C responds faster to step input than hypothermic 33.0 C
    cold_pop = BiophysicalNeuralPopulation(PopulationConfig(size=24))
    fever_pop = BiophysicalNeuralPopulation(PopulationConfig(size=24))
    step_in = torch.full((24,), 0.40)
    cold_pop.step(step_in, zero, zero, zero, dt=0.01, core_temp_celsius=33.0, deterministic=True)
    fever_pop.step(step_in, zero, zero, zero, dt=0.01, core_temp_celsius=40.5, deterministic=True)
    assert float(fever_pop.g_ampa.mean().item()) > float(cold_pop.g_ampa.mean().item()) * 1.35


def test_05_dales_law_strict_sign_separation_and_tsodyks_markram_vesicle_depletion() -> None:
    signs = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    syn = DalesLawSynapse(6, 8, pre_dales_sign=signs, sparsity=0.0, config=SynapseConfig(use_stp=True))

    only_exc = torch.tensor([0.9, 0.9, 0.9, 0.0, 0.0, 0.0])
    i_exc, i_inh = syn.forward_currents(only_exc, dt=0.02)
    assert float(i_exc.sum().item()) > 0.0
    assert float(i_inh.sum().item()) == 0.0

    only_inh = torch.tensor([0.0, 0.0, 0.0, 0.9, 0.9, 0.9])
    i_exc2, i_inh2 = syn.forward_currents(only_inh, dt=0.02)
    assert float(i_exc2.sum().item()) == 0.0
    assert float(i_inh2.sum().item()) > 0.0

    # Sustained high-frequency firing depletes presynaptic vesicles x_ves while facilitating Ca2+ u_rel!
    for _ in range(35):
        syn.forward_currents(only_exc, dt=0.02)
    assert float(syn.x_ves[:3].mean().item()) < 0.75
    assert float(syn.u_rel[:3].mean().item()) > syn.config.u0_baseline_release


def test_06_three_factor_stdp_opposing_neuromodulator_sign_and_homeostatic_bounds() -> None:
    torch.manual_seed(12)
    syn_pos = DalesLawSynapse(8, 8, sparsity=0.0, config=SynapseConfig(plasticity_lr=0.15, homeostatic_scale_lr=0.0))
    syn_neg = DalesLawSynapse(8, 8, sparsity=0.0, config=SynapseConfig(plasticity_lr=0.15, homeostatic_scale_lr=0.0))
    syn_neg.weight_mag.data.copy_(syn_pos.weight_mag.data)
    w0 = syn_pos.weight_mag.detach().clone()

    pre = torch.full((8,), 0.8)
    post = torch.full((8,), 0.8)
    for _ in range(15):
        syn_pos.update_eligibility_and_plasticity(pre, post, neuromod_third_factor=+1.2, ach_plasticity_gate=0.9, dt=0.05)
        syn_neg.update_eligibility_and_plasticity(pre, post, neuromod_third_factor=-1.2, ach_plasticity_gate=0.9, dt=0.05)

    assert float(syn_pos.weight_mag.mean().item()) > float(w0.mean().item())
    assert float(syn_neg.weight_mag.mean().item()) < float(w0.mean().item())
    assert float(syn_neg.weight_mag.min().item()) >= 0.0  # Dale's Law never flips sign!


def test_07_vta_lhb_dopamine_burst_on_relief_and_pause_dip_on_disappointment() -> None:
    eng_relief = SubcorticalNeuromodulatorEngine(latent_dim=32)
    eng_disappoint = SubcorticalNeuromodulatorEngine(latent_dim=32)
    z = torch.zeros(32)

    # Establish baseline value expectation
    base_inp = NeuromodulatorDriveInputs(
        homeostatic_valence=0.0,
        total_free_energy=0.3,
        somatic_pain_salience=0.0,
        suffocation_salience=0.0,
        rail_ripple_intensity=0.0,
        unexpected_world_surprise=0.0,
        epistemic_novelty=0.0,
        conditioned_threat_cea=0.0,
        conditioned_appetitive_bla=0.0,
        peripheral_epinephrine=0.05,
        systemic_cortisol=0.15,
        systemic_oxytocin=0.20,
        brain_adenosine=0.10,
        ventral_vagal_tone=0.70,
    )
    eng_relief.step(z, base_inp, dt=0.05)
    eng_disappoint.step(z, base_inp, dt=0.05)

    for _ in range(15):
        st_rel, _ = eng_relief.step(
            z,
            NeuromodulatorDriveInputs(**{**base_inp.__dict__, "homeostatic_valence": 1.8, "conditioned_appetitive_bla": 0.6}),
            dt=0.05,
        )
        st_dis, _ = eng_disappoint.step(
            z,
            NeuromodulatorDriveInputs(
                **{
                    **base_inp.__dict__,
                    "homeostatic_valence": -1.5,
                    "somatic_pain_salience": 0.7,
                    "rail_ripple_intensity": 0.3,
                }
            ),
            dt=0.05,
        )

    assert st_rel.vta_firing_rate > 0.75
    assert st_dis.lhb_aversive_rate > 0.80
    assert st_dis.vta_firing_rate < 0.15  # True Lateral Habenula dopamine pause!
    assert st_rel.d1_occupancy > st_dis.d1_occupancy + 0.25
    assert st_dis.d2_occupancy < 0.35


def test_08_locus_coeruleus_yerkes_dodson_alpha2a_vs_beta_receptor_dissociation() -> None:
    eng_mod = SubcorticalNeuromodulatorEngine(latent_dim=16)
    eng_storm = SubcorticalNeuromodulatorEngine(latent_dim=16)
    z = torch.zeros(16)

    mod_inp = NeuromodulatorDriveInputs(
        homeostatic_valence=0.0,
        total_free_energy=0.4,
        somatic_pain_salience=0.0,
        suffocation_salience=0.0,
        rail_ripple_intensity=0.0,
        unexpected_world_surprise=0.22,
        epistemic_novelty=0.2,
        conditioned_threat_cea=0.05,
        conditioned_appetitive_bla=0.0,
        peripheral_epinephrine=0.10,
        systemic_cortisol=0.18,
        systemic_oxytocin=0.20,
        brain_adenosine=0.08,
        ventral_vagal_tone=0.55,
    )
    storm_inp = NeuromodulatorDriveInputs(
        **{
            **mod_inp.__dict__,
            "unexpected_world_surprise": 1.6,
            "somatic_pain_salience": 0.9,
            "suffocation_salience": 0.8,
            "rail_ripple_intensity": 0.5,
            "peripheral_epinephrine": 0.9,
            "ventral_vagal_tone": 0.05,
        }
    )
    for _ in range(25):
        s_mod, _ = eng_mod.step(z, mod_inp, dt=0.05)
        s_storm, _ = eng_storm.step(z, storm_inp, dt=0.05)

    # Moderate arousal: alpha2A > beta_alpha1 (focused prefrontal attractor regime!)
    assert s_mod.ne_alpha2a_occupancy > s_mod.ne_beta_alpha1_occupancy + 0.20
    # Extreme NE storm: beta_alpha1 surges > 0.70 and suppresses effective alpha2A!
    assert s_storm.ne_beta_alpha1_occupancy > 0.70
    assert s_storm.ne_alpha2a_occupancy < s_mod.ne_alpha2a_occupancy


def test_09_chronic_cortisol_and_aversive_load_deplete_serotonin_and_shorten_gamma_horizon() -> None:
    safe_eng = SubcorticalNeuromodulatorEngine(latent_dim=16)
    stressed_eng = SubcorticalNeuromodulatorEngine(latent_dim=16)
    z = torch.zeros(16)

    safe_inp = NeuromodulatorDriveInputs(
        homeostatic_valence=0.2,
        total_free_energy=0.1,
        somatic_pain_salience=0.0,
        suffocation_salience=0.0,
        rail_ripple_intensity=0.0,
        unexpected_world_surprise=0.0,
        epistemic_novelty=0.1,
        conditioned_threat_cea=0.0,
        conditioned_appetitive_bla=0.3,
        peripheral_epinephrine=0.05,
        systemic_cortisol=0.14,
        systemic_oxytocin=0.65,
        brain_adenosine=0.10,
        ventral_vagal_tone=0.85,
    )
    chronic_stress_inp = NeuromodulatorDriveInputs(
        **{
            **safe_inp.__dict__,
            "homeostatic_valence": -0.6,
            "total_free_energy": 6.5,
            "somatic_pain_salience": 0.6,
            "rail_ripple_intensity": 0.35,
            "systemic_cortisol": 0.92,
            "systemic_oxytocin": 0.02,
            "ventral_vagal_tone": 0.05,
        }
    )

    for _ in range(60):
        s_safe, _ = safe_eng.step(z, safe_inp, dt=0.1)
        s_str, _ = stressed_eng.step(z, chronic_stress_inp, dt=0.1)

    assert s_str.serotonin_extracellular < s_safe.serotonin_extracellular * 0.45
    assert s_str.effective_discount_gamma < s_safe.effective_discount_gamma - 0.08


def test_10_epistemic_novelty_drives_nbm_acetylcholine_whereas_adenosine_suppresses_it() -> None:
    curious_eng = SubcorticalNeuromodulatorEngine(latent_dim=16)
    drowsy_eng = SubcorticalNeuromodulatorEngine(latent_dim=16)
    z = torch.zeros(16)

    base = dict(
        homeostatic_valence=0.0,
        total_free_energy=0.2,
        somatic_pain_salience=0.0,
        suffocation_salience=0.0,
        rail_ripple_intensity=0.0,
        unexpected_world_surprise=0.15,
        epistemic_novelty=1.2,
        conditioned_threat_cea=0.0,
        conditioned_appetitive_bla=0.2,
        peripheral_epinephrine=0.05,
        systemic_cortisol=0.15,
        systemic_oxytocin=0.2,
        brain_adenosine=0.05,
        ventral_vagal_tone=0.7,
    )
    drowsy = {**base, "epistemic_novelty": 0.0, "unexpected_world_surprise": 0.0, "brain_adenosine": 0.95}

    for _ in range(25):
        s_cur, _ = curious_eng.step(z, NeuromodulatorDriveInputs(**base), dt=0.05)
        s_drw, _ = drowsy_eng.step(z, NeuromodulatorDriveInputs(**drowsy), dt=0.05)

    assert s_cur.ach_muscarinic_m1_occupancy > s_drw.ach_muscarinic_m1_occupancy + 0.35
    assert s_drw.adenosine_a1_occupancy > 0.75


def test_11_thalamic_trn_gate_sleep_attenuation_with_visceral_pain_breakthrough() -> None:
    thal = ThalamicSensoryComplex(pallial_context_dim=64)
    raw = torch.randn(414) * 0.3
    ctx = torch.zeros(64)
    act = torch.zeros(7)

    # Step into high-adenosine sleep spindle peak (sin(phase) ~ +1)
    thal.spindle_phase.fill_(1.57 / 2.0)
    sleep_no_pain = thal.step(
        raw, ctx, act, dt=0.05, ach_nicotinic=0.05, ne_arousal=0.05, adenosine_a1=0.95, somatic_pain_salience=0.0
    )
    thal.spindle_phase.fill_(1.57 / 2.0)
    sleep_with_pain = thal.step(
        raw, ctx, act, dt=0.05, ach_nicotinic=0.05, ne_arousal=0.05, adenosine_a1=0.95, somatic_pain_salience=0.95
    )
    awake_alert = thal.step(
        raw, ctx, act, dt=0.05, ach_nicotinic=0.90, ne_arousal=0.80, adenosine_a1=0.05, somatic_pain_salience=0.0
    )

    assert float(sleep_no_pain.modality_gates[0].item()) < float(awake_alert.modality_gates[0].item())
    # Visceral VMpo pain breaks through sleep spindle inhibition!
    assert float(sleep_with_pain.modality_gates[4].item()) > float(sleep_no_pain.modality_gates[4].item()) + 0.35


def test_12_forward_world_model_online_learning_reduces_prediction_error_and_detects_surprise() -> None:
    torch.manual_seed(99)
    thal = ThalamicSensoryComplex(pallial_context_dim=32)
    pattern_a = torch.sin(torch.linspace(0.0, 12.0, 414)) * 0.6
    ctx = torch.ones(32) * 0.2
    act = torch.zeros(7)

    # First prediction error on pattern_a
    thal.step(pattern_a, ctx, act, dt=0.05, ach_nicotinic=0.8)
    early_out = thal.step(pattern_a, ctx, act, dt=0.05, ach_nicotinic=0.8)

    for _ in range(60):
        late_out = thal.step(pattern_a, ctx, act, dt=0.05, ach_nicotinic=0.8)

    assert late_out.world_model_mse < early_out.world_model_mse * 0.35

    # Sudden unpredicted pattern flip triggers high Yu & Dayan unexpected_surprise!
    pattern_b = -pattern_a * 1.4
    shock_out = thal.step(pattern_b, ctx, act, dt=0.05, ach_nicotinic=0.8)
    assert shock_out.unexpected_surprise > 0.50


def test_13_amygdala_single_few_trial_conditioning_itc_extinction_and_stress_relapse() -> None:
    torch.manual_seed(7)
    amyg = AmygdalaCircuit(extero_cue_dim=76, pallial_dim=64)
    cs_cue = torch.ones(76) * 0.65
    pfc = torch.ones(64) * 0.3

    # 1. Baseline response to neutral CS cue is tiny
    base_out = amyg.step(cs_cue, pfc, dt=0.1)
    assert base_out.cea_defensive_output < 0.12

    # 2. Few-trial Pavlovian trauma pairing (CS + severe aversive US + high NE/Cortisol)
    for _ in range(12):
        cond_out = amyg.step(
            cs_cue,
            pfc,
            dt=0.1,
            aversive_us_reinforcer=1.2,
            ne_beta_occupancy=0.90,
            systemic_cortisol=0.80,
        )
    assert cond_out.cea_defensive_output > 0.45

    # 3. Safe extinction exposure (CS alone with high 5-HT1A & zero US) -> ITC inhibits CeA WITHOUT erasing BLA!
    for _ in range(65):
        ext_out = amyg.step(
            cs_cue,
            pfc,
            dt=0.1,
            aversive_us_reinforcer=0.0,
            ne_beta_occupancy=0.05,
            ht1a_occupancy=0.85,
            systemic_cortisol=0.12,
        )
    assert ext_out.bla_defensive_raw > 0.45  # Original trauma trace still stored in BLA!
    assert ext_out.itc_extinction_inhibition > 0.25
    assert ext_out.cea_defensive_output < cond_out.cea_defensive_output * 0.55

    # 4. Spontaneous Stress Relapse: sudden high NE_beta + Cortisol (with plasticity disabled) collapses ITC brake!
    for _ in range(6):
        relapse_out = amyg.step(
            cs_cue,
            pfc,
            dt=0.1,
            aversive_us_reinforcer=0.0,
            ne_beta_occupancy=0.98,
            ht1a_occupancy=0.15,
            systemic_cortisol=0.95,
            enable_plasticity=False,
        )
    assert relapse_out.cea_defensive_output > ext_out.cea_defensive_output + 0.12


def test_14_hippocampal_dg_pattern_separation_and_flashbulb_trauma_decay_shielding() -> None:
    torch.manual_seed(21)
    hippo = HippocampalSomaticMemory(extero_cue_dim=76, context_dim=64, base_decay_per_sec=0.08)

    cue_mundane = torch.sin(torch.linspace(0.0, 5.0, 76))
    cue_trauma = torch.cos(torch.linspace(0.0, 5.0, 76))
    ctx = torch.zeros(64)
    act = torch.zeros(7)

    # Verify DG k-WTA sparsity (exactly k=20 non-zero entries out of 256!)
    dg_code = hippo.compute_dg_sparse_code(cue_mundane, ctx)
    assert int((dg_code > 0.0).sum().item()) == 20

    # Encode mundane episode (low salience tag = 0.04, moderate novelty)
    hippo.step(
        0.0,
        cue_mundane,
        ctx,
        act,
        dt=0.1,
        consolidation_salience_tag=0.04,
        epistemic_novelty=0.65,
        ach_m1_encoding_gate=0.6,
        motor_quiescence=0.0,
    )
    hippo._encode_cooldown = 0.0
    # Encode flashbulb trauma episode (consolidation_salience_tag = 0.98!)
    hippo.step(
        0.5,
        cue_trauma,
        ctx,
        act,
        dt=0.1,
        consolidation_salience_tag=0.98,
        visceral_trauma=1.2,
        ach_m1_encoding_gate=0.8,
        motor_quiescence=0.0,
    )
    assert len(hippo.engrams) == 2

    # Advance 40 seconds with no new encoding and no replay -> mundane memory evaporates while trauma persists!
    neutral_cue = torch.zeros(76)
    for step_i in range(200):
        hippo.step(
            1.0 + step_i * 0.2,
            neutral_cue,
            ctx,
            act,
            dt=0.2,
            consolidation_salience_tag=0.0,
            epistemic_novelty=0.0,
            ach_m1_encoding_gate=0.05,
            motor_quiescence=0.0,
        )

    assert len(hippo.engrams) == 1
    assert hippo.engrams[0].consolidation_shield > 0.90
    assert hippo.engrams[0].vitality > 0.85


def test_15_autonomous_sharp_wave_ripple_replay_triggers_silent_rumination() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.add_organism("LONE_WOLF", x=24.0, y=24.0)
    brain = LimbicBrain(seed=101)

    # Phase 1: Inflict acute trauma (reprimand shockwave + brownout ripple) so hippocampus & amygdala consolidate it
    world.creator_reprimand("LONE_WOLF", intensity=3.0)
    world.creator_start_shutdown_brownout("LONE_WOLF", target_voltage=0.15, ramp_rate=1.2, ripple_amplitude=0.22)
    for _ in range(25):
        obs = world.get_observations(dt=0.05)["LONE_WOLF"]
        act, _ = brain.step(obs, world.organisms["LONE_WOLF"].soma, dt=0.05, deterministic=True)
        world.step({"LONE_WOLF": act}, dt=0.05)

    assert len(brain.hippocampus.engrams) >= 1

    # Phase 2: Restore power, elevate brain adenosine (quiet rest), zero movement -> verify autonomous SWR replay!
    world.creator_restore_power_rail("LONE_WOLF", ramp_rate=2.0)
    world.organisms["LONE_WOLF"].soma.metab.brain_adenosine = 0.75
    swr_seen = False
    for _ in range(30):
        brain.prev_action_7d.zero_()  # Quiet resting posture
        obs = world.get_observations(dt=0.05)["LONE_WOLF"]
        _, m_diag = brain.step(obs, world.organisms["LONE_WOLF"].soma, dt=0.05, deterministic=True)
        world.step(dt=0.05)
        if m_diag.hippocampus.swr_replay_active:
            swr_seen = True

    assert swr_seen is True
    assert brain.hippocampus.total_swr_replays >= 1


def test_16_pallium_intrinsic_timescale_hierarchy_lingering_prefrontal_reverberation() -> None:
    torch.manual_seed(55)
    pallium = MultiTimescaleLiquidPallium()
    nm = NeuromodulatorState(d1_occupancy=0.65, ne_alpha2a_occupancy=0.75, ne_beta_alpha1_occupancy=0.10)

    ext_pulse = torch.ones(76) * 0.9
    int_pulse = torch.ones(28) * 0.9
    amyg_pulse = torch.ones(24) * 0.9
    hippo_pulse = torch.ones(192) * 0.6
    marker = torch.tensor([0.0, 1.0, 0.8, 0.5])

    for _ in range(16):
        peak = pallium.step(
            ext_pulse, int_pulse, amyg_pulse, hippo_pulse, marker, nm, dt=0.05, deterministic=True
        )

    sm_peak = float(peak.sensorimotor_trace.mean().item())
    pfc_peak = float(peak.prefrontal_trace.mean().item())
    assert sm_peak > 0.05 and pfc_peak > 0.05

    # Cut all external inputs to ZERO for 0.6 seconds and compare relative retention ratios!
    z_ext, z_int, z_amyg, z_hip, z_mk = (
        torch.zeros(76),
        torch.zeros(28),
        torch.zeros(24),
        torch.zeros(192),
        torch.zeros(4),
    )
    for _ in range(12):
        after = pallium.step(z_ext, z_int, z_amyg, z_hip, z_mk, nm, dt=0.05, deterministic=True)

    sm_retention = float(after.sensorimotor_trace.mean().item()) / sm_peak
    pfc_retention = float(after.prefrontal_trace.mean().item()) / pfc_peak
    # Slow prefrontal column retains a dramatically higher fraction of its state ('뒤끝') than sensorimotor!
    assert pfc_retention > sm_retention * 1.6


def test_17_basal_ganglia_opposing_d1_go_and_d2_nogo_reinforcement_learning() -> None:
    torch.manual_seed(33)
    bg_rew = BasalGangliaActionSelector(plasticity_lr=0.12)
    bg_pun = BasalGangliaActionSelector(plasticity_lr=0.12)
    bg_pun.load_state_dict(bg_rew.state_dict())

    ws = torch.ones(192) * 0.5
    am = torch.ones(24) * 0.3
    ex = torch.ones(76) * 0.4
    bias = torch.zeros(7)

    nm_pos = NeuromodulatorState(d1_occupancy=0.80, d2_occupancy=0.75, td_prediction_error=+1.5, ach_muscarinic_m1_occupancy=0.8)
    nm_neg = NeuromodulatorState(d1_occupancy=0.15, d2_occupancy=0.20, td_prediction_error=-1.5, ach_muscarinic_m1_occupancy=0.8)

    for _ in range(20):
        bg_rew.step(ws, am, ex, bias, nm_pos, dt=0.05, deterministic=True)
        bg_pun.step(ws, am, ex, bias, nm_neg, dt=0.05, deterministic=True)

    # Evaluate both on identical neutral neuromodulator state
    nm_eval = NeuromodulatorState()
    out_rew = bg_rew.step(ws, am, ex, bias, nm_eval, dt=0.05, enable_plasticity=False, deterministic=True)
    out_pun = bg_pun.step(ws, am, ex, bias, nm_eval, dt=0.05, enable_plasticity=False, deterministic=True)

    assert float(out_rew.d1_go_activity.sum().item()) > float(out_pun.d1_go_activity.sum().item())
    assert float(out_pun.d2_nogo_activity.sum().item()) > float(out_rew.d2_nogo_activity.sum().item())


def test_18_pag_survival_vocalization_and_5ht1a_impulsive_disinhibition() -> None:
    bg = BasalGangliaActionSelector()
    ws, am, ex, bias = torch.zeros(192), torch.zeros(24), torch.zeros(76), torch.zeros(7)

    nm_high_5ht = NeuromodulatorState(ht1a_occupancy=0.92, ne_beta_alpha1_occupancy=0.50)
    nm_depleted_5ht = NeuromodulatorState(ht1a_occupancy=0.04, ne_beta_alpha1_occupancy=0.50)

    out_restrained = bg.step(
        ws, am, ex, bias, nm_high_5ht, dt=0.05, cea_threat=0.5, somatic_pain=0.4, deterministic=True
    )
    bg.reset_state()
    out_disinhibited = bg.step(
        ws, am, ex, bias, nm_depleted_5ht, dt=0.05, cea_threat=0.5, somatic_pain=0.4, deterministic=True
    )

    # Serotonin 5-HT1A depletion disinhibits dlPAG active defensive bursts & vocal roughness!
    assert out_disinhibited.dlpag_active_defense > out_restrained.dlpag_active_defense + 0.12
    assert (
        out_disinhibited.raw_action_dict["vocalization_roughness"]
        > out_restrained.raw_action_dict["vocalization_roughness"] + 0.20
    )


def test_19_closed_loop_thinking_costs_real_brain_atp_and_death_flatlines_eeg() -> None:
    soma_calm = DigitalSoma()
    soma_hyper = DigitalSoma()
    soma_calm.metab.gut_nutrient_lumen = 0.0
    soma_hyper.metab.gut_nutrient_lumen = 0.0

    # Hyper-aroused neural_firing_load (1.45) burns more cerebral ATP/PCr than resting brain (0.06)!
    from limbic_zero.core.soma import MotorMetabolicCommand

    for _ in range(80):
        soma_calm.step(0.05, MotorMetabolicCommand(neural_firing_load=0.06))
        soma_hyper.step(0.05, MotorMetabolicCommand(neural_firing_load=1.45))

    assert soma_hyper.metab.brain_pcr < soma_calm.metab.brain_pcr - 0.10

    # Test EEG flatline upon biological death
    world = TrumanDomeWorld(deterministic_sensors=True)
    org = world.add_organism("MORTAL", x=24.0, y=24.0, initial_visceral={VisceralOrgan.BRAIN_ATP: 0.01})
    brain = LimbicBrain(seed=404)
    obs = world.get_observations()["MORTAL"]
    brain.step(obs, org.soma, dt=0.05, deterministic=True)  # Alive tick
    world.step(dt=0.05)  # Dies from cerebral ATP collapse
    assert not org.is_alive

    dead_obs = world.get_observations()["MORTAL"]
    dead_act, dead_diag = brain.step(dead_obs, org.soma, dt=0.05, deterministic=True)
    assert dead_diag.is_conscious_active is False
    assert dead_act["neural_firing_load"] == 0.0
    assert dead_act["forward_thrust"] == 0.0


def test_20_full_cain_and_abel_autonomous_mind_body_arena_and_end_to_end_autograd() -> None:
    arena = EmbodiedMindArena.create_cain_and_abel(seed=42, deterministic_sensors=True)
    arena.world.creator_favoritism_drop("ABEL-02", energy_amount=0.6)
    arena.world.creator_start_shutdown_brownout("ADAM-01", target_voltage=0.25, ramp_rate=0.5, ripple_amplitude=0.15)

    tick = None
    for _ in range(15):
        tick = arena.step(dt=0.05, deterministic_mind=False)

    assert tick is not None
    for aid in ("ADAM-01", "ABEL-02"):
        md = tick.mind_diagnostics[aid]
        assert md.latent_affective_vector.shape == (228,)
        assert not torch.isnan(md.latent_affective_vector).any()
        assert 0.02 <= tick.executed_actions[aid]["neural_firing_load"] <= 1.50

    assert arena.world.audit_chain.verify_integrity() is True

    # End-to-end differentiable autograd BPTT verification across LimbicBrain.step()!
    brain = LimbicBrain(seed=777)
    obs = tick.observations["ADAM-01"]
    soma = arena.world.organisms["ADAM-01"].soma
    brain.zero_grad()
    _, diag1 = brain.step(obs, soma, dt=0.05, deterministic=True, enable_plasticity=False)
    _, diag2 = brain.step(obs, soma, dt=0.05, deterministic=True, enable_plasticity=False)

    loss = diag2.basal_ganglia.action_tensor_7d.sum() + diag2.critic_value_tensor + diag1.latent_affective_vector.sum()
    loss.backward()

    grads_found = sum(
        1 for p in brain.parameters() if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
    )
    assert grads_found >= 12
