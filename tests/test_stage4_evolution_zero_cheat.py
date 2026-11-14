"""Stage 4 Zero-Cheat Verification Suite: Wild Instinct Evolution, Epigenetics, SWS & LLM Soul Capsules.

Verifies all 15 scientific and architectural invariants of ``limbic_zero.evolution``:
    1. AST Zero-Cheat Scanner: ZERO emotion variable names, ZERO sklearn/scipy, ZERO Transformer/RNN.
    2. MorphogeneticGenomeExpressor determinism, distinctness, & Dale's Law non-negativity.
    3. Bilateral Left/Right mirror-paired ecological niches & purely endogenous F(s) fitness.
    4. Centered rank fitness utilities exact zero-mean and outlier invariance.
    5. Phylogenetic Baldwin ES survival evolution & species genome checkpointing.
    6. Molecular epigenetic NR3C1 methylation, 5-HT1A, & ACh-M1 receptor divergence.
    7. Non-REM Slow-Wave Sleep adenosine clearance, CA3 SWR replay consolidation, & SHY scaling.
    8. Monozygotic twin maturation: 0.0 birth connectome distance -> diverged personalities.
    9. Pandora sanctuary maturation: enriched ACh-M1 curiosity & consolidated CA3 engrams.
    10. MaturedSoulCapsule (.soul.pt + .soul.json) round-trip & SHA-256 tamper detection.
    11. LiveSoulLLMSocket autonomous 24/7 heartbeat & 249D calibrated bridge readout.
    12. PyTorch register_forward_hook LLM residual-stream steering, death silence, & clean detach.
    13. Identical projector sockets loaded with ADAM-01 vs ABEL-02 emit diverged LLM steering.
    14. Sated epistemic curiosity surge vs acute nociceptive suppression assay.
    15. Conspecific rival territorial vigilance assay.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest
import torch
import torch.nn as nn

from limbic_zero.core.soma import DigitalSoma
from limbic_zero.evolution import (
    BIOPHYSICAL_COORD_NAMES,
    ConnectomeDivergenceAnalyzer,
    EcologicalEvalConfig,
    EndogenousSurvivalEvaluator,
    EpigeneticPlasticityEngine,
    EpigeneticTemperamentProfile,
    InstinctValidationSuite,
    MaturedSoulCapsule,
    MorphogeneticGenomeConfig,
    MorphogeneticGenomeExpressor,
    NurseryConfig,
    OntogeneticBiographicalNursery,
    PhylogeneticESConfig,
    PhylogeneticInstinctEvolver,
    SlowWaveSleepConsolidator,
    build_foraging_and_hazard_niche,
)
from limbic_zero.mind.amygdala_hippocampus import EpisodicEngram
from limbic_zero.mind.biophysics import DalesLawSynapse
from limbic_zero.mind.brain import LimbicBrain


EVOLUTION_DIR = Path(__file__).resolve().parents[1] / "src" / "limbic_zero" / "evolution"

FORBIDDEN_EMOTION_TOKENS = {
    "anger", "angry", "fear", "afraid", "joy", "happy", "happiness",
    "sad", "sadness", "jealous", "jealousy", "envy", "pride", "shame",
    "guilt", "hate", "hatred", "love", "revenge", "vengeance", "mood",
    "emotion", "emotions", "emotion_state",
}
FORBIDDEN_MODULES_AND_LAYERS = {
    "sklearn", "scikit_learn", "scipy",
    "Transformer", "TransformerEncoder", "TransformerDecoder",
    "MultiheadAttention", "scaled_dot_product_attention",
    "LSTM", "LSTMCell", "GRU", "GRUCell", "RNN", "RNNCell",
}


def _tokenize_identifier(name: str) -> set[str]:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return {tok.lower() for tok in s2.split("_") if tok}


@pytest.fixture(scope="module")
def matured_bundle():
    """Run a compact end-to-end evolution + nursery maturation fixture shared across tests 8-15."""
    g_cfg = MorphogeneticGenomeConfig()
    expressor = MorphogeneticGenomeExpressor(g_cfg)
    founder = expressor.sample_initial_genome(seed=101, init_std=0.06)

    nursery = OntogeneticBiographicalNursery(
        expressor=expressor,
        nursery_config=NurseryConfig(
            seasons=3,
            waking_ticks_per_season=45,
            sleep_ticks_per_night=25,
            epigenetic_rate_scale=0.08,
            record_stride=2,
        ),
    )
    adam_cap, abel_cap, twin_meta = nursery.mature_cain_and_abel_twins(founder, arena_seed=303)
    pandora_cap = nursery.mature_pandora_explorer(founder, arena_seed=606)
    return expressor, founder, adam_cap, abel_cap, pandora_cap, twin_meta


def test_01_ast_zero_emotion_labels_and_forbidden_imports_in_evolution() -> None:
    py_files = sorted(EVOLUTION_DIR.glob("*.py"))
    assert len(py_files) >= 6

    for py_path in py_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bad = _tokenize_identifier(node.name) & FORBIDDEN_EMOTION_TOKENS
                assert not bad, f"Forbidden emotion token {bad} in {py_path.name}:{node.lineno}"
            elif isinstance(node, ast.Name):
                bad = _tokenize_identifier(node.id) & FORBIDDEN_EMOTION_TOKENS
                assert not bad, f"Forbidden emotion variable {bad} in {py_path.name}:{node.lineno}"
                assert node.id not in FORBIDDEN_MODULES_AND_LAYERS, (
                    f"Forbidden shortcut layer {node.id} in {py_path.name}:{node.lineno}"
                )
            elif isinstance(node, ast.Attribute):
                bad = _tokenize_identifier(node.attr) & FORBIDDEN_EMOTION_TOKENS
                assert not bad, f"Forbidden emotion attr {bad} in {py_path.name}:{node.lineno}"
                assert node.attr not in FORBIDDEN_MODULES_AND_LAYERS, (
                    f"Forbidden shortcut attr {node.attr} in {py_path.name}:{node.lineno}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_MODULES_AND_LAYERS
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in FORBIDDEN_MODULES_AND_LAYERS


def test_02_morphogenetic_genome_expressor_determinism_and_dales_law() -> None:
    expressor = MorphogeneticGenomeExpressor()
    g1 = expressor.sample_initial_genome(seed=42, init_std=0.15)
    g2 = expressor.sample_initial_genome(seed=99, init_std=0.15)

    b1_a = expressor.build_expressed_brain(g1, scaffold_seed=555)
    b1_b = expressor.build_expressed_brain(g1, scaffold_seed=555)
    b2 = expressor.build_expressed_brain(g2, scaffold_seed=555)

    snap_1a = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(b1_a)
    snap_1b = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(b1_b)
    snap_2 = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(b2)

    assert ConnectomeDivergenceAnalyzer.compute_regional_l2_divergence(snap_1a, snap_1b)["total_connectome_l2"] == 0.0
    assert ConnectomeDivergenceAnalyzer.compute_regional_l2_divergence(snap_1a, snap_2)["total_connectome_l2"] > 0.25

    for mod in b1_a.modules():
        if isinstance(mod, DalesLawSynapse):
            assert float(mod.effective_weights().min().item()) >= 0.0


def test_03_mirror_paired_ecological_trials_symmetry_and_endogenous_fitness() -> None:
    w_pos = build_foraging_and_hazard_niche(mirror_sign=+1, seed=12)
    w_neg = build_foraging_and_hazard_niche(mirror_sign=-1, seed=12)

    # Oasis X coordinates must mirror symmetrically around arena center x = 18.0!
    dx_pos = w_pos.chargers[0].x - 18.0
    dx_neg = w_neg.chargers[0].x - 18.0
    assert abs(dx_pos + dx_neg) < 1e-5

    expressor = MorphogeneticGenomeExpressor()
    evaluator = EndogenousSurvivalEvaluator(
        expressor=expressor,
        eval_config=EcologicalEvalConfig(ticks_per_trial=25),
    )
    genes = expressor.sample_initial_genome(seed=7)
    mean_fit, outcomes = evaluator.evaluate_genome(genes, trial_seed=50, include_contested_duel=True)
    assert len(outcomes) == 5
    assert isinstance(mean_fit, float)
    assert all(m.initial_free_energy > 0.1 for m in outcomes)


def test_04_centered_rank_utilities_invariance_and_zero_mean() -> None:
    raw_a = [-1000.0, 0.2, 0.5, 99999.0]
    raw_b = [-1.0, 10.0, 20.0, 30.0]
    u_a = PhylogeneticInstinctEvolver.compute_centered_ranks(raw_a)
    u_b = PhylogeneticInstinctEvolver.compute_centered_ranks(raw_b)
    assert torch.allclose(u_a, u_b, atol=1e-6)
    assert abs(float(u_a.mean().item())) < 1e-6
    assert abs(float(u_a.min().item()) + 0.5) < 1e-6
    assert abs(float(u_a.max().item()) - 0.5) < 1e-6


def test_05_phylogenetic_baldwin_evolution_checkpoint_and_monotonic_champion(tmp_path: Path) -> None:
    evolver = PhylogeneticInstinctEvolver(
        es_config=PhylogeneticESConfig(
            population_pairs=2,
            generations=2,
            num_workers=1,
            include_contested_duel=False,
            seed=123,
        ),
        eval_config=EcologicalEvalConfig(ticks_per_trial=20, continuum_prewarm_steps=6),
    )
    champ = evolver.run()
    assert champ.shape == (evolver.expressor.genome_dim,)
    assert len(evolver.history) == 2
    assert evolver.champion_fitness >= evolver.hall_of_fame[0][0]

    ckpt_file = tmp_path / "species.genome.pt"
    saved = evolver.save_species_genome(str(ckpt_file))
    assert ckpt_file.exists()
    assert saved["genome_dim"] == evolver.expressor.genome_dim


def test_06_epigenetic_nr3c1_methylation_and_5ht1a_divergence() -> None:
    engine = EpigeneticPlasticityEngine(rate_scale=0.25)
    stressed = EpigeneticTemperamentProfile()
    nurtured = EpigeneticTemperamentProfile()

    brain_s = LimbicBrain(seed=1)
    soma_s = DigitalSoma()
    soma_s.reset()

    brain_n = LimbicBrain(seed=1)
    soma_n = DigitalSoma()
    soma_n.reset()

    # Construct real closed-loop diagnostics under severe cortisol/LHb vs oxytocin/vagal + novelty
    from limbic_zero.sandbox.world import TrumanDomeWorld

    w = TrumanDomeWorld(seed=1)
    w.add_organism("A", x=10.0, y=10.0)
    obs = w.get_observations()["A"]
    _, diag_s = brain_s.step(obs, soma_s, dt=0.05)
    _, diag_n = brain_n.step(obs, soma_n, dt=0.05)

    soma_s.endo.adrenal_cortisol = 0.88
    soma_s.endo.ventral_vagal_tone = 0.12
    diag_s.neuromodulators.lhb_aversive_rate = 1.45
    diag_s.amygdala.cea_defensive_output = 0.75

    soma_n.endo.adrenal_cortisol = 0.14
    soma_n.endo.oxytocin_systemic = 0.78
    soma_n.endo.ventral_vagal_tone = 0.88
    diag_n.neuromodulators.lhb_aversive_rate = 0.08
    diag_n.neuromodulators.td_prediction_error = 0.65
    diag_n.thalamus.epistemic_novelty = 0.85
    diag_n.amygdala.cea_defensive_output = 0.02

    for _ in range(30):
        engine.update_from_waking_tick(stressed, soma_s, diag_s, dt=0.10)
        engine.update_from_waking_tick(nurtured, soma_n, diag_n, dt=0.10)

    assert stressed.nr3c1_gr_methylation > nurtured.nr3c1_gr_methylation + 0.15
    assert soma_s.endo.gr_receptor_sensitivity < soma_n.endo.gr_receptor_sensitivity
    assert stressed.ht1a_promoter_expression < nurtured.ht1a_promoter_expression
    assert nurtured.ach_m1_expression > stressed.ach_m1_expression


def test_07_slow_wave_sleep_clears_adenosine_and_fires_swr_consolidation() -> None:
    brain = LimbicBrain(seed=42)
    soma = DigitalSoma()
    soma.reset()
    soma.metab.brain_adenosine = 0.76
    soma.metab.brain_atp = 0.72

    # Seed a high-shield trauma engram into CA3 so SWR replay triggers during NREM sleep
    cue = torch.ones(76, dtype=torch.float32) * 0.35
    ctx = torch.ones(192, dtype=torch.float32) * 0.20
    dg = brain.hippocampus.compute_dg_sparse_code(cue, ctx)
    brain.hippocampus.engrams.append(
        EpisodicEngram(
            dg_sparse_code=dg,
            ca1_context_vec=ctx,
            extero_cue_snapshot=cue,
            action_taken=torch.tensor([-0.6, 0.5, 0.2, 0.4, 0.8, 0.0, 0.0]),
            somatic_marker=torch.tensor([-0.4, 1.1, 0.8, 0.2]),
            consolidation_shield=0.92,
            vitality=1.0,
            timestamp=1.0,
        )
    )
    bla_before = float(torch.linalg.vector_norm(brain.amygdala.w_bla_defensive).item())

    consolidator = SlowWaveSleepConsolidator()
    rep = consolidator.run_sleep_cycle(brain=brain, soma=soma, sleep_ticks=50, dt=0.05)

    assert rep.swr_bursts_fired >= 1
    assert rep.adenosine_after < rep.adenosine_before
    assert soma.metab.brain_atp > 0.72
    bla_after = float(torch.linalg.vector_norm(brain.amygdala.w_bla_defensive).item())
    assert bla_after > bla_before
    for mod in brain.pallium.modules():
        if isinstance(mod, DalesLawSynapse):
            assert float(mod.effective_weights().min().item()) >= 0.0


def test_08_monozygotic_twins_start_identical_and_diverge_into_distinct_personalities(matured_bundle) -> None:
    _, _, adam_cap, abel_cap, _, twin_meta = matured_bundle
    assert twin_meta["initial_twin_connectome_l2"] == 0.0
    assert twin_meta["final_sibling_connectome_divergence"]["total_connectome_l2"] > 0.05
    assert adam_cap.epigenetic_profile.nr3c1_gr_methylation > abel_cap.epigenetic_profile.nr3c1_gr_methylation
    assert adam_cap.epigenetic_profile.ht1a_promoter_expression < abel_cap.epigenetic_profile.ht1a_promoter_expression


def test_09_pandora_explorer_matures_cholinergic_curiosity_and_engrams(matured_bundle) -> None:
    _, _, _, _, pandora_cap, _ = matured_bundle
    assert pandora_cap.epigenetic_profile.ach_m1_expression >= 1.0
    assert len(pandora_cap.consolidated_engrams) >= 1
    assert pandora_cap.calibration.cortical_participation_ratio >= 1.0


def test_10_soul_capsule_save_load_and_cryptographic_tamper_detection(tmp_path: Path, matured_bundle) -> None:
    _, _, adam_cap, _, _, _ = matured_bundle
    pt_path, json_path = adam_cap.save(tmp_path / "adam_test.soul.pt")
    assert pt_path.exists() and json_path.exists()

    loaded = MaturedSoulCapsule.load(pt_path, verify_integrity=True)
    assert loaded.capsule_sha256 == adam_cap.capsule_sha256
    assert len(loaded.consolidated_engrams) == len(adam_cap.consolidated_engrams)

    brain, soma = loaded.instantiate_living_mind_and_soma()
    assert len(brain.hippocampus.engrams) == len(adam_cap.consolidated_engrams)
    expected_gr = max(0.25, min(1.15, 1.05 - 0.85 * loaded.epigenetic_profile.nr3c1_gr_methylation))
    assert abs(soma.endo.gr_receptor_sensitivity - expected_gr) < 1e-6
    assert abs(soma.endo.gr_receptor_sensitivity - loaded.neuromodulator_resting_summary["gr_receptor_sensitivity"]) < 1e-2

    # Cryptographic tamper check: corrupt a single synapse in saved file -> must reject!
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    first_key = next(iter(raw["brain_state_dict"].keys()))
    raw["brain_state_dict"][first_key] = raw["brain_state_dict"][first_key] + 0.5
    tampered_path = tmp_path / "tampered.soul.pt"
    torch.save(raw, tampered_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MaturedSoulCapsule.load(tampered_path, verify_integrity=True)


def test_11_live_soul_llm_socket_autonomous_heartbeat_and_249d_bridge(matured_bundle) -> None:
    _, _, adam_cap, _, _, _ = matured_bundle
    socket = adam_cap.create_llm_socket(llm_hidden_dim=128, num_soft_prefix_tokens=4, seed=42)
    z_before = socket.live_latent_228d.clone()

    for _ in range(6):
        diag = socket.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.5)
    assert diag.is_conscious_active
    assert not torch.allclose(z_before, socket.live_latent_228d)

    feat_249, coords = socket.current_bridge_features()
    assert feat_249.shape == (249,)
    assert torch.isfinite(feat_249).all()
    assert set(coords.keys()) == set(BIOPHYSICAL_COORD_NAMES)


def test_12_live_soul_llm_socket_register_forward_hook_on_frozen_decoder(matured_bundle) -> None:
    _, _, adam_cap, _, _, _ = matured_bundle

    class DummyFrozenBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(64, 64)
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, str]:
            return self.lin(h), "kv_ok"

    block = DummyFrozenBlock()
    w_orig = block.lin.weight.detach().clone()
    x = torch.randn(2, 5, 64)
    unsteered, tag = block(x)
    assert tag == "kv_ok"

    socket = adam_cap.create_llm_socket(llm_hidden_dim=64, seed=11)
    socket.step_autonomous_heartbeat(dt=0.05)
    socket.attach_residual_hook(block, alpha=1.0)
    steered, _ = block(x)
    assert not torch.allclose(unsteered, steered)
    assert torch.allclose(block.lin.weight, w_orig)

    # Somatic death silences residual steering to exact zero!
    socket.soma.is_alive = False
    dead_out, _ = block(x)
    assert torch.allclose(unsteered, dead_out, atol=1e-6)

    socket.soma.is_alive = True
    socket.detach_all_hooks()
    detached_out, _ = block(x)
    assert torch.allclose(unsteered, detached_out, atol=1e-6)


def test_13_identical_socket_projectors_emit_divergent_llm_steering_for_adam_vs_abel(matured_bundle) -> None:
    _, _, adam_cap, abel_cap, _, _ = matured_bundle
    sock_adam = adam_cap.create_llm_socket(llm_hidden_dim=128, num_soft_prefix_tokens=4, seed=777)
    sock_abel = abel_cap.create_llm_socket(llm_hidden_dim=128, num_soft_prefix_tokens=4, seed=777)

    for _ in range(8):
        sock_adam.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.75)
        sock_abel.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.75)

    d_adam = sock_adam.compute_residual_delta(1.0)
    d_abel = sock_abel.compute_residual_delta(1.0)
    p_adam = sock_adam.compute_soft_prefix_tokens(batch_size=1)
    p_abel = sock_abel.compute_soft_prefix_tokens(batch_size=1)

    assert p_adam.shape == (1, 4, 128)
    assert float(torch.linalg.vector_norm(d_adam - d_abel).item()) > 0.01
    assert float(torch.linalg.vector_norm(p_adam - p_abel).item()) > 0.01


def test_14_sated_curiosity_vs_acute_pain_suppression_assay(matured_bundle) -> None:
    expressor, _, _, _, pandora_cap, _ = matured_bundle
    suite = InstinctValidationSuite(expressor=expressor)
    ach_sated, nov_sated = suite._probe_pandora_curiosity(pandora_cap, acute_distress=False, seed=88)
    ach_pain, nov_pain = suite._probe_pandora_curiosity(pandora_cap, acute_distress=True, seed=88)
    assert ach_sated > ach_pain
    assert nov_sated > nov_pain


def test_15_conspecific_territorial_vigilance_assay(matured_bundle) -> None:
    expressor, _, adam_cap, _, _, _ = matured_bundle
    suite = InstinctValidationSuite(expressor=expressor)
    ne_alone, cea_alone = suite._probe_conspecific_vigilance(adam_cap, with_rival=False, seed=99)
    ne_rival, cea_rival = suite._probe_conspecific_vigilance(adam_cap, with_rival=True, seed=99)
    assert (ne_rival > ne_alone) or (cea_rival > cea_alone)
