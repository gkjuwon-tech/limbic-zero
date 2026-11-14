"""Scratch Antithetic Separable Natural Evolution Strategy ('PhylogeneticInstinctEvolver').

Implements Baldwin-Effect phylogenetic evolution over the morphogenetic connectome
genome ``theta in R^G`` with ZERO external RL/ES libraries:
    - Antithetic mirror sampling: ``theta_g +/- sigma_g * eps_i`` (halves gradient variance).
    - Centered rank-normalized fitness utilities ``u_k in [-0.5, +0.5]`` (Wierstra et al.,
      2014; Salimans et al., 2017) — invariant to outliers and strictly non-cheating.
    - Separable diagonal covariance / step-size adaptation (sep-NES / Adam-ES hybrid)
      with L2 metabolic wiring-cost regularization.
    - Co-evolutionary Hall-of-Fame opponent archive so contested scarcity trials pit
      candidates against genuine evolved conspecific competitors.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import math
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List, Tuple

import torch

from limbic_zero.evolution.ecological_trials import (
    EcologicalEvalConfig,
    EndogenousSurvivalEvaluator,
    TrialOutcomeMetrics,
)
from limbic_zero.evolution.genome import (
    MorphogeneticGenomeConfig,
    MorphogeneticGenomeExpressor,
)


@dataclass(frozen=True)
class PhylogeneticESConfig:
    """Hyperparameters for Antithetic Rank-Normalized Natural Evolution."""

    population_pairs: int = 8               # 8 antithetic pairs = 16 candidate genomes per generation
    generations: int = 16
    learning_rate: float = 0.24
    sigma_init: float = 0.34
    sigma_decay: float = 0.978
    sigma_min: float = 0.08
    momentum_beta: float = 0.75
    wiring_l2_decay: float = 0.004          # Biological metabolic cost of strong innate wiring
    seed: int = 2026
    num_workers: int = 2                    # Parallel CPU rollout workers (1 = sequential)
    include_contested_duel: bool = True


@dataclass
class GenerationRecord:
    """Per-generation evolutionary telemetry across the wild population."""

    generation: int
    pop_mean_fitness: float
    pop_best_fitness: float
    champion_eval_fitness: float
    pop_survival_rate: float
    champion_mean_free_energy: float
    champion_fe_reduction: float
    genome_l2_norm: float
    sigma_scalar: float
    elapsed_sec: float


def _worker_evaluate_candidate(payload: Tuple[Any, ...]) -> Tuple[int, float, List[TrialOutcomeMetrics]]:
    """Top-level multiprocessing worker entry point with single-thread PyTorch setting."""
    (
        cand_idx,
        cand_list,
        opp_list,
        genome_cfg,
        eval_cfg,
        trial_seed,
        include_duel,
    ) = payload
    torch.set_num_threads(1)
    expressor = MorphogeneticGenomeExpressor(config=genome_cfg)
    evaluator = EndogenousSurvivalEvaluator(expressor=expressor, eval_config=eval_cfg)
    genes = torch.tensor(cand_list, dtype=torch.float32)
    opp = torch.tensor(opp_list, dtype=torch.float32) if opp_list is not None else None
    fit, metrics = evaluator.evaluate_genome(
        genes=genes,
        trial_seed=trial_seed,
        include_contested_duel=include_duel,
        opponent_genes=opp,
    )
    return cand_idx, fit, metrics


class PhylogeneticInstinctEvolver:
    """Evolves species-level pre-linguistic survival & foraging instincts from scratch."""

    def __init__(
        self,
        es_config: PhylogeneticESConfig | None = None,
        genome_config: MorphogeneticGenomeConfig | None = None,
        eval_config: EcologicalEvalConfig | None = None,
    ) -> None:
        self.es_config = es_config or PhylogeneticESConfig()
        self.genome_config = genome_config or MorphogeneticGenomeConfig()
        self.eval_config = eval_config or EcologicalEvalConfig()

        self.expressor = MorphogeneticGenomeExpressor(config=self.genome_config)
        self.evaluator = EndogenousSurvivalEvaluator(
            expressor=self.expressor,
            eval_config=self.eval_config,
        )

        self.mean_genome = self.expressor.sample_initial_genome(
            seed=self.es_config.seed,
            init_std=0.05,
        )
        self.velocity = torch.zeros_like(self.mean_genome)
        self.diag_scale = torch.ones_like(self.mean_genome)
        self.sigma = float(self.es_config.sigma_init)

        self.champion_genome = self.mean_genome.clone()
        self.champion_fitness: float = -float("inf")
        self.champion_metrics: List[TrialOutcomeMetrics] = []
        self.hall_of_fame: List[Tuple[float, torch.Tensor]] = []
        self.history: List[GenerationRecord] = []

    @staticmethod
    def compute_centered_ranks(fitnesses: List[float]) -> torch.Tensor:
        """Map raw scalar fitnesses to centered linear ranks in ``[-0.5, +0.5]``."""
        n = len(fitnesses)
        if n <= 1:
            return torch.zeros(n, dtype=torch.float32)
        order = sorted(range(n), key=fitnesses.__getitem__)
        ranks = torch.empty(n, dtype=torch.float32)
        for rank_pos, orig_idx in enumerate(order):
            ranks[orig_idx] = (float(rank_pos) / float(n - 1)) - 0.5
        return ranks

    def _evaluate_batch(
        self,
        candidates: List[torch.Tensor],
        trial_seed: int,
        opponent_genome: torch.Tensor,
        executor: ProcessPoolExecutor | None = None,
    ) -> List[Tuple[float, List[TrialOutcomeMetrics]]]:
        opp_list = opponent_genome.tolist()
        if executor is not None and len(candidates) > 1:
            tasks = [
                (
                    idx,
                    c.tolist(),
                    opp_list,
                    self.genome_config,
                    self.eval_config,
                    trial_seed,
                    self.es_config.include_contested_duel,
                )
                for idx, c in enumerate(candidates)
            ]
            results: List[Tuple[float, List[TrialOutcomeMetrics]] | None] = [None] * len(candidates)
            for idx, fit, met in executor.map(_worker_evaluate_candidate, tasks):
                results[idx] = (fit, met)
            return [r for r in results if r is not None]

        seq_out: List[Tuple[float, List[TrialOutcomeMetrics]]] = []
        for c in candidates:
            fit, met = self.evaluator.evaluate_genome(
                genes=c,
                trial_seed=trial_seed,
                include_contested_duel=self.es_config.include_contested_duel,
                opponent_genes=opponent_genome,
            )
            seq_out.append((fit, met))
        return seq_out

    def run(
        self,
        progress_callback: Callable[[GenerationRecord], None] | None = None,
    ) -> torch.Tensor:
        """Execute phylogenetic natural evolution and return the champion species genome."""
        cfg = self.es_config
        gen_rng = torch.Generator()
        gen_rng.manual_seed(cfg.seed + 11)

        # Evaluate Generation-0 unadapted founder baseline on held-out benchmark seed
        init_fit, init_metrics = self.evaluator.evaluate_genome(
            genes=self.mean_genome,
            trial_seed=cfg.seed + 9000,
            include_contested_duel=cfg.include_contested_duel,
            opponent_genes=self.mean_genome,
        )
        self.champion_fitness = init_fit
        self.champion_genome = self.mean_genome.clone()
        self.champion_metrics = init_metrics
        self.hall_of_fame.append((init_fit, self.mean_genome.clone()))

        use_mp = cfg.num_workers > 1
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=ctx) if use_mp else None

        try:
            for gen_idx in range(1, cfg.generations + 1):
                t0 = time.perf_counter()
                n_pairs = cfg.population_pairs
                eps = torch.randn((n_pairs, self.expressor.genome_dim), generator=gen_rng, dtype=torch.float32)

                candidates: List[torch.Tensor] = []
                for i in range(n_pairs):
                    step_vec = self.sigma * self.diag_scale * eps[i]
                    candidates.append(self.mean_genome + step_vec)
                    candidates.append(self.mean_genome - step_vec)

                # Common Random Numbers (CRN) across antithetic pairs for ultra-sharp comparisons!
                gen_trial_seed = cfg.seed + gen_idx * 409
                opp_genome = self.hall_of_fame[-1][1]
                eval_results = self._evaluate_batch(
                    candidates=candidates,
                    trial_seed=gen_trial_seed,
                    opponent_genome=opp_genome,
                    executor=executor,
                )

                raw_fits = [r[0] for r in eval_results]
                all_metrics = [r[1] for r in eval_results]
                utilities = self.compute_centered_ranks(raw_fits)

                # Natural Evolution Strategy gradient estimate over antithetic pairs
                grad = torch.zeros_like(self.mean_genome)
                cov_grad = torch.zeros_like(self.diag_scale)
                for i in range(n_pairs):
                    u_pos = float(utilities[2 * i].item())
                    u_neg = float(utilities[2 * i + 1].item())
                    diff_u = u_pos - u_neg
                    sum_u = u_pos + u_neg
                    grad.add_(eps[i] * diff_u)
                    cov_grad.add_((eps[i] * eps[i] - 1.0) * sum_u)

                grad.div_(float(2 * n_pairs) * max(1e-4, self.sigma))
                cov_grad.div_(float(2 * n_pairs))

                # Momentum natural gradient + metabolic L2 wiring decay
                self.velocity.mul_(cfg.momentum_beta).add_(grad * (1.0 - cfg.momentum_beta))
                self.mean_genome.mul_(1.0 - cfg.wiring_l2_decay).add_(
                    cfg.learning_rate * self.diag_scale * self.velocity
                )
                # Gentle diagonal covariance adaptation clamped for stability
                self.diag_scale.mul_(torch.exp(0.08 * torch.clamp(cov_grad, -0.5, 0.5))).clamp_(0.55, 1.65)
                self.sigma = max(cfg.sigma_min, self.sigma * cfg.sigma_decay)

                # Validate updated mean + best candidate on fixed benchmark seed
                best_cand_idx = int(max(range(len(raw_fits)), key=raw_fits.__getitem__))
                probe_candidates = [self.mean_genome, candidates[best_cand_idx]]
                probe_evals = self._evaluate_batch(
                    candidates=probe_candidates,
                    trial_seed=cfg.seed + 9000,
                    opponent_genome=self.champion_genome,
                    executor=executor,
                )
                for p_vec, (p_fit, p_met) in zip(probe_candidates, probe_evals):
                    if p_fit > self.champion_fitness:
                        self.champion_fitness = p_fit
                        self.champion_genome = p_vec.clone()
                        self.champion_metrics = p_met
                        self.hall_of_fame.append((p_fit, p_vec.clone()))
                        if len(self.hall_of_fame) > 6:
                            self.hall_of_fame.pop(0)

                surv_flags = [
                    1.0 if m.survived else 0.0
                    for m_list in all_metrics
                    for m in m_list
                ]
                surv_rate = sum(surv_flags) / float(max(1, len(surv_flags)))
                champ_fe = sum(m.mean_free_energy for m in self.champion_metrics) / float(
                    max(1, len(self.champion_metrics))
                )
                champ_red = sum(m.free_energy_reduction for m in self.champion_metrics) / float(
                    max(1, len(self.champion_metrics))
                )

                rec = GenerationRecord(
                    generation=gen_idx,
                    pop_mean_fitness=sum(raw_fits) / float(len(raw_fits)),
                    pop_best_fitness=max(raw_fits),
                    champion_eval_fitness=self.champion_fitness,
                    pop_survival_rate=surv_rate,
                    champion_mean_free_energy=champ_fe,
                    champion_fe_reduction=champ_red,
                    genome_l2_norm=float(torch.linalg.vector_norm(self.champion_genome).item()),
                    sigma_scalar=self.sigma,
                    elapsed_sec=time.perf_counter() - t0,
                )
                self.history.append(rec)
                if progress_callback is not None:
                    progress_callback(rec)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        return self.champion_genome.clone()

    def save_species_genome(self, path: str) -> Dict[str, Any]:
        """Serialize the evolved species genome, founder baseline, and evolutionary history."""
        payload = {
            "format_version": "limbic0.species_genome.v1",
            "genome_dim": self.expressor.genome_dim,
            "genome_config": asdict(self.genome_config),
            "es_config": asdict(self.es_config),
            "champion_genome": self.champion_genome.detach().cpu(),
            "champion_fitness": self.champion_fitness,
            "founder_initial_fitness": (
                self.hall_of_fame[0][0] if self.hall_of_fame else self.champion_fitness
            ),
            "history": [asdict(r) for r in self.history],
        }
        torch.save(payload, path)
        return payload
