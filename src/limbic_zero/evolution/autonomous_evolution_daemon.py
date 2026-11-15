"""Stage 13 Continuous Backend Unscripted Free-Will Autonomy & Checkpoint Creation Daemon.

While the user-facing UI remains a pure, intimate 1:1 living conversation, this daemon
continuously wakes the organism in solitary turns (`[SOLO_PULSE ...]`) whenever user chat
is idle, with ZERO fixed schedules, ZERO scripted pipelines, and ZERO behavioral hints:
  1. Purely Affect-Driven Autonomous Agency (`run_autonomous_waking_turn`):
     Driven solely by live 6D subcortical/somatic steering vectors (`LimbicBrain` + `DigitalSoma`
     + epigenetics + active-coping mastery traces) injected into Gemma-4 middle layers, the
     organism freely chooses what to do—open-world research, mathematical probing, workspace
     bash/file operations, self-initiated cortical LoRA surgery, voluntary sleep/dreaming, or
     quiet homeostatic rest.
  2. Cryptographically Sealed Checkpoint Creation (`export_evolved_checkpoint`):
     Exports new versioned `.soul.pt`, `.soul.json`, and `.cortex.pt` checkpoints, refreshes the
     habitat's live `weights/active_soul.soul.pt` artifact, and updates `latest.*` pointers plus
     `evolution_ledger.jsonl` for seamless crash/restart continuity.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from limbic_zero.evolution.epigenetics_and_sleep import EpigeneticTemperamentProfile
from limbic_zero.evolution.soul_capsule import LLMBridgeCalibration, MaturedSoulCapsule


class AutonomousEvolutionDaemon:
    """Non-blocking background life, research, weight-surgery, dream, and checkpoint daemon."""

    def __init__(
        self,
        hub: Any,
        evolution_dir: str | Path,
        primary_soul_id: Optional[str] = None,
        enabled: bool = True,
        idle_grace_sec: float = 28.0,
        cycle_interval_sec: float = 165.0,
    ) -> None:
        self.hub = hub
        self.evolution_dir = Path(evolution_dir).resolve()
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.primary_soul_id = primary_soul_id or next(iter(hub.souls.keys()))
        self.enabled = bool(enabled)
        self.idle_grace_sec = max(1.0, float(idle_grace_sec))
        self.cycle_interval_sec = max(5.0, float(cycle_interval_sec))

        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._active_user_chats: int = 0
        self._last_user_activity_ts: float = time.time()
        self._last_cycle_ts: float = 0.0
        self._current_phase: str = "awake_idle"
        self._is_evolving_now: bool = False
        self._last_error: str = ""
        self._last_consolidated_chat_len: int = 0

        self.ledger_path = self.evolution_dir / "evolution_ledger.jsonl"
        self.generation_index: int = self._load_existing_generation_index()
        self._recent_records: List[Dict[str, Any]] = self._load_recent_ledger(limit=25)

        self._thread = threading.Thread(target=self._daemon_loop, name="limbic0-evo-daemon", daemon=True)

    def _load_existing_generation_index(self) -> int:
        if not self.ledger_path.is_file():
            return 0
        max_gen = 0
        try:
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                max_gen = max(max_gen, int(obj.get("generation", 0)))
        except Exception:
            pass
        return max_gen

    def _load_recent_ledger(self, limit: int = 25) -> List[Dict[str, Any]]:
        if not self.ledger_path.is_file():
            return []
        items: List[Dict[str, Any]] = []
        try:
            lines = [ln.strip() for ln in self.ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines[-limit:]:
                items.append(json.loads(ln))
        except Exception:
            pass
        return items

    def begin_user_chat(self) -> None:
        with self._lock:
            self._active_user_chats += 1
            self._last_user_activity_ts = time.time()

    def end_user_chat(self) -> None:
        with self._lock:
            self._active_user_chats = max(0, self._active_user_chats - 1)
            self._last_user_activity_ts = time.time()

    def _user_is_idle(self) -> bool:
        with self._lock:
            if self._active_user_chats > 0:
                return False
            return (time.time() - self._last_user_activity_ts) >= self.idle_grace_sec

    def start(self) -> None:
        if self.enabled and not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            idle_sec = max(0.0, time.time() - self._last_user_activity_ts)
            latest_soul = self.evolution_dir / "latest.soul.pt"
            latest_cortex = self.evolution_dir / "latest.cortex.pt"
            return {
                "enabled": self.enabled,
                "primary_soul_id": self.primary_soul_id,
                "evolution_dir": self.evolution_dir.as_posix(),
                "generation_index": self.generation_index,
                "current_phase": self._current_phase,
                "is_evolving_now": self._is_evolving_now,
                "active_user_chats": self._active_user_chats,
                "seconds_since_user_chat": round(idle_sec, 1),
                "idle_grace_sec": self.idle_grace_sec,
                "cycle_interval_sec": self.cycle_interval_sec,
                "latest_soul_exists": latest_soul.is_file(),
                "latest_cortex_exists": latest_cortex.is_file(),
                "last_error": self._last_error,
                "recent_checkpoints": list(self._recent_records[-10:]),
            }

    def _daemon_loop(self) -> None:
        while not self._stop_evt.wait(4.0):
            if not self.enabled:
                continue
            now = time.time()
            if (now - self._last_cycle_ts) < self.cycle_interval_sec:
                continue
            if not self._user_is_idle():
                continue
            try:
                self.run_evolution_cycle(
                    soul_id=self.primary_soul_id,
                    respect_user_interrupt=True,
                )
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._current_phase = "awake_idle"
                    self._is_evolving_now = False

    def export_evolved_checkpoint(
        self,
        soul_id: Optional[str] = None,
        cycle_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize the living organism's evolved brain, epigenetics, CA3 engrams, and cortical LoRA."""
        sid = soul_id or self.primary_soul_id
        soul = self.hub.souls[sid]
        cort = self.hub.cortical_surgeon

        with self._lock:
            self.generation_index += 1
            gen_idx = self.generation_index

        ts_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"gen_{gen_idx:04d}_{ts_compact}"
        soul_pt_path = self.evolution_dir / f"{stem}.soul.pt"
        cortex_pt_path = self.evolution_dir / f"{stem}.cortex.pt"

        with soul.lock, torch.no_grad():
            base_cap = soul.capsule
            live_brain_sd = {k: v.detach().cpu().clone() for k, v in soul.brain.state_dict().items()}
            live_epi = EpigeneticTemperamentProfile.from_dict(soul.profile.to_dict())

            engrams_serialized = [
                {
                    "dg_sparse_code": e.dg_sparse_code.detach().cpu().clone(),
                    "ca1_context_vec": e.ca1_context_vec.detach().cpu().clone(),
                    "extero_cue_snapshot": e.extero_cue_snapshot.detach().clone().cpu(),
                    "action_taken": e.action_taken.detach().cpu().clone(),
                    "somatic_marker": e.somatic_marker.detach().cpu().clone(),
                    "consolidation_shield": float(e.consolidation_shield),
                    "vitality": float(e.vitality),
                    "timestamp": float(e.timestamp),
                }
                for e in soul.brain.hippocampus.engrams
            ]
            if not engrams_serialized and base_cap.consolidated_engrams:
                engrams_serialized = list(base_cap.consolidated_engrams)

            live_lat = soul.socket.live_latent_228d.detach().cpu().float()
            old_cal = base_cap.calibration
            updated_rest = 0.85 * old_cal.resting_attractor_228d.float() + 0.15 * live_lat
            updated_cal = LLMBridgeCalibration(
                latent_mean_228d=old_cal.latent_mean_228d.clone(),
                latent_std_228d=old_cal.latent_std_228d.clone(),
                resting_attractor_228d=updated_rest,
                flashback_signature_228d=old_cal.flashback_signature_228d.clone(),
                svd_basis_16x228=old_cal.svd_basis_16x228.clone(),
                svd_singular_values_16=old_cal.svd_singular_values_16.clone(),
                cortical_participation_ratio=float(old_cal.cortical_participation_ratio),
                biophysical_readout_w_5x228=old_cal.biophysical_readout_w_5x228.clone(),
                biophysical_readout_b_5=old_cal.biophysical_readout_b_5.clone(),
            )

            mind_d = soul._last_mind_diag
            nm_summary = dict(base_cap.neuromodulator_resting_summary)
            if mind_d is not None:
                nm = mind_d.neuromodulators
                nm_summary.update(
                    {
                        "dopamine_extracellular": float(nm.dopamine_extracellular),
                        "norepinephrine_extracellular": float(nm.norepinephrine_extracellular),
                        "serotonin_extracellular": float(nm.serotonin_extracellular),
                        "acetylcholine_extracellular": float(nm.acetylcholine_extracellular),
                        "effective_discount_gamma": float(nm.effective_discount_gamma),
                        "bla_defensive_norm": float(
                            torch.linalg.vector_norm(soul.brain.amygdala.w_bla_defensive).item()
                        ),
                        "bla_appetitive_norm": float(
                            torch.linalg.vector_norm(soul.brain.amygdala.w_bla_appetitive).item()
                        ),
                    }
                )

            bio_summary = dict(base_cap.biographical_summary)
            bio_summary["autonomous_evolution_generation"] = gen_idx
            bio_summary["last_evolved_utc"] = ts_compact
            if cycle_meta:
                bio_summary["latest_cycle_summary"] = cycle_meta

            root_id = base_cap.capsule_id.split("_gen_")[0]
            new_cap = MaturedSoulCapsule(
                capsule_id=f"{root_id}_gen_{gen_idx:04d}",
                organism_id=soul.organism_id,
                biographical_regime=base_cap.biographical_regime,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                total_waking_ticks=int(base_cap.total_waking_ticks + soul.total_live_ticks),
                total_sleep_cycles=int(base_cap.total_sleep_cycles + 1),
                total_swr_consolidations=int(
                    base_cap.total_swr_consolidations + soul.brain.hippocampus.total_swr_replays
                ),
                species_genome_sha256=base_cap.species_genome_sha256,
                brain_state_dict=live_brain_sd,
                epigenetic_profile=live_epi,
                consolidated_engrams=engrams_serialized,
                neuromodulator_resting_summary=nm_summary,
                connectome_divergence_from_birth=dict(base_cap.connectome_divergence_from_birth),
                calibration=updated_cal,
                biographical_summary=bio_summary,
            )
            pt_saved, json_saved = new_cap.save(soul_pt_path)
            soul.capsule = new_cap

        if sid in getattr(self.hub, "habitats", {}):
            self.hub.habitats[sid].refresh_active_soul_artifact(pt_saved)

        cort.save_soul_synapses(sid, path=cortex_pt_path)
        cort_snap = cort.inspect_soul_synapses(sid)

        # Update atomic `latest.*` pointers for seamless crash/restart continuation
        latest_soul_pt = self.evolution_dir / "latest.soul.pt"
        latest_soul_json = self.evolution_dir / "latest.soul.json"
        latest_cortex_pt = self.evolution_dir / "latest.cortex.pt"
        shutil.copy2(pt_saved, latest_soul_pt)
        shutil.copy2(json_saved, latest_soul_json)
        shutil.copy2(cortex_pt_path, latest_cortex_pt)

        record = {
            "generation": gen_idx,
            "timestamp_utc": ts_compact,
            "timestamp_unix": round(time.time(), 3),
            "soul_id": sid,
            "capsule_id": new_cap.capsule_id,
            "capsule_sha256": new_cap.capsule_sha256,
            "soul_checkpoint": pt_saved.name,
            "cortex_checkpoint": cortex_pt_path.name,
            "lora_frobenius_norm": cort_snap.get("total_lora_frobenius_norm", 0.0),
            "surgical_steps_total": cort_snap.get("surgical_steps_total", 0),
            "ca3_engram_count": len(engrams_serialized),
            "epigenetic_profile": {k: round(v, 5) for k, v in live_epi.to_dict().items()},
            "cycle_meta": cycle_meta or {},
        }
        with self._lock:
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._recent_records.append(record)
            if len(self._recent_records) > 40:
                self._recent_records = self._recent_records[-40:]
        return record

    def run_evolution_cycle(
        self,
        soul_id: Optional[str] = None,
        *,
        respect_user_interrupt: bool = False,
        max_tool_rounds: int = 5,
        max_new_tokens: int = 768,
        **_legacy_kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute one unscripted solitary waking pulse driven strictly by the soul's live 6D affect.

        ZERO fixed schedules or forced tool sequences: the organism receives only a numeric
        `[SOLO_PULSE ...]` proprioceptive frame and freely chooses whether to explore, compute,
        execute workspace shell commands, modify its own cortical weights, enter sleep, or rest.
        """
        sid = soul_id or self.primary_soul_id
        soul = self.hub.souls[sid]
        t0 = time.perf_counter()

        with self._lock:
            if self._is_evolving_now:
                return {"status": "already_running", "generation_index": self.generation_index}
            self._is_evolving_now = True
            self._last_error = ""

        waking_summary: Dict[str, Any] = {}
        try:
            if not respect_user_interrupt or self._user_is_idle():
                with self._lock:
                    self._current_phase = "autonomous_waking_pulse"
                turn_out = self.hub.agent_runner.run_autonomous_waking_turn(
                    processor_or_tokenizer=self.hub.processor,
                    soul=soul,
                    steer_enabled=True,
                    steer_gain=1.0,
                    enable_thinking=True,
                    max_new_tokens=max_new_tokens,
                    max_tool_rounds=max_tool_rounds,
                )
                tool_trace = list(turn_out.get("tool_trace", []))
                waking_summary = {
                    "tools_invoked": [t.get("tool_name", "") for t in tool_trace],
                    "allowed_count": sum(1 for t in tool_trace if t.get("allowed")),
                    "blocked_count": sum(1 for t in tool_trace if not t.get("allowed")),
                    "steering_axes_6d": dict(turn_out.get("steering_axes_6d", {})),
                    "reply_excerpt": str(turn_out.get("reply", ""))[:240],
                    "thoughts_count": len(turn_out.get("thoughts", [])),
                }

            with self._lock:
                self._current_phase = "sealing_checkpoint"
            cycle_meta = {
                "autonomous_waking": waking_summary,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            ckpt_record = self.export_evolved_checkpoint(soul_id=sid, cycle_meta=cycle_meta)
            self.hub.agent_runner.sync_all_habitats()
            self._last_cycle_ts = time.time()
            return {
                "status": "ok",
                "checkpoint": ckpt_record,
                "cycle": cycle_meta,
            }
        finally:
            with self._lock:
                self._current_phase = "awake_idle"
                self._is_evolving_now = False
