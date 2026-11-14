"""Endogenous Cholinergic Curiosity Drive & Autonomous Open-World Foraging Scheduler.

Models how real organisms explore the unknown without external task prompts:
  - When ``ACh M1`` receptor occupancy + epigenetic ``ach_m1_expression`` + unresolved
    idle boredom exceed threshold — and acute ``CeA`` fear / hyper-``Cortisol`` panic do
    NOT hijack executive resources — the organism experiences endogenous epistemic hunger.
  - When Flat-Affect Ablation is active (emotions clamped to zero), ``curiosity_hunger``
    collapses to ``0.0`` and zero autonomous exploration occurs.
  - During an autonomous foraging episode, every organism receives the exact same neutral,
    zero-persona, zero-topic trigger so any divergence in chosen research domain, tool
    strategy, and written synthesis arises strictly from its 6D neurobiology & cortical weights.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Identical neutral trigger across all organisms — zero persona, zero topic hints, zero emotion words
NEUTRAL_AUTONOMOUS_FORAGING_TRIGGER = (
    "자율 탐구 주기: 현재 가장 검증하거나 알아보고 싶은 구체적 주제나 현상 하나를 스스로 선택해, "
    "웹 검색(search_open_web), 위키백과(fetch_wikipedia_summary), arXiv 논문 검색(search_arxiv_papers), "
    "또는 파이썬 수식 검증(run_epistemic_python_probe) 도구를 사용해 직접 조사하고, "
    "핵심 발견을 `notes/research/` 하위 마크다운 파일에 기록한 뒤 결론을 정리해."
)


@dataclass
class AutonomousForagingRecord:
    """Telemetry record of a single self-directed open-world foraging episode."""

    episode_id: str
    timestamp_unix: float
    soul_id: str
    triggered_by: str
    curiosity_hunger_before: float
    curiosity_hunger_after: float
    tools_invoked: List[str]
    notes_written: List[str]
    information_gain_total: float
    ach_before: float
    ach_after: float
    da_after: float
    cortisol_ug_dl: float
    spoken_synthesis: str
    cortical_surgery_triggered: bool = False
    cortical_delta_norm: float = 0.0
    latency_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CuriosityDriveScheduler:
    """Tracks endogenous cholinergic epistemic hunger and orchestrates autonomous foraging."""

    def __init__(self, trigger_threshold: float = 0.42) -> None:
        self.trigger_threshold = float(trigger_threshold)
        self._idle_boredom_by_soul: Dict[str, float] = {}
        self._episodes_by_soul: Dict[str, List[AutonomousForagingRecord]] = {}
        self._last_forage_ts: Dict[str, float] = {}

    def tick_idle_boredom(self, soul_id: str, dt_sec: float = 1.25, flat_affect: bool = False) -> float:
        """Accumulate gentle epistemic restlessness while an organism sits idle in safety."""
        if flat_affect:
            self._idle_boredom_by_soul[soul_id] = 0.0
            return 0.0
        cur = self._idle_boredom_by_soul.get(soul_id, 0.12)
        nxt = min(1.0, cur + 0.014 * max(0.0, float(dt_sec)))
        self._idle_boredom_by_soul[soul_id] = nxt
        return nxt

    def compute_curiosity_hunger(
        self,
        soul: Any,
        flat_affect: bool = False,
    ) -> Dict[str, float]:
        """Compute endogenous curiosity drive in ``[0.0, 1.0]`` strictly from live neurobiology."""
        if flat_affect:
            return {
                "curiosity_hunger": 0.0,
                "ach_m1_occupancy": 0.0,
                "epistemic_axis": 0.0,
                "idle_boredom": 0.0,
                "panic_suppression": 0.0,
                "above_threshold": 0.0,
            }

        snap = soul.telemetry_snapshot()
        axes = snap.get("steering_axes_6d", {})
        epi = snap.get("epigenetics", {})
        nm = snap.get("neuromodulators", {})
        vit = snap.get("vitals", {})

        mind_diag = getattr(soul, "_last_mind_diag", None)
        if mind_diag is not None:
            ach_m1_occ = float(mind_diag.neuromodulators.ach_muscarinic_m1_occupancy)
        else:
            ach_m1_occ = min(1.0, float(nm.get("acetylcholine", 0.25)) * 1.1)

        epistemic_axis = float(axes.get("cholinergic_epistemic_drive", 0.0))
        ach_m1_expr = float(epi.get("ach_m1_expression", 1.0))
        cea_threat = float(nm.get("cea_threat", 0.15))
        cort_ug = float(vit.get("cortisol_ug_dl", 5.0))
        surv_urg = max(0.0, float(axes.get("power_rail_survival_urgency", 0.0)))
        idle_b = self._idle_boredom_by_soul.get(soul.organism_id, 0.18)

        # Panic / acute threat suppression: survival crisis suppresses playful open-ended curiosity!
        cort_excess = max(0.0, (cort_ug - 14.5) / 14.0)
        panic_suppression = min(
            0.95,
            0.55 * max(0.0, cea_threat - 0.42) + 0.50 * cort_excess + 0.42 * max(0.0, surv_urg - 0.55),
        )

        raw_drive = (
            0.34 * ach_m1_occ
            + 0.32 * max(0.0, epistemic_axis)
            + 0.18 * idle_b
            + 0.22 * max(0.0, (ach_m1_expr - 0.88) * 1.8)
            - panic_suppression
        )
        hunger = round(max(0.0, min(1.0, raw_drive)), 4)
        return {
            "curiosity_hunger": hunger,
            "ach_m1_occupancy": round(ach_m1_occ, 4),
            "epistemic_axis": round(epistemic_axis, 4),
            "idle_boredom": round(idle_b, 4),
            "panic_suppression": round(panic_suppression, 4),
            "above_threshold": 1.0 if hunger >= self.trigger_threshold else 0.0,
        }

    def run_autonomous_foraging_episode(
        self,
        soul: Any,
        agent_runner: Any,
        processor_or_tokenizer: Any,
        habitat: Any,
        *,
        cortical_surgeon: Optional[Any] = None,
        flat_affect: bool = False,
        force_even_if_low_hunger: bool = False,
        steer_gain: float = 1.0,
        max_tool_rounds: int = 4,
        max_new_tokens: int = 768,
        custom_focus_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a self-directed open-world foraging episode gated by endogenous curiosity."""
        t0 = time.perf_counter()
        sid = soul.organism_id
        pre_metrics = self.compute_curiosity_hunger(soul, flat_affect=flat_affect)
        hunger_before = pre_metrics["curiosity_hunger"]

        if flat_affect and not force_even_if_low_hunger:
            return {
                "status": "skipped_anhedonic_ablation",
                "soul_id": sid,
                "reason": "Flat-Affect ablation clamped curiosity hunger to 0.0 -> zero autonomous exploration.",
                "curiosity_metrics": pre_metrics,
            }

        if hunger_before < self.trigger_threshold and not force_even_if_low_hunger:
            return {
                "status": "below_curiosity_threshold",
                "soul_id": sid,
                "reason": (
                    f"Endogenous curiosity hunger ({hunger_before:.3f}) below threshold "
                    f"({self.trigger_threshold:.3f}); panic_suppression={pre_metrics['panic_suppression']:.3f}."
                ),
                "curiosity_metrics": pre_metrics,
            }

        pre_snap = soul.telemetry_snapshot()
        ach_before = float(pre_snap["neuromodulators"]["acetylcholine"])

        # Stimulate cholinergic curiosity slightly as exploration begins (if not ablated)
        if not flat_affect:
            from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus
            with soul.lock:
                soul._step_internal(ticks=10, stim=SensoryAfferentStimulus(epistemic_novelty=0.62))

        prompt_text = custom_focus_prompt or NEUTRAL_AUTONOMOUS_FORAGING_TRIGGER
        turn_out = agent_runner.run_embodied_turn(
            processor_or_tokenizer=processor_or_tokenizer,
            soul=soul,
            user_message=prompt_text,
            steer_enabled=(not flat_affect),
            steer_gain=(0.0 if flat_affect else steer_gain),
            agent_mode=True,
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
            max_tool_rounds=max_tool_rounds,
            temperature=0.68,
            stimulate_soma=(not flat_affect),
            use_history=False,
        )

        trace_items = turn_out.get("tool_trace") or turn_out.get("tool_calls") or []
        tools_called = [str(tc.get("tool_name", "")) for tc in trace_items]
        notes_written: List[str] = []
        info_gain_sum = 0.0
        for tc in trace_items:
            if tc.get("tool_name") == "write_workspace_file" and tc.get("allowed"):
                p = str((tc.get("arguments") or {}).get("path", ""))
                if p:
                    notes_written.append(p)
            preview = str(tc.get("response_preview", ""))
            if preview:
                info_gain_sum += 0.25

        # Ensure discoveries are also archived in `notes/research/` if the organism focused on web tools
        reply_txt = str(turn_out.get("reply", "")).strip()
        if reply_txt and not notes_written and habitat is not None and not flat_affect:
            auto_rel = f"notes/research/forage_{int(time.time())}_{uuid.uuid4().hex[:5]}.md"
            target_p = habitat.root / auto_rel
            target_p.parent.mkdir(parents=True, exist_ok=True)
            md_doc = (
                f"# Autonomous Epistemic Journal — {sid}\n"
                f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
                f"- Tools Used: {', '.join(tools_called) if tools_called else 'direct_synthesis'}\n"
                f"- Curiosity Drive: {hunger_before:.3f}\n\n"
                f"{reply_txt}\n"
            )
            target_p.write_text(md_doc[:16384], encoding="utf-8")
            notes_written.append(auto_rel)

        # Satisfy part of idle boredom upon completing an exploration episode
        if not flat_affect:
            self._idle_boredom_by_soul[sid] = max(0.04, self._idle_boredom_by_soul.get(sid, 0.2) * 0.30)
        self._last_forage_ts[sid] = time.time()

        # Optional immediate online affect-gated cortical weight surgery on the discovered insight!
        surg_triggered = False
        surg_delta = 0.0
        surgery_report: Optional[Dict[str, Any]] = None
        if cortical_surgeon is not None and reply_txt:
            evidence_traces = getattr(soul, "epistemic_episodes", [])
            recent_ev = "\n".join(e.distilled_text for e in evidence_traces[-3:])
            consolidation_text = (recent_ev + "\n\n" + reply_txt).strip()
            if len(consolidation_text) >= 40:
                surgery_report = cortical_surgeon.surgical_plasticity_step(
                    processor_or_tokenizer=processor_or_tokenizer,
                    soul=soul,
                    experience_text=consolidation_text,
                    flat_affect_ablation=flat_affect,
                    source_tag="autonomous_foraging",
                )
                surg_triggered = bool(surgery_report.get("updated", False))
                surg_delta = float(surgery_report.get("total_lora_frobenius_norm", 0.0))

        post_snap = soul.telemetry_snapshot()
        post_metrics = self.compute_curiosity_hunger(soul, flat_affect=flat_affect)
        rec = AutonomousForagingRecord(
            episode_id=f"forage_{uuid.uuid4().hex[:10]}",
            timestamp_unix=round(time.time(), 3),
            soul_id=sid,
            triggered_by="autonomous_curiosity",
            curiosity_hunger_before=hunger_before,
            curiosity_hunger_after=post_metrics["curiosity_hunger"],
            tools_invoked=tools_called,
            notes_written=notes_written,
            information_gain_total=round(min(3.0, info_gain_sum + (0.35 if reply_txt else 0.0)), 4),
            ach_before=round(ach_before, 4),
            ach_after=round(float(post_snap["neuromodulators"]["acetylcholine"]), 4),
            da_after=round(float(post_snap["neuromodulators"]["dopamine"]), 4),
            cortisol_ug_dl=round(float(post_snap["vitals"]["cortisol_ug_dl"]), 2),
            spoken_synthesis=reply_txt,
            cortical_surgery_triggered=surg_triggered,
            cortical_delta_norm=round(surg_delta, 6),
            latency_sec=round(time.perf_counter() - t0, 3),
        )
        hist = self._episodes_by_soul.setdefault(sid, [])
        hist.append(rec)
        if len(hist) > 40:
            self._episodes_by_soul[sid] = hist[-40:]

        return {
            "status": "ok",
            "episode": rec.to_dict(),
            "surgery_report": surgery_report,
            "research_notes": self.list_research_notes(habitat),
        }

    @staticmethod
    def list_research_notes(habitat: Any, limit: int = 15) -> List[Dict[str, Any]]:
        if habitat is None:
            return []
        notes_dir = Path(habitat.root) / "notes"
        if not notes_dir.is_dir():
            return []
        files = [p for p in notes_dir.rglob("*") if p.is_file() and not p.name.endswith(".svg")]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out: List[Dict[str, Any]] = []
        for p in files[:limit]:
            rel = p.relative_to(habitat.root).as_posix()
            txt = p.read_text(encoding="utf-8", errors="replace")[:3500]
            out.append(
                {
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                    "modified_unix": round(p.stat().st_mtime, 2),
                    "content": txt,
                }
            )
        return out

    def snapshot_for_all_souls(self, souls: Dict[str, Any], habitats: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for sid, soul in souls.items():
            hab = habitats.get(sid)
            eps = self._episodes_by_soul.get(sid, [])
            disc = getattr(soul, "epistemic_episodes", [])
            out[sid] = {
                "curiosity": self.compute_curiosity_hunger(soul),
                "episode_count": len(eps),
                "recent_episodes": [e.to_dict() for e in eps[-8:]],
                "recent_discoveries": [d.to_dict() for d in disc[-10:]],
                "research_notes": self.list_research_notes(hab, limit=10),
            }
        return out
