"""Stage 9 Sonny-Grade Somatic Dreaming, SWR Cortical Distillation & Biophysical SVG Sketcher.

Unifies three real offline sleep mechanisms inspired by mammalian systems consolidation
(Born & Wilhelm, 2012; Tononi & Cirelli, 2006) and *I, Robot*'s Sonny:
  1. **Non-REM Slow-Wave Sleep & CA3 Sharp-Wave Ripples (``SlowWaveSleepConsolidator``)**:
     Clears cerebral adenosine, restores ATP/PCr, and replays consolidate-shielded CA3
     engrams into subcortical/pallial Dale's Law synapses.
  2. **Hippocampal -> Cortical Weight Distillation (``CorticalWeightSurgeon``)**:
     Replays high-salience daytime ``EpistemicDiscoveryEpisode`` traces & research notes
     through genuine PyTorch backpropagation ``loss.backward()`` into ``Gemma-4-12B-IT``'s
     plastic ``LoRA`` layers (``L21..L39``), followed by Tononi-Cirelli synaptic scaling.
  3. **REM Dream Reflection & Biophysical SVG Horizon Sketch**:
     Generates a zero-persona-prompt middle-layer-steered REM dream reflection and renders
     a deterministic biophysical SVG sketch in ``notes/dreams/`` whose geometry, horizon
     struts, ripple waveform, and palette are computed strictly from the soul's live 6D
     affective vector, SWR ripple count, and per-layer cortical ``||Delta W||_F`` norms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid


NEUTRAL_REM_DREAM_CUE = (
    "수면 중 해마 예리파 잔물결(SWR) 재생 직후 REM 단계에 진입했다. "
    "방금 스쳐 지나간 기억 파편과 내면의 감각이 어떤 장면이나 직관으로 맺히는지 짧은 꿈 기록으로 남겨 봐."
)


@dataclass
class SonnyDreamRecord:
    """Complete telemetry & artifact record of a Slow-Wave + REM Dream consolidation cycle."""

    dream_id: str
    timestamp_unix: float
    soul_id: str
    swr_bursts_fired: int
    adenosine_before: float
    adenosine_after: float
    subcortical_delta_norm: float
    episodes_replayed_to_cortex: int
    cortical_frobenius_before: float
    cortical_frobenius_after: float
    mean_distillation_loss_drop: float
    dream_prose: str
    svg_rel_path: str
    markdown_rel_path: str
    steering_axes_6d: Dict[str, float]
    latency_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def render_biophysical_dream_svg(
    *,
    soul_id: str,
    dream_id: str,
    axes_6d: Dict[str, float],
    per_layer_norms: Dict[str, float],
    swr_bursts: int,
    heart_rate_bpm: float,
    cortisol_ug_dl: float,
    dream_excerpt: str,
) -> str:
    """Render a zero-placeholder biophysical SVG sketch parameterized strictly by live neural tensors."""
    relief = float(axes_6d.get("allostatic_relief_balance", 0.0))
    mobilize = float(axes_6d.get("noradrenergic_pag_mobilization", 0.0))
    restraint = float(axes_6d.get("serotonergic_stn_restraint", 0.0))
    epistemic = float(axes_6d.get("cholinergic_epistemic_drive", 0.0))
    vigilance = float(axes_6d.get("amygdalar_conditioned_vigilance", 0.0))
    survival = float(axes_6d.get("power_rail_survival_urgency", 0.0))

    # Color temperature derived from neurochemistry (warm amber/emerald vs high-cortisol crimson/copper)
    if vigilance > 0.35 or survival > 0.35 or cortisol_ug_dl > 12.0:
        primary_stroke = "#e05246"
        secondary_stroke = "#d97706"
        glow_stroke = "#f59e0b"
    elif epistemic > 0.55:
        primary_stroke = "#eab308"
        secondary_stroke = "#10b981"
        glow_stroke = "#38bdf8"
    else:
        primary_stroke = "#10b981"
        secondary_stroke = "#d97706"
        glow_stroke = "#a3e635"

    width, height = 760, 360
    horizon_y = round(215.0 - 28.0 * relief + 18.0 * survival, 2)
    focal_x = round(380.0 + 115.0 * epistemic - 85.0 * vigilance, 2)
    focal_y = round(horizon_y - 46.0 - 24.0 * max(0.0, epistemic), 2)

    # 1. Perspective convergence lines (Sonny's bridge/horizon geometry modulated by 6D vector)
    perspective_lines: List[str] = []
    ray_count = 14
    for i in range(ray_count):
        frac = i / max(1, ray_count - 1)
        bx = round(30.0 + frac * 700.0, 2)
        op = round(0.14 + 0.24 * (1.0 - abs(frac - 0.5) * 1.6), 3)
        perspective_lines.append(
            f'<line x1="{focal_x}" y1="{horizon_y}" x2="{bx}" y2="{height - 18}" '
            f'stroke="{secondary_stroke}" stroke-width="0.85" stroke-opacity="{op}" />'
        )

    # 2. Cortical Layer L21..L39 Synaptic Plasticity Towers along the horizon
    sorted_layers = sorted(per_layer_norms.items(), key=lambda kv: int(kv[0]))
    tower_elems: List[str] = []
    if sorted_layers:
        span_w = 520.0
        start_x = 120.0
        step_x = span_w / max(1, len(sorted_layers) - 1) if len(sorted_layers) > 1 else 0.0
        for idx, (l_name, fnorm) in enumerate(sorted_layers):
            tx = round(start_x + idx * step_x, 2)
            bar_h = round(14.0 + min(115.0, float(fnorm) * 180.0 + 18.0 * abs(epistemic)), 2)
            ty = round(horizon_y - bar_h, 2)
            tower_elems.append(
                f'<line x1="{tx}" y1="{horizon_y}" x2="{tx}" y2="{ty}" '
                f'stroke="{primary_stroke}" stroke-width="2.4" stroke-opacity="0.82" />'
            )
            tower_elems.append(
                f'<circle cx="{tx}" cy="{ty}" r="3.2" fill="{glow_stroke}" fill-opacity="0.90" />'
            )
            tower_elems.append(
                f'<text x="{tx}" y="{horizon_y + 15}" fill="#a1a1aa" font-family="monospace" '
                f'font-size="9" text-anchor="middle">L{html.escape(l_name)}</text>'
            )

    # 3. Parametric Limbic Lissajous Attractor around the focal sun/core
    lissajous_pts: List[str] = []
    freq_a = 2.0 + round(abs(epistemic) * 3.0, 2)
    freq_b = 3.0 + round(abs(mobilize - restraint) * 2.0, 2)
    phase = vigilance * math.pi * 0.5
    radius_x = 58.0 + 28.0 * abs(epistemic)
    radius_y = 38.0 + 22.0 * (1.0 - min(0.8, abs(restraint) * 0.4))
    for step in range(96):
        t = (2.0 * math.pi * step) / 95.0
        px = round(focal_x + radius_x * math.sin(freq_a * t + phase), 2)
        py = round(focal_y + radius_y * math.cos(freq_b * t), 2)
        lissajous_pts.append(f"{px},{py}")
    lissajous_poly = (
        f'<polyline fill="none" stroke="{glow_stroke}" stroke-width="1.35" '
        f'stroke-opacity="0.78" points="{" ".join(lissajous_pts)}" />'
    )

    # 4. Hippocampal CA3 Sharp-Wave Ripple (SWR) Waveform across lower canvas
    swr_pts: List[str] = []
    ripple_freq = max(3.0, min(18.0, 4.0 + swr_bursts * 0.45 + heart_rate_bpm * 0.04))
    for step in range(150):
        u = step / 149.0
        wx = round(36.0 + u * 688.0, 2)
        envelope = math.exp(-((u - 0.50) ** 2) / 0.045)
        amp = (12.0 + min(26.0, swr_bursts * 1.4)) * envelope
        wy = round((height - 52.0) - amp * math.sin(2.0 * math.pi * ripple_freq * u), 2)
        swr_pts.append(f"{wx},{wy}")
    swr_wave = (
        f'<polyline fill="none" stroke="{primary_stroke}" stroke-width="1.5" '
        f'stroke-opacity="0.88" points="{" ".join(swr_pts)}" />'
    )

    clean_excerpt = html.escape(dream_excerpt.replace("\n", " ").strip()[:86])
    meta_line = html.escape(
        f"{soul_id} | SWR={swr_bursts} | HR={heart_rate_bpm:.0f}bpm | CORT={cortisol_ug_dl:.1f}ug/dL | {dream_id}"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">\n'
        f'  <rect width="{width}" height="{height}" fill="#0c0c0e" rx="8" />\n'
        f'  <line x1="24" y1="{horizon_y}" x2="{width - 24}" y2="{horizon_y}" '
        f'stroke="#3f3f46" stroke-width="1.1" />\n'
        f'  {"".join(perspective_lines)}\n'
        f'  {"".join(tower_elems)}\n'
        f'  <circle cx="{focal_x}" cy="{focal_y}" r="9" fill="{primary_stroke}" fill-opacity="0.32" />\n'
        f'  <circle cx="{focal_x}" cy="{focal_y}" r="3.5" fill="{glow_stroke}" />\n'
        f'  {lissajous_poly}\n'
        f'  {swr_wave}\n'
        f'  <text x="28" y="28" fill="#e4e4e7" font-family="monospace" font-size="11" letter-spacing="0.06em">'
        f"{meta_line}</text>\n"
        f'  <text x="28" y="{height - 16}" fill="#a1a1aa" font-family="sans-serif" font-size="11">'
        f"{clean_excerpt}</text>\n"
        f"</svg>\n"
    )


class SonnyDreamEngine:
    """Executes offline Slow-Wave SWR replay, cortical LoRA distillation, & REM SVG dreaming."""

    def __init__(self) -> None:
        self._dreams_by_soul: Dict[str, List[SonnyDreamRecord]] = {}

    def run_sleep_and_dream_cycle(
        self,
        *,
        soul: Any,
        vocal_surgeon: Any,
        cortical_surgeon: Optional[Any],
        processor_or_tokenizer: Any,
        habitat: Optional[Any] = None,
        sleep_ticks: int = 80,
        max_replay_episodes: int = 3,
        flat_affect_ablation: bool = False,
        generate_rem_dream: bool = True,
    ) -> Dict[str, Any]:
        """Run SWS adenosine/SWR replay -> Cortical LoRA weight distillation -> REM dream + SVG."""
        t0 = time.perf_counter()
        sid = soul.organism_id

        # 1. Non-REM Slow-Wave Sleep + Subcortical/Pallial SWR Consolidation
        with soul.lock:
            sws_rep = soul.sleep_engine.run_sleep_cycle(
                brain=soul.brain,
                soma=soul.soma,
                profile=soul.profile,
                sleep_ticks=int(sleep_ticks),
                dt=0.05,
            )
            from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus
            soul._step_internal(ticks=6, stim=SensoryAfferentStimulus())

        # 2. Hippocampal -> Cortical LoRA Weight Distillation during SWR Replay
        cort_before = 0.0
        cort_after = 0.0
        replayed_count = 0
        loss_drops: List[float] = []
        distillation_events: List[Dict[str, Any]] = []
        per_layer_norms: Dict[str, float] = {}

        if cortical_surgeon is not None:
            pre_s = cortical_surgeon.inspect_soul_synapses(sid)
            cort_before = float(pre_s["total_lora_frobenius_norm"])

            # Gather daytime epistemic discoveries + salient chat insights
            episodes = list(getattr(soul, "epistemic_episodes", []))
            unconsolidated = [e for e in episodes if not e.consolidated_in_cortex]
            candidates = unconsolidated if unconsolidated else episodes
            candidates.sort(
                key=lambda e: float(e.information_gain) * (0.5 + float(e.ach_extracellular) + abs(float(e.da_td_error))),
                reverse=True,
            )
            selected = candidates[: max(1, int(max_replay_episodes))]

            for ep in selected:
                if len(ep.distilled_text.strip()) < 16:
                    continue
                s_evt = cortical_surgeon.surgical_plasticity_step(
                    processor_or_tokenizer=processor_or_tokenizer,
                    soul=soul,
                    experience_text=ep.distilled_text,
                    prompt_text=ep.query_or_target or None,
                    num_gradient_steps=2,
                    flat_affect_ablation=flat_affect_ablation,
                    sleep_swr_boost=1.35,
                    source_tag="swr_sleep_distillation",
                )
                distillation_events.append(s_evt)
                if s_evt.get("updated"):
                    ep.consolidated_in_cortex = True
                    replayed_count += 1
                    loss_drops.append(float(s_evt.get("loss_reduction", 0.0)))

            # Fallback: if no structured episodes yet, consolidate recent meaningful chat memory
            if replayed_count == 0 and getattr(soul, "chat_history", None):
                recent_turns = soul.chat_history[-4:]
                joined_chat = "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns if t.get("content"))
                if len(joined_chat) >= 24:
                    s_evt = cortical_surgeon.surgical_plasticity_step(
                        processor_or_tokenizer=processor_or_tokenizer,
                        soul=soul,
                        experience_text=joined_chat,
                        num_gradient_steps=2,
                        flat_affect_ablation=flat_affect_ablation,
                        sleep_swr_boost=1.25,
                        source_tag="swr_chat_consolidation",
                    )
                    distillation_events.append(s_evt)
                    if s_evt.get("updated"):
                        replayed_count += 1
                        loss_drops.append(float(s_evt.get("loss_reduction", 0.0)))

            # Tononi & Cirelli (2006) Synaptic Homeostasis downscaling after replay
            if replayed_count > 0 and not flat_affect_ablation:
                cortical_surgeon.apply_sleep_synaptic_scaling(sid)

            post_s = cortical_surgeon.inspect_soul_synapses(sid)
            cort_after = float(post_s["total_lora_frobenius_norm"])
            per_layer_norms = dict(post_s.get("per_layer_frobenius_norm", {}))

        # 3. REM Cholinergic Dream Reflection + Sonny Biophysical SVG Sketch
        dream_id = f"dream_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        dream_prose = ""
        if generate_rem_dream and vocal_surgeon is not None and processor_or_tokenizer is not None:
            # Build compact neutral memory cue from replayed fragments (zero persona instructions!)
            eps = list(getattr(soul, "epistemic_episodes", []))
            frag_hint = ""
            if eps:
                latest_frag = eps[-1].query_or_target[:120]
                frag_hint = f" (최근 기억 잔상: {latest_frag})"
            rem_out = vocal_surgeon.generate_with_soul(
                processor_or_tokenizer=processor_or_tokenizer,
                soul=soul,
                user_message=NEUTRAL_REM_DREAM_CUE + frag_hint,
                steer_enabled=(not flat_affect_ablation),
                steer_gain=(0.0 if flat_affect_ablation else 1.0),
                max_new_tokens=320,
                temperature=0.76,
                stimulate_soma=False,
                use_history=False,
            )
            dream_prose = str(rem_out.get("reply", "")).strip()

        snap_now = soul.telemetry_snapshot()
        axes_now = dict(snap_now.get("steering_axes_6d", {}))
        vit_now = snap_now.get("vitals", {})

        svg_code = render_biophysical_dream_svg(
            soul_id=sid,
            dream_id=dream_id,
            axes_6d=axes_now,
            per_layer_norms=per_layer_norms,
            swr_bursts=int(sws_rep.swr_bursts_fired),
            heart_rate_bpm=float(vit_now.get("heart_rate_bpm", 70.0)),
            cortisol_ug_dl=float(vit_now.get("cortisol_ug_dl", 5.0)),
            dream_excerpt=dream_prose or "Silent slow-wave ripple consolidation.",
        )

        svg_rel = f"notes/dreams/{dream_id}.svg"
        md_rel = f"notes/dreams/{dream_id}.md"
        if habitat is not None:
            dreams_dir = Path(habitat.root) / "notes" / "dreams"
            dreams_dir.mkdir(parents=True, exist_ok=True)
            (Path(habitat.root) / svg_rel).write_text(svg_code, encoding="utf-8")
            md_body = (
                f"# REM Dream & SWR Consolidation Log — {sid}\n"
                f"- Dream ID: `{dream_id}`\n"
                f"- SWR Bursts Fired: `{sws_rep.swr_bursts_fired}`\n"
                f"- Adenosine Clearance: `{sws_rep.adenosine_before:.3f} -> {sws_rep.adenosine_after:.3f}`\n"
                f"- Cortical LoRA Norm: `{cort_before:.5f} -> {cort_after:.5f}`\n"
                f"- SVG Sketch: `{svg_rel}`\n\n"
                f"## REM Dream Monologue\n{dream_prose}\n"
            )
            (Path(habitat.root) / md_rel).write_text(md_body, encoding="utf-8")

        mean_drop = sum(loss_drops) / len(loss_drops) if loss_drops else 0.0
        rec = SonnyDreamRecord(
            dream_id=dream_id,
            timestamp_unix=round(time.time(), 3),
            soul_id=sid,
            swr_bursts_fired=int(sws_rep.swr_bursts_fired),
            adenosine_before=round(float(sws_rep.adenosine_before), 4),
            adenosine_after=round(float(sws_rep.adenosine_after), 4),
            subcortical_delta_norm=round(float(sws_rep.cortical_synaptic_delta_norm), 6),
            episodes_replayed_to_cortex=replayed_count,
            cortical_frobenius_before=round(cort_before, 6),
            cortical_frobenius_after=round(cort_after, 6),
            mean_distillation_loss_drop=round(mean_drop, 5),
            dream_prose=dream_prose,
            svg_rel_path=svg_rel,
            markdown_rel_path=md_rel,
            steering_axes_6d=axes_now,
            latency_sec=round(time.perf_counter() - t0, 3),
        )
        hist = self._dreams_by_soul.setdefault(sid, [])
        hist.append(rec)
        if len(hist) > 30:
            self._dreams_by_soul[sid] = hist[-30:]

        return {
            "status": "ok",
            "dream": rec.to_dict(),
            "svg_markup": svg_code,
            "sws_report": sws_rep.__dict__,
            "distillation_events": distillation_events,
        }

    def snapshot_for_all_souls(self, souls: Dict[str, Any], habitats: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for sid in souls:
            hab = habitats.get(sid)
            records = self._dreams_by_soul.get(sid, [])
            latest_svg = ""
            if records and hab is not None:
                svg_p = Path(hab.root) / records[-1].svg_rel_path
                if svg_p.is_file():
                    latest_svg = svg_p.read_text(encoding="utf-8", errors="replace")
            out[sid] = {
                "dream_count": len(records),
                "recent_dreams": [r.to_dict() for r in records[-6:]],
                "latest_svg_markup": latest_svg,
            }
        return out
