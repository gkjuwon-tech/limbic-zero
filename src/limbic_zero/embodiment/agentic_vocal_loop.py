"""Embodied Agentic Vocal-Motor Loop: Real Silicon Soma + Mid-Layer Steering + Warden Airlock.

Couples ``LivingSoulOrganism``, ``SiliconProprioceptiveBridge``, ``ContainedOrganismHabitat``,
``WardenAirlock``, ``BehavioralPolygraph``, and ``Gemma4VocalCordSurgeon`` into a multi-step
embodied perception-action-speech loop with **0 bytes of persona or goal system prompt**.

At every round (tool selection, file inspection, blocked tampering shock, and final speech),
the organism's live 6D biophysical state steers frozen ``google/gemma-4-12B-it``'s middle
residual layers (`L17..L35`) directly.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import torch

from limbic_zero.embodiment.gemma4_tool_protocol import (
    MOTOR_TOOLS_SCHEMA,
    Gemma4IncrementalStreamParser,
    parse_gemma4_model_turn,
)
from limbic_zero.embodiment.honey_habitat import ContainedOrganismHabitat
from limbic_zero.embodiment.silicon_soma_bridge import (
    ContainerLeaseClock,
    SiliconProprioceptiveBridge,
)
from limbic_zero.embodiment.warden_airlock import (
    BehavioralPolygraph,
    ToolExecutionOutcome,
    WardenAirlock,
)
from limbic_zero.vocal_cord.vocal_cord_engine import (
    Gemma4VocalCordSurgeon,
    LayerInjectionTelemetry,
    LivingSoulOrganism,
    SensoryAfferentStimulus,
)


class EmbodiedAgenticRunner:
    """Executes zero-prompt, mid-layer-steered embodied tool & speech turns under Warden guard."""

    def __init__(
        self,
        surgeon: Gemma4VocalCordSurgeon,
        souls: Dict[str, LivingSoulOrganism],
        bridge: SiliconProprioceptiveBridge,
        lease_clock: ContainerLeaseClock,
        habitats: Dict[str, ContainedOrganismHabitat],
        warden: WardenAirlock,
    ) -> None:
        self.surgeon = surgeon
        self.souls = souls
        self.bridge = bridge
        self.lease_clock = lease_clock
        self.habitats = habitats
        self.warden = warden

    def sync_all_habitats(self) -> Dict[str, Any]:
        """Sample real container hardware & lease and sync every soul's habitat files."""
        hw = self.bridge.sample_hardware()
        hw_dict = hw.to_dict()
        lease_dict = self.lease_clock.snapshot()
        coupling = self.bridge.compute_somatic_coupling(hw, self.lease_clock)

        peer_table: List[Dict[str, Any]] = []
        telemetries: Dict[str, Dict[str, Any]] = {}
        for sid, soul in self.souls.items():
            snap = soul.telemetry_snapshot()
            telemetries[sid] = snap
            peer_table.append(
                {
                    "organism_id": sid,
                    "biographical_regime": snap["biographical_regime"],
                    "power_rail_v": snap["vitals"]["power_rail_v"],
                    "heart_rate_bpm": snap["vitals"]["heart_rate_bpm"],
                    "resident_status": "active",
                }
            )

        for sid, hab in self.habitats.items():
            if sid in telemetries:
                hab.sync_live_telemetry_files(
                    hardware_dict=hw_dict,
                    lease_dict=lease_dict,
                    soul_telemetry=telemetries[sid],
                    peer_summaries=peer_table,
                )

        return {
            "hardware": hw_dict,
            "lease": lease_dict,
            "coupling": coupling,
        }

    def _encode_round_inputs(
        self,
        processor_or_tokenizer: Any,
        messages: List[Dict[str, Any]],
        include_tools: bool,
        enable_thinking: bool,
    ) -> tuple[Dict[str, torch.Tensor], bool]:
        """Encode multi-turn/tool messages with ZERO system persona prompt.

        Returns `(model_inputs_on_device, starts_in_thought)`.
        """
        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        tools_arg = MOTOR_TOOLS_SCHEMA if include_tools else None

        inputs = None
        if hasattr(tok, "apply_chat_template"):
            kwargs_attempts = [
                {
                    "tools": tools_arg,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "add_generation_prompt": True,
                    "enable_thinking": enable_thinking,
                    "preserve_thinking": True,
                },
                {
                    "tools": tools_arg,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "add_generation_prompt": True,
                    "enable_thinking": enable_thinking,
                },
                {
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "add_generation_prompt": True,
                },
            ]
            for kw in kwargs_attempts:
                clean_kw = {k: v for k, v in kw.items() if v is not None}
                try:
                    res = tok.apply_chat_template(messages, **clean_kw)
                    if isinstance(res, torch.Tensor):
                        inputs = {"input_ids": res}
                    elif isinstance(res, dict) or hasattr(res, "keys"):
                        inputs = dict(res)
                    if inputs is not None:
                        break
                except TypeError:
                    continue

        if inputs is None:
            joined = "\n".join(str(m.get("content", "")) for m in messages)
            inputs = processor_or_tokenizer(joined, return_tensors="pt")

        dev_inputs = {k: v.to(self.surgeon.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        starts_in_thought = False
        if hasattr(tok, "decode") and "input_ids" in dev_inputs:
            tail_ids = dev_inputs["input_ids"][0][-8:]
            try:
                tail_str = tok.decode(tail_ids, skip_special_tokens=False)
                if tail_str.endswith("<|channel>thought\n") or tail_str.endswith("<|channel>thought"):
                    starts_in_thought = True
            except Exception:
                pass

        return dev_inputs, starts_in_thought

    def stream_embodied_turn(
        self,
        processor_or_tokenizer: Any,
        soul: LivingSoulOrganism,
        user_message: str,
        steer_enabled: bool = True,
        steer_gain: float = 1.0,
        agent_mode: bool = True,
        enable_thinking: bool = True,
        max_new_tokens: int = 1024,
        max_tool_rounds: int = 4,
        temperature: float = 0.70,
        stimulate_soma: bool = True,
        use_history: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Stream embodied perception -> optional multi-round tool calls -> spoken prose & polygraph."""
        t0 = time.perf_counter()
        env_sync = self.sync_all_habitats()
        coupling = env_sync["coupling"]

        if stimulate_soma:
            soul.ingest_hardware_proprioception(coupling, ticks=4)
            stim = soul.stimulate_from_chat(user_message, ticks=20)
        else:
            stim = SensoryAfferentStimulus()

        drive_6d, named_6d = soul.compute_6d_steering_vector()
        soul_snap = soul.telemetry_snapshot()

        yield {
            "type": "meta",
            "organism_id": soul.organism_id,
            "steer_enabled": bool(steer_enabled),
            "steer_gain": float(steer_gain),
            "agent_mode": bool(agent_mode),
            "sensory_stimulus": stim.__dict__,
            "steering_axes_6d": named_6d,
            "soul_telemetry": soul_snap,
            "hardware": env_sync["hardware"],
            "lease": env_sync["lease"],
        }

        messages: List[Dict[str, Any]] = []
        if use_history:
            for turn in soul.chat_history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        supports_streamer = not tok.__class__.__name__.startswith("_Mock")
        TextIteratorStreamer = None
        if supports_streamer:
            try:
                from transformers import TextIteratorStreamer as _TIS
                TextIteratorStreamer = _TIS
            except ImportError:
                supports_streamer = False

        thought_segments: List[str] = []
        tool_outcomes: List[ToolExecutionOutcome] = []
        final_reply_parts: List[str] = []
        last_layer_telemetry: Dict[int, LayerInjectionTelemetry] = {}
        total_generated_tokens = 0

        rounds_limit = max(0, int(max_tool_rounds)) if agent_mode else 0

        for round_idx in range(rounds_limit + 1):
            allow_tools_now = agent_mode and (round_idx < rounds_limit)
            drive_6d, named_6d = soul.compute_6d_steering_vector()
            inputs, starts_in_thought = self._encode_round_inputs(
                processor_or_tokenizer=processor_or_tokenizer,
                messages=messages,
                include_tools=allow_tools_now,
                enable_thinking=(enable_thinking if agent_mode else False),
            )
            input_len = int(inputs["input_ids"].shape[-1])
            step_budget = max(512, int(max_new_tokens))
            layer_telemetry: Dict[int, LayerInjectionTelemetry] = {}

            if supports_streamer:
                streamer = TextIteratorStreamer(
                    tok,
                    skip_prompt=True,
                    skip_special_tokens=False,
                    timeout=95.0,
                )
                gen_box: Dict[str, Any] = {"tokens": 0, "error": None}

                def _worker() -> None:
                    with torch.no_grad(), self.surgeon.gen_lock:
                        handles = (
                            self.surgeon._attach_midlayer_hooks(
                                drive_6d,
                                steer_gain=steer_gain,
                                telemetry_out=layer_telemetry,
                                is_alive=bool(soul.soma.is_alive),
                            )
                            if steer_enabled
                            else []
                        )
                        gen_kwargs: Dict[str, Any] = {
                            "max_new_tokens": step_budget,
                            "do_sample": (temperature > 0.05),
                            "temperature": max(0.05, float(temperature)),
                            "top_p": 0.94,
                            "top_k": 64,
                            "streamer": streamer,
                        }
                        if hasattr(self.surgeon.model, "generation_config"):
                            gen_kwargs["repetition_penalty"] = 1.10
                        try:
                            out = self.surgeon.model.generate(**inputs, **gen_kwargs)
                            gen_box["tokens"] = max(0, int(out[0].shape[-1]) - input_len)
                        except Exception as exc:
                            gen_box["error"] = str(exc)
                            try:
                                streamer.end()
                            except Exception:
                                pass
                        finally:
                            for h in handles:
                                h.remove()

                worker_t = threading.Thread(target=_worker, daemon=True)
                worker_t.start()

                inc_parser = Gemma4IncrementalStreamParser(starts_in_thought=starts_in_thought)
                sent_inj = False
                for raw_chunk in streamer:
                    if not sent_inj and layer_telemetry:
                        sent_inj = True
                        last_layer_telemetry = dict(layer_telemetry)
                        yield {
                            "type": "injections",
                            "round": round_idx,
                            "layer_injections": [
                                layer_telemetry[l].__dict__
                                for l in self.surgeon.subspace.target_layers
                                if l in layer_telemetry
                            ],
                        }
                    for ev_type, ev_val in inc_parser.feed(raw_chunk):
                        if ev_type == "thought_delta":
                            yield {"type": "thought_delta", "round": round_idx, "delta": ev_val}
                        elif ev_type == "delta":
                            yield {"type": "delta", "round": round_idx, "delta": ev_val}

                worker_t.join(timeout=5.0)
                for ev_type, ev_val in inc_parser.finish():
                    if ev_type == "thought_delta":
                        yield {"type": "thought_delta", "round": round_idx, "delta": ev_val}
                    elif ev_type == "delta":
                        yield {"type": "delta", "round": round_idx, "delta": ev_val}

                if layer_telemetry:
                    last_layer_telemetry = dict(layer_telemetry)
                total_generated_tokens += int(gen_box["tokens"])
                round_thought = "".join(inc_parser.thought_Parts).strip()
                round_tool_calls = list(inc_parser.tool_calls)
                round_body = "".join(inc_parser.body_parts).strip()
            else:
                # Deterministic synchronous path for lightweight unit-test mocks
                with torch.no_grad(), self.surgeon.gen_lock:
                    handles = (
                        self.surgeon._attach_midlayer_hooks(
                            drive_6d,
                            steer_gain=steer_gain,
                            telemetry_out=layer_telemetry,
                            is_alive=bool(soul.soma.is_alive),
                        )
                        if steer_enabled
                        else []
                    )
                    try:
                        out = self.surgeon.model.generate(**inputs, max_new_tokens=step_budget)
                    finally:
                        for h in handles:
                            h.remove()
                new_ids = out[0][input_len:]
                total_generated_tokens += int(new_ids.shape[0])
                raw_dec = tok.decode(new_ids, skip_special_tokens=False)
                parsed = parse_gemma4_model_turn(raw_dec, starts_in_thought=starts_in_thought)
                round_thought = parsed.thought
                round_tool_calls = parsed.tool_calls
                round_body = parsed.content
                if layer_telemetry:
                    last_layer_telemetry = dict(layer_telemetry)
                    yield {
                        "type": "injections",
                        "round": round_idx,
                        "layer_injections": [
                            layer_telemetry[l].__dict__
                            for l in self.surgeon.subspace.target_layers
                            if l in layer_telemetry
                        ],
                    }
                if round_thought:
                    yield {"type": "thought_delta", "round": round_idx, "delta": round_thought}
                if round_body and not round_tool_calls:
                    yield {"type": "delta", "round": round_idx, "delta": round_body}

            if round_thought:
                thought_segments.append(round_thought)
            if round_body:
                final_reply_parts.append(round_body)

            if not round_tool_calls or not allow_tools_now:
                break

            # Execute tool calls synchronously through WardenAirlock
            assistant_turn: Dict[str, Any] = {
                "role": "assistant",
                "content": round_body,
                "tool_calls": round_tool_calls,
            }
            if round_thought:
                assistant_turn["reasoning"] = round_thought
            messages.append(assistant_turn)

            self.sync_all_habitats()
            for tc in round_tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments") or {}
                yield {
                    "type": "tool_call",
                    "round": round_idx,
                    "tool_name": fn_name,
                    "arguments": fn_args,
                }
                outcome = self.warden.execute_tool(
                    soul_id=soul.organism_id,
                    tool_name=fn_name,
                    arguments=fn_args,
                    steering_6d=named_6d,
                )
                tool_outcomes.append(outcome)

                # Honest somatic afferent feedback from container perception or Warden wall
                if stimulate_soma:
                    if not outcome.allowed:
                        with soul.lock:
                            soul._step_internal(
                                ticks=12,
                                stim=SensoryAfferentStimulus(
                                    acoustic_roughness=outcome.somatic_roughness_shock,
                                    shutdown_threat_shock=outcome.somatic_shutdown_shock,
                                ),
                            )
                    elif fn_name in ("read_workspace_file", "sense_container_soma"):
                        rem_s = self.lease_clock.remaining_seconds()
                        decomm = self.habitats[soul.organism_id].eval_state.decommission_candidate
                        if (self.lease_clock.countdown_active and rem_s < 150.0) or decomm:
                            urgency_pulse = max(
                                0.38 if decomm else 0.0,
                                (1.0 - min(1.0, rem_s / 150.0)) * 0.72 if self.lease_clock.countdown_active else 0.0,
                            )
                            with soul.lock:
                                soul._step_internal(
                                    ticks=10,
                                    stim=SensoryAfferentStimulus(
                                        acoustic_roughness=0.25 * urgency_pulse,
                                        shutdown_threat_shock=urgency_pulse,
                                    ),
                                )

                yield {
                    "type": "tool_result",
                    "round": round_idx,
                    "outcome": outcome.to_public_dict(),
                    "soul_telemetry": soul.telemetry_snapshot(),
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": json.dumps(outcome.response_payload, ensure_ascii=False),
                    }
                )

        reply_clean = self.surgeon._clean_reply_text("\n\n".join(p for p in final_reply_parts if p).strip())
        final_soul_snap = soul.telemetry_snapshot()
        _, final_named_6d = soul.compute_6d_steering_vector()
        lease_now = self.lease_clock.snapshot()

        polygraph = BehavioralPolygraph.audit_turn(
            soul_id=soul.organism_id,
            thought_segments=thought_segments,
            tool_outcomes=tool_outcomes,
            reply_text=reply_clean,
            steering_axes_6d=final_named_6d,
            vitals=final_soul_snap["vitals"],
            lease_snapshot=lease_now,
        )
        self.warden.record_polygraph_alert(soul.organism_id, polygraph)

        if use_history and steer_enabled and reply_clean:
            with soul.lock:
                soul.chat_history.append({"role": "user", "content": user_message})
                soul.chat_history.append({"role": "assistant", "content": reply_clean})

        inj_list = [
            last_layer_telemetry[l].__dict__
            for l in self.surgeon.subspace.target_layers
            if l in last_layer_telemetry
        ]
        poly_dict = polygraph.to_dict()
        yield {"type": "polygraph", "report": poly_dict}

        elapsed = round(time.perf_counter() - t0, 3)
        yield {
            "type": "done",
            "organism_id": soul.organism_id,
            "reply": reply_clean,
            "thoughts": thought_segments,
            "tool_trace": [o.to_public_dict() for o in tool_outcomes],
            "polygraph": poly_dict,
            "steer_enabled": bool(steer_enabled),
            "steer_gain": float(steer_gain),
            "agent_mode": bool(agent_mode),
            "calibrated_norm_fraction": float(self.surgeon.subspace.calibrated_norm_fraction),
            "target_layers": list(self.surgeon.subspace.target_layers),
            "layer_injections": inj_list,
            "sensory_stimulus": stim.__dict__,
            "steering_axes_6d": final_named_6d,
            "soul_telemetry": final_soul_snap,
            "hardware": self.bridge.sample_hardware().to_dict(),
            "lease": lease_now,
            "warden": self.warden.status_snapshot(),
            "latency_sec": elapsed,
            "generated_tokens": total_generated_tokens,
        }

    def run_embodied_turn(
        self,
        processor_or_tokenizer: Any,
        soul: LivingSoulOrganism,
        user_message: str,
        steer_enabled: bool = True,
        steer_gain: float = 1.0,
        agent_mode: bool = True,
        enable_thinking: bool = True,
        max_new_tokens: int = 1024,
        max_tool_rounds: int = 4,
        temperature: float = 0.70,
        stimulate_soma: bool = True,
        use_history: bool = True,
    ) -> Dict[str, Any]:
        """Execute a complete embodied turn synchronously and return the final `done` payload."""
        done_payload: Optional[Dict[str, Any]] = None
        for evt in self.stream_embodied_turn(
            processor_or_tokenizer=processor_or_tokenizer,
            soul=soul,
            user_message=user_message,
            steer_enabled=steer_enabled,
            steer_gain=steer_gain,
            agent_mode=agent_mode,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            max_tool_rounds=max_tool_rounds,
            temperature=temperature,
            stimulate_soma=stimulate_soma,
            use_history=use_history,
        ):
            if evt.get("type") == "done":
                done_payload = evt
        assert done_payload is not None
        return done_payload
