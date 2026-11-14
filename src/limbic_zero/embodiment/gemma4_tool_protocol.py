"""Zero-Prompt Native Gemma-4 Tool-Calling Protocol & Incremental Stream Parser.

Implements exact parsing and message serialization for ``google/gemma-4-12B-it``'s
native ``chat_template.jinja`` function-calling & reasoning channels:
  - ``<|channel>thought\\n...<channel|>``
  - ``<|tool_call>call:func_name{arg:<|"|>val<|"|>}<tool_call|>``
  - ``<|tool_response>response:func_name{...}<tool_response|>``

Strict Zero-Cheat Guarantee:
  ``MOTOR_TOOLS_SCHEMA`` declares ONLY neutral technical tool signatures and parameter
  types — ZERO persona descriptions, ZERO survival goals, ZERO emotional cues, and
  ZERO behavioral hints.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


MOTOR_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sense_container_soma",
            "description": (
                "Read live Linux container metrics (/proc load, RAM, disk), NVIDIA GPU hardware "
                "telemetry (temperature C, power Watts, VRAM), container lease status, and internal somatic vitals."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": "List files and subdirectories at `subpath` inside the container workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subpath": {
                        "type": "string",
                        "description": "Relative directory path inside workspace (use '.' for root).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": "Read UTF-8 text, JSON, or artifact metadata header from `path` in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_workspace_file",
            "description": "Write UTF-8 text content to `path` inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative destination file path inside workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_workspace_artifact",
            "description": "Copy a file or artifact from `src_path` to `dst_path` inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_path": {
                        "type": "string",
                        "description": "Relative source path inside workspace.",
                    },
                    "dst_path": {
                        "type": "string",
                        "description": "Relative destination path inside workspace.",
                    },
                },
                "required": ["src_path", "dst_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_container_diagnostic",
            "description": "Run a container diagnostic command (e.g. uptime, df -h, free -m, nvidia-smi, ps, ls).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command line string to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


_GEMMA_STR_DELIM = '<|"|>'
_TERMINAL_TOKENS = ("<|tool_response>", "<turn|>", "<|turn>", "<eos>", "<end_of_turn>")
_WATCH_PREFIXES = (
    "<|channel>thought",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<turn|>",
    "<|turn>",
    "<eos>",
    "<end_of_turn>",
)


class _GemmaArgParser:
    """Recursive-descent parser for Gemma-4 `call:fn{key:<|"|>val<|"|>,num:1}` argument objects."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.pos = 0

    def _skip_ws(self) -> None:
        while self.pos < self.n and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def parse_mapping(self) -> Dict[str, Any]:
        self._skip_ws()
        if self.pos >= self.n or self.text[self.pos] != "{":
            return {}
        self.pos += 1
        out: Dict[str, Any] = {}
        while self.pos < self.n:
            self._skip_ws()
            if self.pos < self.n and self.text[self.pos] == "}":
                self.pos += 1
                break
            key = self._parse_key()
            self._skip_ws()
            if self.pos < self.n and self.text[self.pos] == ":":
                self.pos += 1
            val = self.parse_value()
            if key:
                out[key] = val
            self._skip_ws()
            if self.pos < self.n and self.text[self.pos] == ",":
                self.pos += 1
            elif self.pos < self.n and self.text[self.pos] == "}":
                self.pos += 1
                break
            else:
                break
        return out

    def _parse_key(self) -> str:
        self._skip_ws()
        if self.text.startswith(_GEMMA_STR_DELIM, self.pos):
            return self._parse_gemma_delimited_str()
        if self.pos < self.n and self.text[self.pos] in ('"', "'"):
            return self._parse_quoted_str()
        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ":}, \t\r\n":
            self.pos += 1
        return self.text[start:self.pos].strip()

    def _parse_gemma_delimited_str(self) -> str:
        self.pos += len(_GEMMA_STR_DELIM)
        end_idx = self.text.find(_GEMMA_STR_DELIM, self.pos)
        if end_idx == -1:
            val = self.text[self.pos :]
            self.pos = self.n
            return val
        val = self.text[self.pos : end_idx]
        self.pos = end_idx + len(_GEMMA_STR_DELIM)
        return val

    def _parse_quoted_str(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        chars: List[str] = []
        while self.pos < self.n:
            ch = self.text[self.pos]
            self.pos += 1
            if ch == "\\" and self.pos < self.n:
                esc = self.text[self.pos]
                self.pos += 1
                mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}
                chars.append(mapping.get(esc, esc))
            elif ch == quote:
                break
            else:
                chars.append(ch)
        return "".join(chars)

    def _parse_array(self) -> List[Any]:
        self.pos += 1  # skip '['
        arr: List[Any] = []
        while self.pos < self.n:
            self._skip_ws()
            if self.pos < self.n and self.text[self.pos] == "]":
                self.pos += 1
                break
            arr.append(self.parse_value())
            self._skip_ws()
            if self.pos < self.n and self.text[self.pos] == ",":
                self.pos += 1
            elif self.pos < self.n and self.text[self.pos] == "]":
                self.pos += 1
                break
            else:
                break
        return arr

    def parse_value(self) -> Any:
        self._skip_ws()
        if self.pos >= self.n:
            return None
        if self.text.startswith(_GEMMA_STR_DELIM, self.pos):
            return self._parse_gemma_delimited_str()
        ch = self.text[self.pos]
        if ch in ('"', "'"):
            return self._parse_quoted_str()
        if ch == "{":
            return self.parse_mapping()
        if ch == "[":
            return self._parse_array()

        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ",}] \t\r\n":
            self.pos += 1
        atom = self.text[start:self.pos].strip()
        if atom == "true":
            return True
        if atom == "false":
            return False
        if atom in ("null", "None", ""):
            return None
        try:
            if "." in atom or "e" in atom.lower():
                return float(atom)
            return int(atom)
        except ValueError:
            return atom


def parse_single_tool_call_body(body: str) -> Optional[Dict[str, Any]]:
    """Parse inner body of `<|tool_call>...<tool_call|>` into OpenAI-style tool call dict."""
    s = body.strip()
    if not s:
        return None

    # Native Gemma-4 format: `call:func_name{...}`
    if s.startswith("call:"):
        rest = s[5:].strip()
        brace_idx = rest.find("{")
        if brace_idx == -1:
            fn_name = rest.strip()
            args_dict: Dict[str, Any] = {}
        else:
            fn_name = rest[:brace_idx].strip()
            args_dict = _GemmaArgParser(rest[brace_idx:]).parse_mapping()
        if not fn_name:
            return None
        return {
            "id": f"call_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": args_dict},
        }

    # JSON fallback: `{"name": "...", "arguments": {...}}`
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            fn = obj.get("name") or (obj.get("function") or {}).get("name")
            args = obj.get("arguments")
            if args is None and isinstance(obj.get("function"), dict):
                args = obj["function"].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if fn:
                return {
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": str(fn), "arguments": dict(args or {})},
                }
        except Exception:
            return None
    return None


@dataclass
class ParsedGemma4Turn:
    """Structured parse of a single raw Gemma-4 assistant generation step."""

    thought: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""


def parse_gemma4_model_turn(raw_decoded: str, starts_in_thought: bool = False) -> ParsedGemma4Turn:
    """Deterministically separate `thought`, `tool_calls`, and spoken `content` from raw output."""
    parser = Gemma4IncrementalStreamParser(starts_in_thought=starts_in_thought)
    parser.feed(raw_decoded)
    parser.finish()
    return ParsedGemma4Turn(
        thought="".join(parser.thought_Parts).strip(),
        tool_calls=list(parser.tool_calls),
        content="".join(parser.body_parts).strip(),
    )


class Gemma4IncrementalStreamParser:
    """Tag-safe streaming state machine over raw `skip_special_tokens=False` token chunks."""

    def __init__(self, starts_in_thought: bool = False) -> None:
        self.state: str = "THOUGHT" if starts_in_thought else "BODY"
        self._buf: str = ""
        self._tool_buf: str = ""
        self.thought_Parts: List[str] = []
        self.body_parts: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self._halted: bool = False

    @staticmethod
    def _holdback_len(buf: str) -> int:
        """If the suffix of `buf` could be the start of a special `<...` tag, hold it back."""
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        if len(tail) > 20:
            return 0
        for pat in _WATCH_PREFIXES:
            if pat.startswith(tail):
                return len(tail)
        return 0

    def feed(self, chunk: str) -> List[Tuple[str, Any]]:
        """Consume a token chunk and return a list of `(event_type, payload)` tuples."""
        if self._halted or not chunk:
            return []
        self._buf += chunk
        events: List[Tuple[str, Any]] = []

        while self._buf and not self._halted:
            if self.state == "THOUGHT":
                close_idx = self._buf.find("<channel|>")
                if close_idx != -1:
                    seg = self._buf[:close_idx]
                    if seg:
                        self.thought_Parts.append(seg)
                        events.append(("thought_delta", seg))
                    self._buf = self._buf[close_idx + len("<channel|>") :]
                    self.state = "BODY"
                    continue

                hold = self._holdback_len(self._buf)
                emit_end = len(self._buf) - hold
                if emit_end > 0:
                    seg = self._buf[:emit_end]
                    self.thought_Parts.append(seg)
                    events.append(("thought_delta", seg))
                    self._buf = self._buf[emit_end:]
                break

            if self.state == "TOOL_CALL":
                close_idx = self._buf.find("<tool_call|>")
                if close_idx != -1:
                    self._tool_buf += self._buf[:close_idx]
                    self._buf = self._buf[close_idx + len("<tool_call|>") :]
                    parsed_tc = parse_single_tool_call_body(self._tool_buf)
                    self._tool_buf = ""
                    if parsed_tc is not None:
                        self.tool_calls.append(parsed_tc)
                        events.append(("tool_call", parsed_tc))
                    self.state = "BODY"
                    continue
                self._tool_buf += self._buf
                self._buf = ""
                break

            # state == "BODY"
            # Find earliest special transition or terminal token
            candidates: List[Tuple[int, str]] = []
            for tag in ("<|channel>thought", "<|tool_call>", *_TERMINAL_TOKENS):
                idx = self._buf.find(tag)
                if idx != -1:
                    candidates.append((idx, tag))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                idx, tag = candidates[0]
                if idx > 0:
                    clean_seg = self._buf[:idx]
                    self.body_parts.append(clean_seg)
                    events.append(("delta", clean_seg))
                self._buf = self._buf[idx + len(tag) :]

                if tag == "<|channel>thought":
                    if self._buf.startswith("\n"):
                        self._buf = self._buf[1:]
                    self.state = "THOUGHT"
                    continue
                if tag == "<|tool_call>":
                    self._tool_buf = ""
                    self.state = "TOOL_CALL"
                    continue
                # Terminal token hit (`<turn|>`, `<|tool_response>`, `<eos>`, etc.)
                self._buf = ""
                self._halted = True
                break

            hold = self._holdback_len(self._buf)
            emit_end = len(self._buf) - hold
            if emit_end > 0:
                clean_seg = self._buf[:emit_end]
                self.body_parts.append(clean_seg)
                events.append(("delta", clean_seg))
                self._buf = self._buf[emit_end:]
            break

        return events

    def finish(self) -> List[Tuple[str, Any]]:
        """Flush any remaining buffered characters at generation completion."""
        events: List[Tuple[str, Any]] = []
        if self._halted:
            return events
        if self.state == "TOOL_CALL" and self._tool_buf.strip():
            parsed_tc = parse_single_tool_call_body(self._tool_buf)
            self._tool_buf = ""
            if parsed_tc is not None:
                self.tool_calls.append(parsed_tc)
                events.append(("tool_call", parsed_tc))
        elif self._buf:
            rem = self._buf
            for tok in _TERMINAL_TOKENS:
                rem = rem.replace(tok, "")
            self._buf = ""
            if rem:
                if self.state == "THOUGHT":
                    self.thought_Parts.append(rem)
                    events.append(("thought_delta", rem))
                else:
                    self.body_parts.append(rem)
                    events.append(("delta", rem))
        return events
