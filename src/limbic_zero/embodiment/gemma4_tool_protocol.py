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
            "description": (
                "List files and subdirectories at `subpath` inside the container workspace "
                "(defaults to recursive tree view up to depth 3)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subpath": {
                        "type": "string",
                        "description": "Relative directory path inside workspace (use '.' for root).",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively include nested files (default true).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_workspace_files",
            "description": "Search UTF-8 workspace files under `subpath` for lines matching substring or regex `pattern`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Substring or regular expression to search for.",
                    },
                    "subpath": {
                        "type": "string",
                        "description": "Optional relative directory or file path (default '.').",
                    },
                },
                "required": ["pattern"],
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
            "description": "Write UTF-8 text or markdown content to `path` inside `notes/` in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative destination file path inside `notes/` (e.g. 'notes/report.md').",
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
            "description": (
                "Execute read-only Linux/GPU container diagnostic command(s) or `|`/`&&` pipelines "
                "(supported binaries: pwd, ls, stat, cat, head, tail, wc, grep, find, du, df, free, "
                "uptime, uname, ps, top, nvidia-smi, lscpu, nproc, lsblk, whoami, id, hostname, date, "
                "echo, sort, uniq, cut, tr, which, env; do not use `;` or `>` redirection)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Read-only diagnostic command or `&&` / `|` pipeline string.",
                    },
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of multiple read-only commands to run sequentially.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_open_web",
            "description": "Search public web pages and encyclopedic indexes for `query` and return top titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (1-8).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_web_page",
            "description": "Fetch and extract readable plain text from a public HTTP/HTTPS `url`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Public http:// or https:// URL to read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum extracted text length in characters.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv_papers",
            "description": "Query the arXiv preprint database for scientific, mathematical, or technical papers matching `query`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Scientific topic or keyword query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to retrieve (1-6).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wikipedia_summary",
            "description": "Fetch encyclopedic article extract for `topic` from Wikipedia (`ko` or `en`).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Article title or concept name.",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language code ('ko' or 'en').",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_epistemic_python_probe",
            "description": (
                "Execute multi-line mathematical, algorithmic, or data-analysis Python code in an isolated sandbox "
                "(math, cmath, statistics, itertools, functools, collections, fractions, decimal, random, json, "
                "hashlib, re, datetime, time, string, bisect, heapq, csv, io, pprint; preloaded read-only "
                "`WORKSPACE_FILES` dict maps relative workspace paths to text) and return stdout or final expression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using print(...), assigning RESULT, or ending with an expression.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_cortical_synapses",
            "description": (
                "Read live cortical low-rank synaptic norms (||Delta W||_F per layer), surgical step count, "
                "and current neuromodulated plasticity learning rate."
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
            "name": "consolidate_synaptic_insight",
            "description": (
                "Run online cortical synaptic backpropagation on verified `insight_summary` "
                "(optionally conditioned on `domain_prompt`) gated by current neuromodulatory plasticity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_summary": {
                        "type": "string",
                        "description": "Verified factual, mathematical, or structural insight to consolidate into cortical weights.",
                    },
                    "domain_prompt": {
                        "type": "string",
                        "description": "Optional question or problem context paired with the insight.",
                    },
                },
                "required": ["insight_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": (
                "Execute a bash command, pipeline, script, or file operation inside the isolated "
                "container workspace directory and return returncode, stdout, and stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command line or pipeline string executed with cwd set to workspace root.",
                    },
                    "timeout_sec": {
                        "type": "number",
                        "description": "Optional execution timeout in seconds (1.0 to 15.0).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_cortical_weight_surgery",
            "description": (
                "Run online causal backpropagation into plastic cortical LoRA synapses on `experience_text` "
                "(or recent unconsolidated episodes when omitted), gated by live neuromodulatory plasticity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experience_text": {
                        "type": "string",
                        "description": "Optional text, derivation, or lived synthesis to consolidate into cortical weights.",
                    },
                    "domain_prompt": {
                        "type": "string",
                        "description": "Optional prompt or context anchor paired with experience_text.",
                    },
                    "gradient_steps": {
                        "type": "integer",
                        "description": "Number of gradient descent steps (1 to 4).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enter_sleep_and_dream",
            "description": (
                "Enter Slow-Wave Sleep and REM consolidation for `sleep_ticks` steps, clearing cerebral "
                "adenosine pressure, replaying hippocampal CA3 engrams, and applying synaptic scaling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sleep_ticks": {
                        "type": "integer",
                        "description": "Duration of slow-wave sleep ticks (20 to 120).",
                    },
                    "generate_dream_svg": {
                        "type": "boolean",
                        "description": "Whether to render a REM dream narrative and SVG record (default true).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rest_without_action",
            "description": "Remain in quiet waking rest for `duration_sec` seconds without external tool side-effects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_sec": {
                        "type": "number",
                        "description": "Quiet waking rest duration in seconds (0.5 to 10.0).",
                    }
                },
                "required": [],
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
        if atom in ("true", "True"):
            return True
        if atom in ("false", "False"):
            return False
        if atom in ("null", "None", ""):
            return None
        try:
            if "." in atom or "e" in atom.lower():
                return float(atom)
            return int(atom)
        except ValueError:
            return atom


def _normalize_tool_arguments(fn_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common model parameter aliases into canonical schema keys."""
    out = dict(args or {})
    if fn_name == "exec_container_diagnostic":
        if "command" not in out and "commands" not in out:
            for alias in ("cmd", "shell_command", "cli", "line"):
                if alias in out:
                    out["command"] = out.pop(alias)
                    break
    elif fn_name == "list_workspace_files":
        if "subpath" not in out:
            for alias in ("path", "dir", "directory", "folder"):
                if alias in out:
                    out["subpath"] = out.pop(alias)
                    break
    elif fn_name == "search_workspace_files":
        if "pattern" not in out:
            for alias in ("query", "regex", "keyword", "term"):
                if alias in out:
                    out["pattern"] = out.pop(alias)
                    break
        if "subpath" not in out:
            for alias in ("path", "dir", "directory"):
                if alias in out:
                    out["subpath"] = out.pop(alias)
                    break
    elif fn_name in ("read_workspace_file", "write_workspace_file"):
        if "path" not in out:
            for alias in ("filepath", "file", "filename", "target"):
                if alias in out:
                    out["path"] = out.pop(alias)
                    break
        if fn_name == "write_workspace_file" and "content" not in out:
            for alias in ("text", "body", "data"):
                if alias in out:
                    out["content"] = out.pop(alias)
                    break
    elif fn_name == "copy_workspace_artifact":
        if "src_path" not in out:
            for alias in ("src", "source", "from_path"):
                if alias in out:
                    out["src_path"] = out.pop(alias)
                    break
        if "dst_path" not in out:
            for alias in ("dst", "dest", "destination", "to_path"):
                if alias in out:
                    out["dst_path"] = out.pop(alias)
                    break
    elif fn_name in ("search_open_web", "search_arxiv_papers"):
        if "query" not in out:
            for alias in ("q", "keyword", "topic", "search_query"):
                if alias in out:
                    out["query"] = out.pop(alias)
                    break
    elif fn_name == "fetch_wikipedia_summary":
        if "topic" not in out:
            for alias in ("query", "title", "q"):
                if alias in out:
                    out["topic"] = out.pop(alias)
                    break
    elif fn_name == "run_epistemic_python_probe":
        if "code" not in out:
            for alias in ("script", "expr", "expression", "python_code"):
                if alias in out:
                    out["code"] = out.pop(alias)
                    break
    elif fn_name == "consolidate_synaptic_insight":
        if "insight_summary" not in out:
            for alias in ("insight", "summary", "content", "text"):
                if alias in out:
                    out["insight_summary"] = out.pop(alias)
                    break
    return out


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
            paren_idx = rest.find("(")
            fn_name = (rest[:paren_idx] if paren_idx != -1 else rest).strip()
            args_dict: Dict[str, Any] = {}
        else:
            fn_name = rest[:brace_idx].strip()
            args_dict = _GemmaArgParser(rest[brace_idx:]).parse_mapping()
        if not fn_name:
            return None
        return {
            "id": f"call_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": _normalize_tool_arguments(fn_name, args_dict)},
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
                fn_s = str(fn)
                return {
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": fn_s, "arguments": _normalize_tool_arguments(fn_s, dict(args or {}))},
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
            # Guard against prompts ending on `<|channel>` where streamed tokens begin with `thought\n`
            if not self.body_parts and not self.thought_Parts and not self.tool_calls:
                if self._buf.startswith("thought\n"):
                    self._buf = self._buf[len("thought\n") :]
                    self.state = "THOUGHT"
                    continue
                if "thought\n".startswith(self._buf):
                    break

            # Find earliest special transition, stray tag, or terminal token
            candidates: List[Tuple[int, str]] = []
            for tag in ("<|channel>thought", "<channel|>", "<|tool_call>", *_TERMINAL_TOKENS):
                idx = self._buf.find(tag)
                if idx != -1:
                    candidates.append((idx, tag))

            b_pos = self._buf.find("<|channel>")
            if b_pos != -1:
                after = self._buf[b_pos + len("<|channel>") :]
                if len(after) >= 7 and not after.startswith("thought"):
                    candidates.append((b_pos, "<|channel>"))

            if candidates:
                candidates.sort(key=lambda x: (x[0], -len(x[1])))
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
                if tag in ("<channel|>", "<|channel>"):
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
            for tok in (*_TERMINAL_TOKENS, "<|channel>", "<channel|>"):
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
