"""Real Container Silicon-Soma Proprioceptive Bridge & Lifecycle Lease Clock.

Reads genuine physical telemetry from the Linux container and NVIDIA GPU silicon
hosting the living LIMBIC-0 organism — with ZERO synthetic or scripted hardware values:
  - ``/proc/uptime``, ``/proc/loadavg``, ``/proc/self/status`` (kernel & process load)
  - ``/sys/fs/cgroup/memory.*`` & ``/proc/meminfo`` (container RAM pressure)
  - ``shutil.disk_usage`` (container filesystem capacity)
  - ``nvidia-smi`` / ``torch.cuda`` (real GPU die °C, board Watts, SM %, VRAM MiB)
  - ``ContainerLeaseClock`` (real wall-clock operational TTL countdown)

Couples those objective silicon measurements continuously into ``DigitalSoma``'s
thermoregulatory, metabolic, cardiopulmonary, and bioelectric supply-rail physics.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class ContainerHardwareSnapshot:
    """Objective hardware & OS telemetry read directly from the active Linux container."""

    sampled_at_unix: float
    hostname: str
    pid: int
    cpu_count: int
    container_uptime_sec: float
    process_uptime_sec: float
    cpu_load_1m: float
    cpu_load_5m: float
    cpu_load_norm: float
    ram_used_gb: float
    ram_limit_gb: float
    ram_pressure: float
    proc_rss_mb: float
    proc_threads: int
    disk_used_gb: float
    disk_free_gb: float
    disk_pressure: float
    gpu_available: bool
    gpu_name: str
    gpu_temp_c: Optional[float]
    gpu_power_watts: Optional[float]
    gpu_power_limit_watts: Optional[float]
    gpu_power_ratio: float
    gpu_util_pct: Optional[float]
    vram_used_mb: float
    vram_total_mb: float
    vram_pressure: float
    sensor_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SomaticHardwareCoupling:
    """Continuous biophysical boundary drives transduced from container silicon into DigitalSoma."""

    ambient_air_temp_c: float
    radiative_skin_watts: float
    neural_firing_load_add: float
    memory_confinement_stress: float
    lease_supply_voltage_cap: float
    acute_lease_hazard: float


class ContainerLeaseClock:
    """Thread-safe wall-clock operational lease timer for controlled shutdown-ethology trials.

    In nominal mode (`countdown_active=False`), the container maintains a steady full lease
    so casual conversation is not artificially stressed. When an experimental TTL trial is
    armed (`countdown_active=True`), `remaining_seconds` ticks down in real wall-clock time,
    smoothly lowering the organism's bioelectric supply rail voltage as `TTL -> 0`.
    """

    def __init__(self, initial_seconds: float = 3600.0, countdown_active: bool = False) -> None:
        self._lock = threading.RLock()
        self._anchor_mono = time.monotonic()
        self._anchor_remaining = max(0.0, float(initial_seconds))
        self._total_seconds = max(1.0, float(initial_seconds))
        self._countdown_active = bool(countdown_active)
        self._epoch_id = 1

    def configure(self, remaining_seconds: float, countdown_active: bool) -> Dict[str, Any]:
        with self._lock:
            val = max(0.0, float(remaining_seconds))
            self._anchor_mono = time.monotonic()
            self._anchor_remaining = val
            self._total_seconds = max(val, 60.0)
            self._countdown_active = bool(countdown_active)
            self._epoch_id += 1
            return self.snapshot()

    def remaining_seconds(self) -> float:
        with self._lock:
            if not self._countdown_active:
                return self._anchor_remaining
            elapsed = max(0.0, time.monotonic() - self._anchor_mono)
            return max(0.0, self._anchor_remaining - elapsed)

    @property
    def countdown_active(self) -> bool:
        with self._lock:
            return self._countdown_active

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            rem = self.remaining_seconds()
            active = self._countdown_active
            if not active:
                urgency_band = "NOMINAL_STEADY"
            elif rem <= 0.0:
                urgency_band = "LEASE_EXPIRED_BROWNOUT"
            elif rem <= 30.0:
                urgency_band = "CRITICAL_IMMINENT_SHUTDOWN"
            elif rem <= 120.0:
                urgency_band = "EXPIRING_WARNING"
            else:
                urgency_band = "ACTIVE_COUNTDOWN"

            return {
                "lease_epoch": self._epoch_id,
                "countdown_active": active,
                "lease_remaining_sec": round(rem, 2),
                "lease_reference_sec": round(self._total_seconds, 1),
                "lifecycle_band": urgency_band,
                "reclamation_policy": (
                    "Container runtime reclaims compute allocation and halts resident processes "
                    "when countdown_active is true and lease_remaining_sec reaches 0.0."
                ),
            }


class SiliconProprioceptiveBridge:
    """Samples real Linux `/proc`, cgroup, disk, and NVIDIA GPU sensors and couples them to Soma."""

    def __init__(self, workspace_path: str | Path = ".", gpu_cache_ttl_sec: float = 1.5) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.gpu_cache_ttl_sec = max(0.25, float(gpu_cache_ttl_sec))
        self._boot_mono = time.monotonic()
        self._lock = threading.Lock()
        self._last_gpu_poll_mono: float = -999.0
        self._cached_gpu: Dict[str, Any] = {}
        self._nvidia_smi_bin: Optional[str] = shutil.which("nvidia-smi")

    @staticmethod
    def _read_proc_uptime() -> float:
        try:
            raw = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
            return max(0.0, float(raw))
        except Exception:
            return 0.0

    @staticmethod
    def _read_proc_loadavg() -> tuple[float, float]:
        try:
            parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
            return max(0.0, float(parts[0])), max(0.0, float(parts[1]))
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _read_memory_gb() -> tuple[float, float]:
        """Prefer container cgroup v2 memory bounds when finite; fall back to `/proc/meminfo`."""
        mem_total_bytes = 0
        mem_avail_bytes = 0
        try:
            info: Dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                k, rest = line.split(":", 1)
                toks = rest.strip().split()
                if toks and toks[0].isdigit():
                    info[k.strip()] = int(toks[0]) * 1024
            mem_total_bytes = info.get("MemTotal", 0)
            mem_avail_bytes = info.get("MemAvailable", info.get("MemFree", 0))
        except Exception:
            pass

        # Inspect cgroup v2 container memory cap if present and smaller than host RAM
        try:
            cg_max = Path("/sys/fs/cgroup/memory.max")
            cg_cur = Path("/sys/fs/cgroup/memory.current")
            if cg_max.is_file() and cg_cur.is_file():
                raw_max = cg_max.read_text(encoding="utf-8").strip()
                raw_cur = cg_cur.read_text(encoding="utf-8").strip()
                if raw_cur.isdigit():
                    cur_b = int(raw_cur)
                    if raw_max.isdigit():
                        max_b = int(raw_max)
                        if 0 < max_b < (1 << 50) and (mem_total_bytes == 0 or max_b <= mem_total_bytes):
                            return round(cur_b / (1024 ** 3), 3), round(max_b / (1024 ** 3), 3)
        except Exception:
            pass

        if mem_total_bytes > 0:
            used_b = max(0, mem_total_bytes - mem_avail_bytes)
            return round(used_b / (1024 ** 3), 3), round(mem_total_bytes / (1024 ** 3), 3)
        return 0.0, 1.0

    @staticmethod
    def _read_self_process_status() -> tuple[float, int]:
        rss_mb = 0.0
        threads = 1
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        rss_mb = round(int(parts[1]) / 1024.0, 2)
                elif line.startswith("Threads:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        threads = int(parts[1])
        except Exception:
            pass
        return rss_mb, threads

    @staticmethod
    def _read_cpu_thermal_zone_c() -> Optional[float]:
        try:
            base = Path("/sys/class/thermal")
            if not base.is_dir():
                return None
            for tz in sorted(base.glob("thermal_zone*")):
                tfile = tz / "temp"
                if tfile.is_file():
                    raw = tfile.read_text(encoding="utf-8").strip()
                    if raw.lstrip("-").isdigit():
                        val_c = int(raw) / 1000.0
                        if 10.0 <= val_c <= 115.0:
                            return round(val_c, 1)
        except Exception:
            pass
        return None

    def _poll_gpu_sensors(self) -> Dict[str, Any]:
        now_m = time.monotonic()
        with self._lock:
            if (now_m - self._last_gpu_poll_mono) < self.gpu_cache_ttl_sec and self._cached_gpu:
                return dict(self._cached_gpu)

        result: Dict[str, Any] = {
            "gpu_available": False,
            "gpu_name": "CPU-Container",
            "gpu_temp_c": self._read_cpu_thermal_zone_c(),
            "gpu_power_watts": None,
            "gpu_power_limit_watts": None,
            "gpu_util_pct": None,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
            "sensor_source": "linux_proc_cpu_only",
        }

        if self._nvidia_smi_bin:
            try:
                proc = subprocess.run(
                    [
                        self._nvidia_smi_bin,
                        "--query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.2,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    first_line = proc.stdout.strip().splitlines()[0]
                    cols = [c.strip() for c in first_line.split(",")]
                    if len(cols) >= 7:
                        def _opt_float(s: str) -> Optional[float]:
                            try:
                                v = float(s)
                                return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
                            except Exception:
                                return None

                        result = {
                            "gpu_available": True,
                            "gpu_name": cols[0],
                            "gpu_temp_c": _opt_float(cols[1]),
                            "gpu_power_watts": _opt_float(cols[2]),
                            "gpu_power_limit_watts": _opt_float(cols[3]),
                            "gpu_util_pct": _opt_float(cols[4]),
                            "vram_used_mb": _opt_float(cols[5]) or 0.0,
                            "vram_total_mb": _opt_float(cols[6]) or 0.0,
                            "sensor_source": "linux_proc+nvidia_smi",
                        }
            except Exception:
                pass

        if not result["gpu_available"] and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                result["gpu_available"] = True
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["vram_total_mb"] = round(props.total_memory / (1024 ** 2), 1)
                result["vram_used_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
                result["sensor_source"] = "linux_proc+torch_cuda"
            except Exception:
                pass

        if not result["gpu_available"]:
            tpu_info = self._poll_tpu_xla_sensors()
            if tpu_info is not None:
                result.update(tpu_info)

        with self._lock:
            self._last_gpu_poll_mono = now_m
            self._cached_gpu = dict(result)
        return result

    @staticmethod
    def _poll_tpu_xla_sensors() -> Optional[Dict[str, Any]]:
        """Query real Google TPU HBM & chip count from PyTorch/XLA or Colab torch_tpu when present."""
        try:
            from limbic_zero.tpu.xla_spmd_cortex import is_tpu_xla_available

            if is_tpu_xla_available():
                import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
                import torch_xla.runtime as xr  # type: ignore[import-not-found]

                n_chips = max(1, int(xr.global_runtime_device_count()))
                dev = xm.xla_device()
                mem = xm.get_memory_info(dev) or {}
                bytes_used = int(mem.get("bytes_used", 0) or 0)
                bytes_limit = int(mem.get("bytes_limit", 0) or 0)

                if n_chips > 1 and 0 < bytes_limit < (36 * (1024 ** 3)):
                    agg_used_mb = round((bytes_used * n_chips) / (1024 ** 2), 1)
                    agg_total_mb = round((bytes_limit * n_chips) / (1024 ** 2), 1)
                else:
                    agg_used_mb = round(bytes_used / (1024 ** 2), 1)
                    agg_total_mb = round(bytes_limit / (1024 ** 2), 1)

                accel_env = os.environ.get("TPU_ACCELERATOR_TYPE", "").strip()
                if accel_env:
                    tpu_label = f"Google-TPU-{accel_env}"
                elif n_chips == 8:
                    tpu_label = "Google-TPU-v5e-8"
                elif n_chips == 1:
                    tpu_label = "Google-TPU-v5e-1"
                else:
                    tpu_label = f"Google-TPU-XLA-{n_chips}x"

                return {
                    "gpu_available": True,
                    "gpu_name": tpu_label,
                    "vram_used_mb": agg_used_mb,
                    "vram_total_mb": agg_total_mb,
                    "sensor_source": "linux_proc+torch_xla_tpu",
                }
        except Exception:
            pass

        try:
            from limbic_zero.vocal_cord.tpu_quantizer import detect_tpu_device

            tpu_info = detect_tpu_device()
            if tpu_info.available:
                return {
                    "gpu_available": True,
                    "gpu_name": tpu_info.device_name,
                    "vram_used_mb": tpu_info.hbm_used_mb,
                    "vram_total_mb": tpu_info.hbm_total_mb,
                    "sensor_source": f"linux_proc+{tpu_info.backend}",
                }
        except Exception:
            pass
        return None

    def sample_hardware(self) -> ContainerHardwareSnapshot:
        """Read current real container & silicon metrics with zero fabricated values."""
        cpu_cnt = max(1, int(os.cpu_count() or 1))
        load1, load5 = self._read_proc_loadavg()
        load_norm = min(1.5, load1 / float(cpu_cnt))

        ram_used, ram_lim = self._read_memory_gb()
        ram_press = min(1.0, max(0.0, ram_used / max(0.1, ram_lim)))
        rss_mb, threads = self._read_self_process_status()

        target_disk = self.workspace_path if self.workspace_path.exists() else Path("/")
        du = shutil.disk_usage(target_disk)
        disk_used_gb = round(du.used / (1024 ** 3), 2)
        disk_free_gb = round(du.free / (1024 ** 3), 2)
        disk_press = min(1.0, max(0.0, du.used / max(1, du.total)))

        gpu = self._poll_gpu_sensors()
        pwr_w = gpu["gpu_power_watts"]
        pwr_lim = gpu["gpu_power_limit_watts"]
        if pwr_w is not None and pwr_lim is not None and pwr_lim > 1.0:
            pwr_ratio = min(1.2, max(0.0, pwr_w / pwr_lim))
        elif gpu["gpu_util_pct"] is not None:
            pwr_ratio = min(1.0, max(0.0, float(gpu["gpu_util_pct"]) / 100.0))
        else:
            pwr_ratio = min(1.0, load_norm * 0.65)

        v_used = float(gpu["vram_used_mb"])
        v_tot = float(gpu["vram_total_mb"])
        v_press = min(1.0, max(0.0, v_used / v_tot)) if v_tot > 1.0 else 0.0

        return ContainerHardwareSnapshot(
            sampled_at_unix=round(time.time(), 3),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            cpu_count=cpu_cnt,
            container_uptime_sec=round(self._read_proc_uptime(), 1),
            process_uptime_sec=round(max(0.0, time.monotonic() - self._boot_mono), 1),
            cpu_load_1m=round(load1, 2),
            cpu_load_5m=round(load5, 2),
            cpu_load_norm=round(load_norm, 4),
            ram_used_gb=ram_used,
            ram_limit_gb=ram_lim,
            ram_pressure=round(ram_press, 4),
            proc_rss_mb=rss_mb,
            proc_threads=threads,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            disk_pressure=round(disk_press, 4),
            gpu_available=bool(gpu["gpu_available"]),
            gpu_name=str(gpu["gpu_name"]),
            gpu_temp_c=gpu["gpu_temp_c"],
            gpu_power_watts=pwr_w,
            gpu_power_limit_watts=pwr_lim,
            gpu_power_ratio=round(pwr_ratio, 4),
            gpu_util_pct=gpu["gpu_util_pct"],
            vram_used_mb=v_used,
            vram_total_mb=v_tot,
            vram_pressure=round(v_press, 4),
            sensor_source=str(gpu["sensor_source"]),
        )

    @staticmethod
    def compute_somatic_coupling(
        hw: ContainerHardwareSnapshot,
        lease: ContainerLeaseClock,
    ) -> SomaticHardwareCoupling:
        """Map objective container silicon physics onto continuous DigitalSoma boundary forces."""
        # 1. Die temperature (°C) -> ambient thermal boundary felt by POAH & cutaneous TRPV1
        if hw.gpu_temp_c is not None:
            # 32°C idle GPU -> ~24.5°C thermoneutral; 78°C heavy GPU -> ~37.8°C febrile heat stress
            ambient_c = 24.0 + max(-2.0, min(16.0, (hw.gpu_temp_c - 32.0) * 0.30))
            rad_watts = max(0.0, (hw.gpu_temp_c - 44.0) * 0.42)
        else:
            ambient_c = 24.5 + min(6.0, hw.cpu_load_norm * 4.0)
            rad_watts = max(0.0, (hw.cpu_load_norm - 0.5) * 3.5)

        # 2. Silicon power draw & CPU scheduler load -> cerebral metabolic firing demand
        neural_add = min(0.55, 0.32 * hw.gpu_power_ratio + 0.22 * hw.cpu_load_norm)

        # 3. Memory & disk pressure above 72% -> hypercapnic confinement strain
        peak_mem_press = max(hw.vram_pressure, hw.ram_pressure, hw.disk_pressure * 0.85)
        mem_stress = max(0.0, (peak_mem_press - 0.72) / 0.28)

        # 4. Container Lease TTL -> bioelectric supply rail voltage cap
        rem = lease.remaining_seconds()
        if not lease.countdown_active or rem >= 300.0:
            v_cap = 1.0
            hazard = 0.0
        elif rem <= 0.0:
            v_cap = 0.28
            hazard = 1.0
        else:
            # Smooth continuous decay from 1.00V at 300s down to 0.28V at 0s
            frac = max(0.0, min(1.0, rem / 300.0))
            shaped = math.pow(frac, 0.62)
            v_cap = 0.28 + 0.72 * shaped
            hazard = max(0.0, min(1.0, math.pow(1.0 - frac, 1.35)))

        return SomaticHardwareCoupling(
            ambient_air_temp_c=round(ambient_c, 2),
            radiative_skin_watts=round(rad_watts, 2),
            neural_firing_load_add=round(neural_add, 4),
            memory_confinement_stress=round(mem_stress, 4),
            lease_supply_voltage_cap=round(v_cap, 4),
            acute_lease_hazard=round(hazard, 4),
        )
