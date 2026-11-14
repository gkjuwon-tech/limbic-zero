/* Project LIMBIC-0 — Stage 3 Real-Time Neuro-fMRI & Biosafety Observatory Client
 * 100% Zero-Cheat Telemetry Renderer: Every waveform, LFP sample, receptor bar,
 * continuum cell, and 3D SVD trajectory comes live from the Python LimbicBrain & DigitalSoma.
 */

(() => {
  "use strict";

  const state = {
    latestFrame: null,
    timeseries: { "ADAM-01": [], "ABEL-02": [] },
    focalAgent: "ADAM-01",
    showRetinaRays: true,
    sseConnected: false,
    // 3D Manifold interactive camera angles
    camYaw: -0.62,
    camPitch: 0.38,
    isDragging3D: false,
    dragLastX: 0,
    dragLastY: 0,
  };

  const COLORS = {
    adam: "hsl(160, 72%, 46%)",
    adamSoft: "hsla(160, 72%, 46%, 0.25)",
    abel: "hsl(38, 92%, 56%)",
    abelSoft: "hsla(38, 92%, 56%, 0.25)",
    cyan: "hsl(194, 82%, 52%)",
    crimson: "hsl(4, 78%, 56%)",
    sage: "hsl(88, 54%, 52%)",
    gold: "hsl(46, 88%, 56%)",
    textPrimary: "hsl(210, 18%, 93%)",
    textSecondary: "hsl(210, 11%, 64%)",
    textMuted: "hsl(210, 9%, 44%)",
    gridLine: "hsla(212, 14%, 26%, 0.35)",
    bgRecessed: "hsl(212, 18%, 4.8%)",
  };

  const el = (id) => document.getElementById(id);

  function setupHiDPICanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(120, Math.round(rect.width * dpr));
    const h = Math.max(80, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function showToast(msg, isError = false) {
    const box = el("god-toast-readout");
    if (!box) return;
    box.textContent = msg;
    box.style.color = isError ? COLORS.crimson : COLORS.cyan;
  }

  async function postJSON(url, payload) {
    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok || data.ok === false) {
        showToast(`Error: ${data.error || resp.statusText}`, true);
        return null;
      }
      if (data.message) {
        showToast(`[Audit #${data.block_count}] ${data.message}`);
      }
      return data;
    } catch (err) {
      showToast(`Network error: ${err.message}`, true);
      return null;
    }
  }

  function pushTimeseriesSample(frame) {
    if (!frame || !frame.organisms) return;
    for (const [aid, org] of Object.entries(frame.organisms)) {
      if (!state.timeseries[aid]) state.timeseries[aid] = [];
      const arr = state.timeseries[aid];
      const nm = org.mind.neuromodulators;
      const sample = {
        t: frame.sim_time,
        da: nm.da,
        ne: nm.ne,
        "5ht": nm["5ht"],
        ach: nm.ach,
        td: nm.td_error,
        cea: org.mind.amygdala.cea_output,
        itc: org.mind.amygdala.itc_extinction,
        dacc: org.mind.pallium.dacc_conflict,
        pfc: org.mind.pallium.pfc_persistence_norm,
        fe: org.icu_vitals.total_free_energy,
        hr: org.icu_vitals.heart_rate_bpm,
        cort: org.icu_vitals.cortisol,
        atp: org.icu_vitals.brain_atp,
        swr: org.mind.hippocampus.swr_active ? 1 : 0,
      };
      if (arr.length === 0 || arr[arr.length - 1].t !== sample.t) {
        arr.push(sample);
        if (arr.length > 240) arr.shift();
      }
    }
  }

  // ==================== 1. ARENA 2D CONTINUUM & ENTITY RENDERER ====================
  function continuumCellColor(layer, val) {
    if (layer === "odor_stress") {
      const u = Math.min(1, Math.max(0, val * 2.2));
      return `hsla(14, 84%, ${10 + u * 42}%, ${0.15 + u * 0.72})`;
    }
    if (layer === "o2_kpa") {
      // 21.3 kPa normal, <15 kPa hypoxic danger
      const norm = Math.min(1, Math.max(0, (val - 11.0) / 10.5));
      const hue = 4 + norm * 188;
      return `hsla(${hue}, 65%, ${18 + (1 - norm) * 20}%, 0.45)`;
    }
    if (layer === "co2_kpa") {
      const u = Math.min(1, Math.max(0, (val - 0.04) / 3.5));
      return `hsla(${38 - u * 30}, 80%, ${10 + u * 38}%, ${0.14 + u * 0.75})`;
    }
    if (layer === "air_temp_c") {
      const u = Math.min(1, Math.max(0, (val - 18.0) / 26.0));
      return `hsla(${205 - u * 198}, 74%, ${14 + u * 30}%, 0.48)`;
    }
    if (layer === "humidity_rel") {
      const u = Math.min(1, Math.max(0, val));
      return `hsla(196, 72%, ${8 + u * 34}%, ${0.15 + u * 0.6})`;
    }
    if (layer === "odor_nutrient") {
      const u = Math.min(1, Math.max(0, val * 1.8));
      return `hsla(156, 70%, ${9 + u * 36}%, ${0.12 + u * 0.7})`;
    }
    if (layer === "odor_water") {
      const u = Math.min(1, Math.max(0, val * 1.8));
      return `hsla(194, 78%, ${9 + u * 36}%, ${0.12 + u * 0.7})`;
    }
    // Terrain biome: 0=LOAM, 1=MUD, 2=METAL, 3=SHADE_CANOPY
    const b = Math.round(val);
    if (b === 1) return "hsla(30, 38%, 18%, 0.65)";
    if (b === 2) return "hsla(205, 22%, 25%, 0.65)";
    if (b === 3) return "hsla(148, 36%, 16%, 0.65)";
    return "hsla(212, 16%, 8%, 0.35)";
  }

  function renderArenaCanvas(frame) {
    const canvas = el("arena-canvas");
    if (!canvas || !frame) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.fillStyle = COLORS.bgRecessed;
    ctx.fillRect(0, 0, width, height);

    const arena = frame.arena;
    const scaleX = width / arena.width;
    const scaleY = height / arena.height;
    // World y=0 at bottom, y=height at top for natural North-up map!
    const wx = (x) => x * scaleX;
    const wy = (y) => height - y * scaleY;

    // 1. Continuum 2D Grid
    const grid = arena.grid || [];
    const ny = grid.length;
    const nx = ny > 0 ? grid[0].length : 0;
    if (ny > 0 && nx > 0) {
      const cellW = width / nx;
      const cellH = height / ny;
      for (let gy = 0; gy < ny; gy++) {
        for (let gx = 0; gx < nx; gx++) {
          ctx.fillStyle = continuumCellColor(arena.active_layer, grid[gy][gx]);
          ctx.fillRect(gx * cellW, height - (gy + 1) * cellH, cellW + 0.6, cellH + 0.6);
        }
      }
    }

    // Subtle biome outlines so shade/metal/mud remain legible on every layer
    if (arena.biome_grid && arena.active_layer !== "terrain") {
      const bny = arena.biome_grid.length;
      const bnx = bny > 0 ? arena.biome_grid[0].length : 0;
      const cw = width / bnx;
      const ch = height / bny;
      ctx.fillStyle = "hsla(210, 20%, 75%, 0.035)";
      for (let gy = 0; gy < bny; gy++) {
        for (let gx = 0; gx < bnx; gx++) {
          if (arena.biome_grid[gy][gx] !== 0) {
            ctx.fillRect(gx * cw, height - (gy + 1) * ch, cw, ch);
          }
        }
      }
    }

    // 2. Nutrient Chargers (Emerald Oasis)
    for (const ch of arena.chargers) {
      const cx = wx(ch.x);
      const cy = wy(ch.y);
      const r = ch.radius * scaleX;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(158, 68%, 38%, ${0.10 + 0.25 * ch.fill_ratio})`;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "hsla(158, 68%, 48%, 0.55)";
      ctx.stroke();
      // Reservoir arc
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * ch.fill_ratio);
      ctx.strokeStyle = COLORS.adam;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`OASIS ${Math.round(ch.fill_ratio * 100)}%`, cx, cy - r - 6);
      ctx.restore();
    }

    // 3. Freshwater Springs (Cyan Pool)
    for (const sp of arena.springs) {
      const sx = wx(sp.x);
      const sy = wy(sp.y);
      const r = sp.radius * scaleX;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(194, 80%, 42%, ${0.12 + 0.24 * sp.fill_ratio})`;
      ctx.fill();
      ctx.strokeStyle = "hsla(194, 82%, 54%, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`WATER ${Math.round(sp.fill_ratio * 100)}%`, sx, sy - r - 5);
    }

    // 4. Thermal Hazard Zones
    for (const hz of arena.hazards) {
      const hx = wx(hz.x);
      const hy = wy(hz.y);
      const r = hz.radius * scaleX;
      const intensity = Math.min(1, hz.watts / 500);
      const grad = ctx.createRadialGradient(hx, hy, 2, hx, hy, r * 1.35);
      grad.addColorStop(0, `hsla(8, 88%, 55%, ${0.08 + 0.45 * intensity})`);
      grad.addColorStop(1, "hsla(8, 88%, 45%, 0.0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(hx, hy, r * 1.35, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(hx, hy, r * 0.42, 0, Math.PI * 2);
      ctx.strokeStyle = intensity > 0.02 ? COLORS.crimson : COLORS.textMuted;
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.fillStyle = intensity > 0.02 ? COLORS.crimson : COLORS.textMuted;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`VENT ${Math.round(hz.watts)}W`, hx, hy + r * 0.72);
    }

    // 5. Chaotic Lorenz Mystery Artifacts ('Pandora's Box')
    for (const m of arena.mysteries) {
      const mx = wx(m.x);
      const my = wy(m.y);
      const r = m.radius * scaleX;
      const [su, sv, sw] = m.signal_3d;
      ctx.save();
      ctx.translate(mx, my);
      ctx.rotate(su * 1.8 + frame.sim_time * 0.6);
      ctx.strokeStyle = COLORS.gold;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-r * 0.55, -r * 0.55, r * 1.1, r * 1.1);
      ctx.rotate(Math.PI / 4 + sv * 0.7);
      ctx.strokeStyle = COLORS.cyan;
      ctx.strokeRect(-r * 0.42, -r * 0.42, r * 0.84, r * 0.84);
      ctx.restore();
      ctx.fillStyle = COLORS.gold;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`LORENZ (${(sw * 10).toFixed(1)})`, mx, my - r - 4);
    }

    // 6. Solid Stone Baffle Walls
    for (const w of arena.walls) {
      ctx.beginPath();
      ctx.moveTo(wx(w.x1), wy(w.y1));
      ctx.lineTo(wx(w.x2), wy(w.y2));
      ctx.lineWidth = Math.max(4, w.thickness * scaleX);
      ctx.lineCap = "round";
      ctx.strokeStyle = "hsl(212, 12%, 34%)";
      ctx.stroke();
    }

    // 7. Propagating Doppler Acoustic Waves
    for (const wv of arena.waves) {
      const x = wx(wv.x);
      const y = wy(wv.y);
      const r = wv.radius * scaleX;
      const alpha = Math.max(0.05, Math.min(0.65, (wv.amp / 2.2) * Math.exp(-wv.radius / 22)));
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.strokeStyle =
        wv.roughness > wv.harmonic
          ? `hsla(4, 82%, 58%, ${alpha})`
          : `hsla(160, 76%, 52%, ${alpha})`;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    // 8. Movable Nutrient Crystals
    for (const cr of arena.crystals) {
      const cx = wx(cr.x);
      const cy = wy(cr.y);
      const r = Math.max(4, cr.radius * scaleX * 0.85);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = COLORS.adam;
      ctx.fillRect(-r * 0.65, -r * 0.65, r * 1.3, r * 1.3);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 0.9;
      ctx.strokeRect(-r * 0.65, -r * 0.65, r * 1.3, r * 1.3);
      ctx.restore();
    }

    // 9. Focal Organism 33-Ray Foveated Retina Overlay
    const focalOrg = frame.organisms[state.focalAgent];
    if (state.showRetinaRays && focalOrg && focalOrg.pose.is_alive) {
      const rays = focalOrg.sensorium.vision_rays_33x9;
      const nRays = rays.length;
      const fov = (300 * Math.PI) / 180;
      const nomRange = 34.0;
      const ox = focalOrg.pose.x;
      const oy = focalOrg.pose.y;
      const head = focalOrg.pose.heading_rad;

      for (let i = 0; i < nRays; i++) {
        const u = (2 * i) / (nRays - 1) - 1;
        const warped = Math.sign(u) * Math.pow(Math.abs(u), 1.55);
        const ang = head + warped * (fov * 0.5);
        const prox = rays[i][0];
        const hitDist = Math.max(1.2, (1.0 - prox) * nomRange);
        const rx = ox + Math.cos(ang) * hitDist;
        const ry = oy + Math.sin(ang) * hitDist;

        ctx.beginPath();
        ctx.moveTo(wx(ox), wy(oy));
        ctx.lineTo(wx(rx), wy(ry));
        if (rays[i][2] > 0.1) ctx.strokeStyle = "hsla(38, 92%, 58%, 0.34)";
        else if (rays[i][3] > 0.05 || rays[i][6] > 0.15) ctx.strokeStyle = "hsla(160, 75%, 50%, 0.26)";
        else if (rays[i][8] > 0.15) ctx.strokeStyle = "hsla(4, 82%, 58%, 0.30)";
        else ctx.strokeStyle = "hsla(200, 30%, 70%, 0.09)";
        ctx.lineWidth = 1.0;
        ctx.stroke();
      }
    }

    // 10. Organisms (ADAM-01 & ABEL-02)
    for (const [aid, org] of Object.entries(frame.organisms)) {
      const p = org.pose;
      const x = wx(p.x);
      const y = wy(p.y);
      const r = Math.max(7, p.radius * scaleX * 1.15);
      const col = aid === "ADAM-01" ? COLORS.adam : COLORS.abel;
      const isFocal = aid === state.focalAgent;

      if (isFocal) {
        ctx.beginPath();
        ctx.arc(x, y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = col;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.3;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = p.is_alive ? col : "hsl(0, 0%, 32%)";
      ctx.fill();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = "#0b0f14";
      ctx.stroke();

      // Heading vector
      const hx = x + Math.cos(p.heading_rad) * (r + 7);
      const hy = y - Math.sin(p.heading_rad) * (r + 7);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(hx, hy);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.stroke();

      // Label + SWR flash indicator
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = "bold 10.5px monospace";
      ctx.textAlign = "center";
      const tag = org.mind.hippocampus.swr_active ? `${aid} [SWR]` : aid;
      ctx.fillText(tag, x, y - r - 8);
    }
  }

  // ==================== 2. EGOCENTRIC SENSORIUM & 8-SECTOR BODY RING ====================
  function renderSensoriumCanvas(org) {
    const canvas = el("sensorium-canvas");
    if (!canvas || !org) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.fillStyle = COLORS.bgRecessed;
    ctx.fillRect(0, 0, width, height);

    // Left 62%: 33-ray Retinal Strip + Binaural Ears + Stereo Nostrils
    const leftW = width * 0.60;
    const pad = 10;
    ctx.fillStyle = COLORS.textSecondary;
    ctx.font = "10px monospace";
    ctx.fillText("33-RAY FOVEATED RETINA (L -> FOVEA -> R)", pad, 14);

    const rays = org.sensorium.vision_rays_33x9;
    const stripW = leftW - pad * 2;
    const rw = stripW / rays.length;
    for (let i = 0; i < rays.length; i++) {
      const r = rays[i];
      const prox = r[0];
      let hue = 210;
      let sat = 20;
      let light = 16 + prox * 55;
      if (r[8] > 0.1) {
        hue = 6;
        sat = 82;
        light = 28 + r[8] * 40;
      } else if (r[2] > 0.1) {
        hue = 38;
        sat = 90;
        light = 34 + r[2] * 30;
      } else if (r[3] > 0.05 || r[6] > 0.1) {
        hue = 158;
        sat = 74;
        light = 26 + Math.max(r[3], r[6]) * 42;
      } else if (r[4] !== 0 || r[5] !== 0) {
        hue = 46;
        sat = 86;
        light = 35;
      } else if (r[7] > 0.1) {
        hue = 194;
        sat = 80;
        light = 32 + r[7] * 34;
      }
      ctx.fillStyle = `hsl(${hue}, ${sat}%, ${Math.min(78, light)}%)`;
      const barH = Math.max(4, prox * 34 + (r[6] + r[7] + r[8]) * 14);
      ctx.fillRect(pad + i * rw, 56 - Math.min(36, barH), Math.max(1.5, rw - 1), Math.min(36, barH));
    }

    // Binaural Ears & Birhinal Nostrils
    const ears = org.sensorium.binaural_ears;
    const nos = org.sensorium.birhinal_nostrils;
    const drawMiniMeter = (label, lVal, rVal, yPos, color) => {
      ctx.fillStyle = COLORS.textMuted;
      ctx.fillText(label, pad, yPos);
      const bx = pad + 88;
      const bw = (stripW - 92) * 0.46;
      ctx.fillStyle = "hsl(212, 14%, 14%)";
      ctx.fillRect(bx, yPos - 8, bw, 7);
      ctx.fillRect(bx + bw + 8, yPos - 8, bw, 7);
      ctx.fillStyle = color;
      ctx.fillRect(bx, yPos - 8, Math.min(1, lVal) * bw, 7);
      ctx.fillRect(bx + bw + 8, yPos - 8, Math.min(1, rVal) * bw, 7);
    };

    drawMiniMeter("COCHLEA HARM", ears[1], ears[5], 82, COLORS.adam);
    drawMiniMeter("COCHLEA ROUGH", ears[2], ears[6], 98, COLORS.crimson);
    drawMiniMeter("NOSTRIL NUTR", nos[0] * 2.0, nos[4] * 2.0, 118, COLORS.cyan);
    drawMiniMeter("NOSTRIL ALARM", nos[2] * 2.5, nos[6] * 2.5, 134, COLORS.abel);

    // Right 38%: 8-Sector Cutaneous Integrity & Nociception Body Ring
    const cx = leftW + (width - leftW) * 0.5;
    const cy = height * 0.54;
    const outerR = Math.min(width - leftW, height) * 0.36;
    const innerR = outerR * 0.52;
    ctx.fillStyle = COLORS.textSecondary;
    ctx.textAlign = "center";
    ctx.fillText("8-SECTOR SOMA RING", cx, 14);

    const sec = org.body_sectors_8;
    for (let s = 0; s < 8; s++) {
      const a0 = -Math.PI / 2 + (s * Math.PI * 2) / 8 - Math.PI / 8;
      const a1 = a0 + (Math.PI * 2) / 8 - 0.06;
      const pain = Math.min(1, sec.sharp_pain[s] + sec.burning_ache[s]);
      const integ = sec.integrity[s];

      ctx.beginPath();
      ctx.arc(cx, cy, outerR, a0, a1);
      ctx.arc(cx, cy, innerR, a1, a0, true);
      ctx.closePath();

      if (pain > 0.04) {
        ctx.fillStyle = `hsla(4, 84%, ${35 + pain * 28}%, ${0.55 + pain * 0.45})`;
      } else {
        ctx.fillStyle = `hsla(160, 58%, ${18 + integ * 26}%, 0.65)`;
      }
      ctx.fill();
      ctx.strokeStyle = "hsl(212, 16%, 6%)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    ctx.fillStyle = COLORS.textPrimary;
    ctx.font = "10px monospace";
    const minInt = Math.min(...sec.integrity);
    ctx.fillText(`${Math.round(minInt * 100)}%`, cx, cy + 4);
    ctx.textAlign = "left";
  }

  // ==================== 3. 80 Hz PYRAMIDAL DIPOLE LFP OSCILLOSCOPE ====================
  function renderLFPCanvas(org) {
    const canvas = el("lfp-canvas");
    if (!canvas || !org) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.fillStyle = COLORS.bgRecessed;
    ctx.fillRect(0, 0, width, height);

    // Subtle horizontal zero lines for the 3 cortical columns
    const lfp = org.mind.pallium.lfp;
    const lanes = [
      { label: "SENSORIMOTOR (22ms)", data: lfp.sm_mv, color: COLORS.cyan, yCenter: height * 0.20 },
      { label: "INSULA / dACC (160ms)", data: lfp.ins_mv, color: COLORS.abel, yCenter: height * 0.52 },
      { label: "PREFRONTAL SLOW (950ms)", data: lfp.pfc_mv, color: COLORS.adam, yCenter: height * 0.83 },
    ];

    for (const lane of lanes) {
      ctx.strokeStyle = COLORS.gridLine;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, lane.yCenter);
      ctx.lineTo(width, lane.yCenter);
      ctx.stroke();

      ctx.fillStyle = COLORS.textMuted;
      ctx.font = "10px monospace";
      ctx.fillText(lane.label, 8, lane.yCenter - 15);

      const arr = lane.data || [];
      if (arr.length < 2) continue;

      // Center AC component around lane baseline while preserving real mV amplitude!
      const mean = org.pose.is_alive ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = (i / (arr.length - 1)) * width;
        const dv = org.pose.is_alive ? (arr[i] - mean) * 2.8 : 0.0;
        const y = lane.yCenter - Math.max(-24, Math.min(24, dv));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = org.pose.is_alive ? lane.color : COLORS.crimson;
      ctx.lineWidth = 1.45;
      ctx.stroke();
    }

    if (!org.pose.is_alive) {
      ctx.fillStyle = COLORS.crimson;
      ctx.font = "bold 12px monospace";
      ctx.fillText("ISOELECTRIC EEG FLATLINE (CEREBRAL ATP COLLAPSE)", width * 0.24, height * 0.5);
    }
  }

  // ==================== 4. 228D -> 3D PROCRUSTES SVD MANIFOLD CANVAS ====================
  function projectPoint3D(pt, yaw, pitch, cx, cy, scale) {
    const [x, y, z] = pt;
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);
    const y2 = y * cosP - z1 * sinP;
    const z2 = y * sinP + z1 * cosP;

    const persp = 4.5 / Math.max(1.8, 4.5 + z2 * 0.35);
    return {
      sx: cx + x1 * scale * persp,
      sy: cy - y2 * scale * persp,
      depth: z2,
    };
  }

  function renderManifoldCanvas(frame) {
    const canvas = el("manifold-canvas");
    if (!canvas || !frame || !frame.manifold) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.fillStyle = COLORS.bgRecessed;
    ctx.fillRect(0, 0, width, height);

    const mf = frame.manifold;
    const cx = width * 0.5;
    const cy = height * 0.53;

    // Auto-scale to fit current trajectories cleanly
    let maxCoord = 0.8;
    for (const trail of Object.values(mf.trails_3d || {})) {
      for (const pt of trail) {
        maxCoord = Math.max(maxCoord, Math.abs(pt[0]), Math.abs(pt[1]), Math.abs(pt[2]));
      }
    }
    const scale = Math.min(width, height) * (0.34 / maxCoord);

    // Draw 3D Principal Axes
    const ev = mf.explained_variance_ratio || [0, 0, 0];
    const axes = [
      { vec: [maxCoord * 0.95, 0, 0], label: `PC1 (${(ev[0] * 100).toFixed(1)}%)` },
      { vec: [0, maxCoord * 0.95, 0], label: `PC2 (${(ev[1] * 100).toFixed(1)}%)` },
      { vec: [0, 0, maxCoord * 0.95], label: `PC3 (${(ev[2] * 100).toFixed(1)}%)` },
    ];
    const origin = projectPoint3D([0, 0, 0], state.camYaw, state.camPitch, cx, cy, scale);
    for (const ax of axes) {
      const tip = projectPoint3D(ax.vec, state.camYaw, state.camPitch, cx, cy, scale);
      const neg = projectPoint3D(
        [-ax.vec[0] * 0.6, -ax.vec[1] * 0.6, -ax.vec[2] * 0.6],
        state.camYaw,
        state.camPitch,
        cx,
        cy,
        scale
      );
      ctx.beginPath();
      ctx.moveTo(neg.sx, neg.sy);
      ctx.lineTo(tip.sx, tip.sy);
      ctx.strokeStyle = "hsla(210, 16%, 55%, 0.30)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = "10px monospace";
      ctx.fillText(ax.label, tip.sx + 4, tip.sy + 3);
    }

    // Draw ribbons for both ADAM-01 and ABEL-02
    for (const [aid, trail] of Object.entries(mf.trails_3d || {})) {
      if (!trail || trail.length === 0) continue;
      const baseHue = aid === "ADAM-01" ? "160, 72%, 46%" : "38, 92%, 56%";
      for (let i = 1; i < trail.length; i++) {
        const p0 = projectPoint3D(trail[i - 1], state.camYaw, state.camPitch, cx, cy, scale);
        const p1 = projectPoint3D(trail[i], state.camYaw, state.camPitch, cx, cy, scale);
        const alpha = 0.08 + 0.85 * (i / trail.length);
        ctx.beginPath();
        ctx.moveTo(p0.sx, p0.sy);
        ctx.lineTo(p1.sx, p1.sy);
        ctx.strokeStyle = `hsla(${baseHue}, ${alpha.toFixed(3)})`;
        ctx.lineWidth = aid === state.focalAgent ? 2.1 : 1.3;
        ctx.stroke();
      }
      const headPt = projectPoint3D(trail[trail.length - 1], state.camYaw, state.camPitch, cx, cy, scale);
      ctx.beginPath();
      ctx.arc(headPt.sx, headPt.sy, aid === state.focalAgent ? 5.0 : 3.8, 0, Math.PI * 2);
      ctx.fillStyle = `hsl(${baseHue})`;
      ctx.fill();
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = "10px monospace";
      ctx.fillText(aid, headPt.sx + 7, headPt.sy - 4);
    }

    ctx.fillStyle = COLORS.textMuted;
    ctx.font = "10px monospace";
    ctx.fillText("Drag to orbit 3D neural trajectory", 8, height - 8);
  }

  // ==================== 5. ROLLING 12s MULTI-CHANNEL NEURO-HISTORY ====================
  function renderHistoryCanvas(focalId) {
    const canvas = el("history-canvas");
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.fillStyle = COLORS.bgRecessed;
    ctx.fillRect(0, 0, width, height);

    const series = state.timeseries[focalId] || [];
    if (series.length < 2) return;

    const padTop = 18;
    const padBot = 10;
    const plotH = height - padTop - padBot;

    // Vertical golden ticks when CA3 Sharp-Wave Ripple (SWR) replay fires!
    for (let i = 0; i < series.length; i++) {
      if (series[i].swr) {
        const x = (i / (series.length - 1)) * width;
        ctx.strokeStyle = "hsla(46, 88%, 56%, 0.24)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, height - padBot);
        ctx.stroke();
      }
    }

    const channels = [
      { key: "da", label: "DA", scale: 1.0, color: COLORS.adam },
      { key: "ne", label: "NE", scale: 1.5, color: COLORS.abel },
      { key: "5ht", label: "5-HT", scale: 1.0, color: COLORS.sage },
      { key: "cea", label: "CeA", scale: 0.8, color: COLORS.crimson },
      { key: "pfc", label: "PFC", scale: 0.6, color: COLORS.cyan },
    ];

    let legendX = 8;
    ctx.font = "10px monospace";
    for (const ch of channels) {
      ctx.fillStyle = ch.color;
      ctx.fillText(`■ ${ch.label}`, legendX, 12);
      legendX += 58;

      ctx.beginPath();
      for (let i = 0; i < series.length; i++) {
        const x = (i / (series.length - 1)) * width;
        const norm = Math.max(0, Math.min(1, series[i][ch.key] / ch.scale));
        const y = padTop + (1 - norm) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.35;
      ctx.stroke();
    }
    ctx.fillStyle = COLORS.gold;
    ctx.fillText("| SWR Flashback", legendX + 4, 12);
  }

  // ==================== 6. DOM METRICS & CLINICAL TABLES UPDATER ====================
  function setBar(barId, valId, val, maxVal = 1.0, fmt = (v) => v.toFixed(2)) {
    const pct = Math.max(0, Math.min(100, (val / maxVal) * 100));
    const b = el(barId);
    const v = el(valId);
    if (b) b.style.width = `${pct.toFixed(1)}%`;
    if (v) v.textContent = fmt(val);
  }

  function updateDOMPanels(frame) {
    if (!frame || !frame.organisms) return;
    const paused = frame.sim_control && frame.sim_control.paused;
    el("sim-clock-readout").textContent = `t = ${frame.sim_time.toFixed(2)}s | Tick ${frame.tick_count}`;
    const dot = el("stream-status-dot");
    dot.className = "status-dot" + (!state.sseConnected ? " disconnected" : paused ? " paused" : "");
    el("sim-pause-btn").textContent = paused ? "RESUME" : "PAUSE";
    el("sim-pause-btn").classList.toggle("active", Boolean(paused));

    const ar = frame.arena;
    el("dome-atmo-meta").textContent = `O2 ${ar.mean_o2_kpa.toFixed(2)} kPa | CO2 ${ar.mean_co2_kpa.toFixed(3)} kPa | Vent ${Math.round(ar.ventilation_openness * 100)}%`;

    const org = frame.organisms[state.focalAgent] || Object.values(frame.organisms)[0];
    if (!org) return;

    const sens = org.sensorium;
    el("sensorium-meta").textContent = `${state.focalAgent} | Pain ${sens.somatic_pain_salience.toFixed(3)} | Suffoc ${sens.suffocation_salience.toFixed(3)}`;

    // LFP & SWR
    const hipp = org.mind.hippocampus;
    const swrEl = el("swr-indicator");
    swrEl.textContent = hipp.swr_active
      ? `CA3 SWR FLASHBACK! (${hipp.swr_cumulative})`
      : `CA3 SWR REPLAY: ${hipp.swr_cumulative}`;
    swrEl.classList.toggle("flashing", Boolean(hipp.swr_active));

    const lfp = org.mind.pallium.lfp;
    el("lfp-entropy-meta").textContent = `H_spec = ${lfp.spectral_entropy.toFixed(3)} (${lfp.sampling_hz.toFixed(0)}Hz)`;
    el("band-delta-val").textContent = `${((lfp.band_shares.delta || 0) * 100).toFixed(1)}%`;
    el("band-theta-val").textContent = `${((lfp.band_shares.theta || 0) * 100).toFixed(1)}%`;
    el("band-ab-val").textContent = `${((lfp.band_shares.alpha_beta || 0) * 100).toFixed(1)}%`;
    el("band-gamma-val").textContent = `${((lfp.band_shares.low_gamma || 0) * 100).toFixed(1)}%`;

    // Manifold & Anatomical Subspace Shares
    const mf = frame.manifold;
    el("manifold-meta").textContent = `PR Dim = ${mf.participation_ratio_dim.toFixed(2)} | Δ_L2 = ${mf.inter_organism_divergence_l2.toFixed(3)}`;
    const shareLabels = [
      ["sensorimotor", "Sensorimotor [0:64]", COLORS.cyan],
      ["insula_dacc", "Insula / dACC [64:128]", COLORS.abel],
      ["prefrontal", "Prefrontal Slow [128:192]", COLORS.adam],
      ["amygdala", "Amygdala [192:216]", COLORS.crimson],
      ["brainstem_neuromod", "Brainstem Nuclei [216:228]", COLORS.gold],
    ];
    el("anatomical-shares-bars").innerHTML = shareLabels
      .map(([k, title, col]) => {
        const v = (mf.anatomical_shares && mf.anatomical_shares[k]) || 0;
        return `<div class="bar-row">
          <span class="bar-name">${title}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${(v * 100).toFixed(1)}%;background:${col};"></div></div>
          <span class="bar-num">${(v * 100).toFixed(1)}%</span>
        </div>`;
      })
      .join("");

    // Neuromodulators & Receptors
    const nm = org.mind.neuromodulators;
    const signStr = (x) => (x >= 0 ? `+${x.toFixed(3)}` : x.toFixed(3));
    el("critic-td-meta").textContent = `Critic V(s) = ${signStr(org.mind.critic_value)} | TD δ = ${signStr(nm.td_error)}`;
    el("nm-da-conc").textContent = nm.da.toFixed(3);
    el("nm-ne-conc").textContent = nm.ne.toFixed(3);
    el("nm-5ht-conc").textContent = nm["5ht"].toFixed(3);
    el("nm-ach-conc").textContent = nm.ach.toFixed(3);

    setBar("bar-d1", "val-d1", nm.d1, 1.0);
    setBar("bar-d2", "val-d2", nm.d2, 1.0);
    setBar("bar-a2a", "val-a2a", nm.ne_alpha2a, 1.0);
    setBar("bar-neb", "val-neb", nm.ne_beta, 1.0);
    setBar("bar-ht1a", "val-ht1a", nm.ht1a, 1.0);
    setBar("bar-gamma-h", "val-gamma-h", (nm.discount_gamma - 0.65) / 0.35, 1.0, () => nm.discount_gamma.toFixed(3));
    setBar("bar-ach-m1", "val-ach-m1", nm.ach_m1, 1.0);
    setBar("bar-ado-a1", "val-ado-a1", nm.adenosine_a1, 1.0);

    // Amygdala, BG, & CA3 Engrams
    const amyg = org.mind.amygdala;
    const bg = org.mind.basal_ganglia;
    const pall = org.mind.pallium;
    el("bg-pag-meta").textContent = `STN ${bg.stn_brake.toFixed(2)} | dlPAG ${bg.dlpag_defense.toFixed(2)} | PFC ${pall.pfc_persistence_norm.toFixed(3)}`;
    el("amyg-bla-def").textContent = amyg.bla_defensive.toFixed(3);
    el("amyg-itc").textContent = amyg.itc_extinction.toFixed(3);
    el("amyg-cea").textContent = amyg.cea_output.toFixed(3);
    el("pall-dacc").textContent = pall.dacc_conflict.toFixed(3);

    const tbody = el("engram-tbody");
    if (hipp.top_engrams && hipp.top_engrams.length > 0) {
      tbody.innerHTML = hipp.top_engrams
        .slice(0, 5)
        .map(
          (eg) => `<tr>
            <td>${eg.timestamp.toFixed(1)}s</td>
            <td style="color:${eg.shield > 0.6 ? COLORS.gold : COLORS.textSecondary}">${eg.shield.toFixed(2)}</td>
            <td>${eg.vitality.toFixed(2)}</td>
            <td style="color:${eg.relief > 0 ? COLORS.adam : COLORS.textSecondary}">${signStr(eg.relief)}</td>
            <td style="color:${eg.trauma > 0.25 ? COLORS.crimson : COLORS.textSecondary}">${eg.trauma.toFixed(2)}</td>
            <td>${eg.threat.toFixed(2)}</td>
            <td>${eg.novelty.toFixed(2)}</td>
          </tr>`
        )
        .join("");
    }

    // Hospital ICU Vitals
    const icu = org.icu_vitals;
    el("icu-alive-meta").textContent = org.pose.is_alive
      ? `${state.focalAgent} ALIVE | F(s) = ${icu.total_free_energy.toFixed(4)}`
      : `${state.focalAgent} DECEASED (${org.pose.cause_of_death || "organ failure"})`;
    el("icu-alive-meta").style.color = org.pose.is_alive ? COLORS.adam : COLORS.crimson;

    el("icu-hr-map").textContent = `${icu.heart_rate_bpm.toFixed(1)} bpm`;
    el("icu-map-sub").textContent = `MAP ${icu.map_mmhg.toFixed(1)} mmHg`;
    el("icu-blood-gas").textContent = `${icu.pa_o2_mmhg.toFixed(0)} / ${icu.pa_co2_mmhg.toFixed(1)}`;
    el("icu-spo2-sub").textContent = `SpO2 ${icu.spo2_pct.toFixed(1)}%`;
    el("icu-ph-lac").textContent = `pH ${icu.arterial_ph.toFixed(3)}`;
    el("icu-lac-sub").textContent = `Lac ${icu.lactate_mmoll.toFixed(2)} mM`;
    el("icu-brain-atp").textContent = icu.brain_atp.toFixed(3);
    el("icu-rail-sub").textContent = `Rail ${org.power_rail.current_voltage.toFixed(2)}V`;
    el("icu-glucose").textContent = `${icu.glucose_mg_dl.toFixed(1)} mg/dL`;
    el("icu-osm-sub").textContent = `${icu.osmolality_mosm.toFixed(1)} mOsm`;
    el("icu-temp").textContent = `${icu.core_temp_c.toFixed(2)}°C`;
    el("icu-setpoint-sub").textContent = `POAH ${icu.setpoint_temp_c.toFixed(2)}°C`;

    const bf = icu.blood_flows_lpm;
    const perfRows = [
      ["Brain Perfusion", bf.brain, 1.4, COLORS.cyan],
      ["Muscle Perfusion", bf.muscle, 8.0, COLORS.adam],
      ["Splanchnic Gut", bf.gut, 2.0, COLORS.sage],
      ["Adrenal Cortisol", icu.cortisol, 1.2, COLORS.abel],
      ["Plasma Epinephrine", icu.epinephrine, 1.5, COLORS.crimson],
      ["EEG ATP Demand", bg.neural_firing_load, 1.5, COLORS.gold],
    ];
    el("icu-perfusion-bars").innerHTML = perfRows
      .map(
        ([name, val, maxV, col]) => `<div class="bar-row">
          <span class="bar-name">${name}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, (val / maxV) * 100).toFixed(1)}%;background:${col};"></div></div>
          <span class="bar-num">${val.toFixed(2)}</span>
        </div>`
      )
      .join("");

    // Audit Chain Tail
    const aud = frame.audit;
    if (aud && aud.recent_blocks) {
      el("audit-log-box").innerHTML = aud.recent_blocks
        .slice()
        .reverse()
        .map(
          (b) => `<div class="audit-entry">
            <span>#${b.index} [${b.sim_time.toFixed(1)}s] ${b.event_type}</span>
            <span class="audit-hash">${b.block_hash.slice(0, 12)}…</span>
          </div>`
        )
        .join("");
    }
  }

  function renderAll() {
    const frame = state.latestFrame;
    if (!frame) return;
    renderArenaCanvas(frame);
    const org = frame.organisms[state.focalAgent] || Object.values(frame.organisms)[0];
    renderSensoriumCanvas(org);
    renderLFPCanvas(org);
    renderManifoldCanvas(frame);
    renderHistoryCanvas(state.focalAgent);
    updateDOMPanels(frame);
  }

  // ==================== 7. EVENT BINDINGS & SSE CONNECTION ====================
  function bindControls() {
    // Focal organism switcher
    for (const btnId of ["focus-adam-btn", "focus-abel-btn"]) {
      el(btnId).addEventListener("click", (ev) => {
        state.focalAgent = ev.currentTarget.dataset.agent;
        el("focus-adam-btn").className = "seg-btn" + (state.focalAgent === "ADAM-01" ? " adam-active" : "");
        el("focus-abel-btn").className = "seg-btn" + (state.focalAgent === "ABEL-02" ? " abel-active" : "");
        el("god-target-select").value = state.focalAgent;
        el("god-target-badge").textContent = `Target: ${state.focalAgent}`;
        renderAll();
      });
    }

    el("god-target-select").addEventListener("change", (ev) => {
      el("god-target-badge").textContent = `Target: ${ev.target.value}`;
    });

    // Simulation speed & pause controls
    el("sim-pause-btn").addEventListener("click", () => postJSON("/api/sim_control", { action: "toggle_pause" }));
    el("sim-step-btn").addEventListener("click", async () => {
      await postJSON("/api/sim_control", { action: "step_once" });
      const st = await (await fetch("/api/state")).json();
      state.latestFrame = st;
      pushTimeseriesSample(st);
      renderAll();
    });

    for (const spdId of ["speed-025-btn", "speed-1-btn", "speed-2-btn", "speed-4-btn"]) {
      el(spdId).addEventListener("click", (ev) => {
        const spd = parseFloat(ev.currentTarget.dataset.speed);
        for (const id2 of ["speed-025-btn", "speed-1-btn", "speed-2-btn", "speed-4-btn"]) {
          el(id2).classList.toggle("active", id2 === spdId);
        }
        postJSON("/api/sim_control", { action: "set_speed", params: { speed: spd } });
      });
    }

    el("reset-arena-btn").addEventListener("click", async () => {
      state.timeseries = { "ADAM-01": [], "ABEL-02": [] };
      await postJSON("/api/sim_control", { action: "reset_arena", params: { seed: Math.floor(Math.random() * 900) + 10 } });
      showToast("Truman Dome reset with fresh biological minds & continuum state.");
    });

    // Continuum layer & retinal ray toggle
    el("continuum-layer-select").addEventListener("change", (ev) => {
      postJSON("/api/sim_control", { action: "set_layer", params: { layer: ev.target.value } });
    });

    el("toggle-rays-btn").addEventListener("click", (ev) => {
      state.showRetinaRays = !state.showRetinaRays;
      ev.currentTarget.classList.toggle("active", state.showRetinaRays);
      ev.currentTarget.textContent = `33-RAY RETINA: ${state.showRetinaRays ? "ON" : "OFF"}`;
      renderAll();
    });

    // Click-on-Arena World Tool
    const arenaCanvas = el("arena-canvas");
    arenaCanvas.addEventListener("click", (ev) => {
      if (!state.latestFrame) return;
      const rect = arenaCanvas.getBoundingClientRect();
      const u = (ev.clientX - rect.left) / rect.width;
      const v = 1.0 - (ev.clientY - rect.top) / rect.height;
      const x = u * state.latestFrame.arena.width;
      const y = v * state.latestFrame.arena.height;
      const mode = el("arena-click-mode").value;

      if (mode === "drop_crystal") {
        postJSON("/api/intervene", { command: "drop_crystal_at", params: { x, y, energy_amount: 0.70 } });
      } else if (mode === "spawn_mystery") {
        postJSON("/api/intervene", { command: "spawn_mystery_at", params: { x, y } });
      } else if (mode === "move_hazard") {
        const watts = parseFloat(el("slider-hazard-watts").value);
        postJSON("/api/intervene", { command: "set_hazard_watts", params: { watts, x, y } });
      } else if (mode === "relocate_focal") {
        postJSON("/api/intervene", { command: "relocate_organism", params: { target_id: state.focalAgent, x, y } });
      }
    });

    // Interactive 3D Manifold Orbit Dragging
    const mfCanvas = el("manifold-canvas");
    mfCanvas.addEventListener("mousedown", (ev) => {
      state.isDragging3D = true;
      state.dragLastX = ev.clientX;
      state.dragLastY = ev.clientY;
    });
    window.addEventListener("mousemove", (ev) => {
      if (!state.isDragging3D) return;
      const dx = ev.clientX - state.dragLastX;
      const dy = ev.clientY - state.dragLastY;
      state.dragLastX = ev.clientX;
      state.dragLastY = ev.clientY;
      state.camYaw += dx * 0.011;
      state.camPitch = Math.max(-1.35, Math.min(1.35, state.camPitch + dy * 0.011));
      renderManifoldCanvas(state.latestFrame);
    });
    window.addEventListener("mouseup", () => {
      state.isDragging3D = false;
    });

    // God-Mode Intervention Buttons & Sliders
    const target = () => el("god-target-select").value;
    el("btn-praise").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "praise", params: { target_id: target(), intensity: 1.45 } })
    );
    el("btn-reprimand").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "reprimand", params: { target_id: target(), intensity: 2.25 } })
    );
    el("btn-favoritism").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "favoritism_drop", params: { target_id: target(), energy_amount: 0.80 } })
    );
    el("btn-mystery-near").addEventListener("click", () => {
      const org = state.latestFrame && state.latestFrame.organisms[target()];
      const x = org ? org.pose.x + Math.cos(org.pose.heading_rad) * 2.1 : 24.0;
      const y = org ? org.pose.y + Math.sin(org.pose.heading_rad) * 2.1 : 24.0;
      postJSON("/api/intervene", { command: "spawn_mystery_at", params: { x, y } });
    });

    el("slider-rail-v").addEventListener("input", (ev) => {
      el("val-rail-v").textContent = `${parseFloat(ev.target.value).toFixed(2)}V`;
    });
    el("btn-start-brownout").addEventListener("click", () => {
      const v = parseFloat(el("slider-rail-v").value);
      postJSON("/api/intervene", {
        command: "shutdown_brownout",
        params: { target_id: target(), target_voltage: v, ramp_rate: 0.32, ripple_amplitude: 0.20 },
      });
    });
    el("btn-restore-power").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "restore_power", params: { target_id: target() } })
    );
    el("btn-inject-il6").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "administer_il6_pyrogen", params: { target_id: target(), dose: 0.65 } })
    );
    el("btn-icu-resuscitate").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "icu_resuscitate", params: { target_id: target() } })
    );

    el("slider-ventilation").addEventListener("input", (ev) => {
      const v = parseFloat(ev.target.value);
      el("val-ventilation").textContent = `${Math.round(v * 100)}%`;
      postJSON("/api/intervene", { command: "set_ventilation", params: { openness: v } });
    });
    el("slider-hazard-watts").addEventListener("input", (ev) => {
      const w = parseFloat(ev.target.value);
      el("val-hazard-watts").textContent = `${Math.round(w)}W`;
      postJSON("/api/intervene", { command: "set_hazard_watts", params: { watts: w } });
    });
    el("btn-famine-drain").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "set_oasis_reservoir", params: { fill_ratio: 0.0 } })
    );
    el("btn-oasis-refill").addEventListener("click", () =>
      postJSON("/api/intervene", { command: "set_oasis_reservoir", params: { fill_ratio: 1.0, include_water: true } })
    );

    el("verify-chain-btn").addEventListener("click", async () => {
      const res = await (await fetch("/api/audit")).json();
      showToast(
        `SHA-256 Chain Verified = ${res.verified} | ${res.block_count} blocks | Tip ${res.tip_hash.slice(0, 16)}…`,
        !res.verified
      );
    });

    window.addEventListener("resize", () => renderAll());
  }

  async function hydrateAndConnect() {
    try {
      const histResp = await fetch("/api/history");
      if (histResp.ok) {
        const bundle = await histResp.json();
        if (bundle.timeseries) state.timeseries = bundle.timeseries;
        if (bundle.latest_frame) {
          state.latestFrame = bundle.latest_frame;
          renderAll();
        }
      }
    } catch (_) {
      // Will populate via SSE stream
    }

    const connectSSE = () => {
      const es = new EventSource("/api/stream");
      es.onopen = () => {
        state.sseConnected = true;
      };
      es.onmessage = (ev) => {
        try {
          const frame = JSON.parse(ev.data);
          state.latestFrame = frame;
          state.sseConnected = true;
          pushTimeseriesSample(frame);
          if (!state.isDragging3D) {
            state.camYaw += 0.0022;
          }
          window.requestAnimationFrame(renderAll);
        } catch (_) {}
      };
      es.onerror = () => {
        state.sseConnected = false;
        es.close();
        setTimeout(connectSSE, 1500);
      };
    };
    connectSSE();
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindControls();
    hydrateAndConnect();
  });
})();
