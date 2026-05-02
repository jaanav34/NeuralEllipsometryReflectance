from __future__ import annotations

import colorsys
import json
import uuid
from typing import Any

import streamlit.components.v1 as components


THICKNESS_RANGE = (10.0, 300.0)
N_RANGE = (1.3, 2.5)
K_RANGE = (0.0, 0.5)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))



def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)



def _hls_to_rgba(h_deg: float, l_pct: float, s_pct: float, alpha: float) -> str:
    h = (h_deg % 360.0) / 360.0
    l = _clamp(l_pct / 100.0, 0.0, 1.0)
    s = _clamp(s_pct / 100.0, 0.0, 1.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (
        f"rgba({int(round(r * 255))}, {int(round(g * 255))}, "
        f"{int(round(b * 255))}, {_clamp(alpha, 0.0, 1.0):.3f})"
    )



def _film_palette(thickness: float, n: float, k: float, alpha: float = 0.88) -> dict[str, str]:
    t_norm = _norm(thickness, *THICKNESS_RANGE)
    n_norm = _norm(n, *N_RANGE)
    k_norm = _norm(k, *K_RANGE)

    hue = 228.0 - 180.0 * n_norm
    sat = 72.0 + 10.0 * t_norm
    light = 71.0 - 26.0 * k_norm

    accent = _hls_to_rgba(hue, light, sat, 1.0)
    top = _hls_to_rgba(hue, light + 6.0, sat - 4.0, alpha)
    front = _hls_to_rgba(hue, light - 2.0, sat + 2.0, min(alpha + 0.04, 0.98))
    side = _hls_to_rgba(hue, light - 8.0, sat + 3.0, min(alpha + 0.06, 0.98))
    glow = _hls_to_rgba(hue, light + 12.0, sat - 10.0, 0.30)
    shell = _hls_to_rgba(hue, light + 8.0, sat - 2.0, 0.22)
    return {
        "accent": accent,
        "top": top,
        "front": front,
        "side": side,
        "glow": glow,
        "shell": shell,
    }



def make_visual_state(
    label: str,
    thickness: float,
    n: float,
    k: float,
    *,
    ci_thickness: float | None = None,
    ci_n: float | None = None,
    ci_k: float | None = None,
    spectral_mae: float | None = None,
    role: str = "prediction",
    emphasis: bool = False,
) -> dict[str, Any]:
    palette = _film_palette(thickness, n, k)
    return {
        "label": str(label),
        "thickness": float(thickness),
        "n": float(n),
        "k": float(k),
        "ci_thickness": None if ci_thickness is None else float(ci_thickness),
        "ci_n": None if ci_n is None else float(ci_n),
        "ci_k": None if ci_k is None else float(ci_k),
        "spectral_mae": None if spectral_mae is None else float(spectral_mae),
        "role": role,
        "emphasis": bool(emphasis),
        "palette": palette,
    }



def render_prediction_cards(
    states: list[dict[str, Any]],
    *,
    primary_label: str | None = None,
) -> str:
    if not states:
        return ""

    cards: list[str] = []
    for state in states:
        label = state["label"]
        accent = state["palette"]["accent"]
        is_primary = primary_label is not None and label == primary_label
        ci_line = ""
        if state.get("ci_thickness") is not None:
            ci_line = (
                "<div class='tfv-meta'>"
                f"95% CI: ±{state['ci_thickness']:.2f} nm"
                f" · ±{state['ci_n']:.4f} n"
                f" · ±{state['ci_k']:.4f} k"
                "</div>"
            )
        mae_line = ""
        if state.get("spectral_mae") is not None:
            mae_line = (
                "<div class='tfv-meta'>"
                f"Spectral MAE: {state['spectral_mae']:.6f}"
                "</div>"
            )
        role_text = state.get("role", "prediction").replace("_", " ").title()
        primary_badge = "<span class='tfv-badge'>Primary</span>" if is_primary else ""
        cards.append(
            "<div class='tfv-card'>"
            f"<div class='tfv-card-head'><div class='tfv-dot' style='background:{accent};'></div>"
            f"<div class='tfv-title'>{label}</div>{primary_badge}</div>"
            f"<div class='tfv-role'>{role_text}</div>"
            f"<div class='tfv-grid'>"
            f"<div><span class='tfv-k'>Thickness</span><span class='tfv-v'>{state['thickness']:.2f} nm</span></div>"
            f"<div><span class='tfv-k'>n</span><span class='tfv-v'>{state['n']:.4f}</span></div>"
            f"<div><span class='tfv-k'>k</span><span class='tfv-v'>{state['k']:.4f}</span></div>"
            "</div>"
            f"{ci_line}{mae_line}"
            "</div>"
        )

    return f"""
<div class="tfv-wrap">
  <style>
    .tfv-wrap {{
      display: grid;
      gap: 0.75rem;
      margin-top: 0.2rem;
    }}
    .tfv-card {{
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 16px;
      padding: 0.95rem 1rem;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
    }}
    .tfv-card-head {{
      display: flex;
      align-items: center;
      gap: 0.55rem;
      margin-bottom: 0.15rem;
    }}
    .tfv-dot {{
      width: 0.8rem;
      height: 0.8rem;
      border-radius: 999px;
      box-shadow: 0 0 0 4px rgba(148, 163, 184, 0.12);
      flex: 0 0 auto;
    }}
    .tfv-title {{
      font-weight: 700;
      font-size: 0.98rem;
      color: #0f172a;
      flex: 1 1 auto;
    }}
    .tfv-role {{
      color: #64748b;
      font-size: 0.82rem;
      margin-bottom: 0.65rem;
    }}
    .tfv-badge {{
      background: rgba(15, 23, 42, 0.06);
      color: #0f172a;
      border-radius: 999px;
      font-size: 0.72rem;
      padding: 0.18rem 0.45rem;
      font-weight: 700;
    }}
    .tfv-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.6rem;
    }}
    .tfv-k {{
      display: block;
      font-size: 0.76rem;
      color: #64748b;
      margin-bottom: 0.1rem;
    }}
    .tfv-v {{
      display: block;
      font-size: 1.9rem;
      font-weight: 700;
      color: #111827;
    }}
    .tfv-meta {{
      margin-top: 0.55rem;
      color: #475569;
      font-size: 0.82rem;
      line-height: 1.35;
    }}
  </style>
  {''.join(cards)}
</div>
"""



def render_thinfilm_visualizer(
    states: list[dict[str, Any]],
    *,
    layout: str = "Overlay",
    animation_speed: float = 1.0,
    thickness_exaggeration: float = 1.6,
    show_uncertainty: bool = True,
    show_wavelength_sweep: bool = True,
    show_rays: bool = True,
    height: int = 560,
    key: str | None = None,
) -> None:
    if not states:
        return

    component_id = key or f"thinfilm-viz-{uuid.uuid4().hex[:8]}"
    payload = {
        "layout": layout,
        "animationSpeed": float(animation_speed),
        "thicknessExaggeration": float(thickness_exaggeration),
        "showUncertainty": bool(show_uncertainty),
        "showWavelengthSweep": bool(show_wavelength_sweep),
        "showRays": bool(show_rays),
        "states": states,
    }
    html = _build_component_html(component_id, payload)
    components.html(html, height=height, scrolling=False)



def _build_component_html(component_id: str, payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload)

    template = r'''
<div class="tfv-root">
  <style>
    .tfv-root {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #e5eef7;
      width: 100%;
    }
    .tfv-shell {
      background: radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 32%),
                  linear-gradient(180deg, #0f172a 0%, #111827 45%, #0b1120 100%);
      border-radius: 20px;
      padding: 16px 16px 10px 16px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 50px rgba(15, 23, 42, 0.28);
      border: 1px solid rgba(148,163,184,0.16);
    }
    .tfv-topline {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .tfv-title {
      font-size: 1.02rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 4px;
    }
    .tfv-subtitle {
      font-size: 0.82rem;
      color: #cbd5e1;
      max-width: 44rem;
      line-height: 1.45;
    }
    .tfv-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .tfv-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.42);
      border: 1px solid rgba(148,163,184,0.20);
      font-size: 0.78rem;
      color: #e2e8f0;
    }
    .tfv-chip-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      flex: 0 0 auto;
    }
    .tfv-canvas {
      width: 100%;
      height: 420px;
      display: block;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(30,41,59,0.38), rgba(2,6,23,0.75));
    }
    .tfv-foot {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 12px;
      margin-top: 10px;
      align-items: start;
    }
    .tfv-note {
      font-size: 0.78rem;
      color: #cbd5e1;
      line-height: 1.55;
    }
    .tfv-note strong {
      color: #f8fafc;
    }
    .tfv-mini {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      flex-wrap: wrap;
    }
    .tfv-pill {
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.73rem;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(30,41,59,0.55);
      color: #dbeafe;
    }
  </style>
  <div class="tfv-shell">
    <div class="tfv-topline">
      <div>
        <div class="tfv-title">Thin-Film Digital Twin</div>
        <div class="tfv-subtitle">
          A qualitative 3D encoding of the inferred film state. Height tracks thickness,
          hue shifts with refractive index, darkening tracks absorption, and the translucent
          shell visualizes uncertainty on the neural estimate.
        </div>
      </div>
      <div class="tfv-chip-row" id="__ID___chips"></div>
    </div>
    <canvas id="__ID___canvas" class="tfv-canvas"></canvas>
    <div class="tfv-foot">
      <div class="tfv-note">
        <strong>Visual mapping.</strong> Height = thickness. Warmer color = larger refractive index.
        Darker slab = larger extinction coefficient. Animated wavefront and wavelength sweep give a compact
        intuition for how the spectrum is being interpreted by the model.
      </div>
      <div class="tfv-mini">
        <div class="tfv-pill" id="__ID___wl">λ 400 nm</div>
        <div class="tfv-pill" id="__ID___mode"></div>
      </div>
    </div>
  </div>
</div>
<script>
(() => {
  const payload = __PAYLOAD__;
  const canvas = document.getElementById("__ID___canvas");
  const chips = document.getElementById("__ID___chips");
  const wlLabel = document.getElementById("__ID___wl");
  const modeLabel = document.getElementById("__ID___mode");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  modeLabel.textContent = `${payload.layout} view`;

  payload.states.forEach((state) => {
    const chip = document.createElement("div");
    chip.className = "tfv-chip";
    chip.innerHTML = `<span class="tfv-chip-dot" style="background:${state.palette.accent}"></span>${state.label}`;
    chips.appendChild(chip);
  });

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(2, Math.floor(rect.width * dpr));
    canvas.height = Math.max(2, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function parseRgba(value, fallbackAlpha = 1.0) {
    if (!value || typeof value !== "string") {
      return {r: 255, g: 255, b: 255, a: fallbackAlpha};
    }
    const match = value.match(/rgba?\(([^)]+)\)/i);
    if (!match) {
      return {r: 255, g: 255, b: 255, a: fallbackAlpha};
    }
    const parts = match[1].split(",").map((x) => Number(x.trim()));
    return {
      r: parts[0] || 255,
      g: parts[1] || 255,
      b: parts[2] || 255,
      a: Number.isFinite(parts[3]) ? parts[3] : fallbackAlpha,
    };
  }

  function rgba(c, alpha) {
    const a = alpha === undefined ? c.a : alpha;
    return `rgba(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)}, ${a})`;
  }

  function shift(color, delta) {
    return {
      r: Math.max(0, Math.min(255, color.r + delta)),
      g: Math.max(0, Math.min(255, color.g + delta)),
      b: Math.max(0, Math.min(255, color.b + delta)),
      a: color.a,
    };
  }

  function project(x, y, z, originX, originY) {
    return {
      x: originX + (x - y) * 0.92,
      y: originY + (x + y) * 0.44 - z,
    };
  }

  function polygon(points, fill, stroke = null, lineWidth = 1.0) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }

  function drawCuboid(originX, originY, dims, fills, opts = {}) {
    const x0 = 0;
    const y0 = 0;
    const x1 = dims.w;
    const y1 = dims.d;
    const z1 = dims.h;

    const top = [
      project(x0, y0, z1, originX, originY),
      project(x1, y0, z1, originX, originY),
      project(x1, y1, z1, originX, originY),
      project(x0, y1, z1, originX, originY),
    ];
    const front = [
      project(x0, y0, 0, originX, originY),
      project(x1, y0, 0, originX, originY),
      project(x1, y0, z1, originX, originY),
      project(x0, y0, z1, originX, originY),
    ];
    const side = [
      project(x1, y0, 0, originX, originY),
      project(x1, y1, 0, originX, originY),
      project(x1, y1, z1, originX, originY),
      project(x1, y0, z1, originX, originY),
    ];

    if (opts.glow) {
      ctx.save();
      ctx.shadowBlur = opts.glow.blur || 24;
      ctx.shadowColor = opts.glow.color || fills.top;
      polygon(top, fills.top);
      ctx.restore();
    }
    polygon(side, fills.side, opts.stroke || "rgba(255,255,255,0.16)", opts.lineWidth || 1.0);
    polygon(front, fills.front, opts.stroke || "rgba(255,255,255,0.16)", opts.lineWidth || 1.0);
    polygon(top, fills.top, opts.stroke || "rgba(255,255,255,0.16)", opts.lineWidth || 1.0);

    return {
      top,
      front,
      side,
      labelAnchor: top[2],
      centerTop: {
        x: (top[0].x + top[2].x) / 2,
        y: (top[0].y + top[2].y) / 2,
      },
      dims,
    };
  }

  function drawWireCuboid(originX, originY, dims, stroke) {
    const pts = {
      a: project(0, 0, 0, originX, originY),
      b: project(dims.w, 0, 0, originX, originY),
      c: project(dims.w, dims.d, 0, originX, originY),
      d: project(0, dims.d, 0, originX, originY),
      e: project(0, 0, dims.h, originX, originY),
      f: project(dims.w, 0, dims.h, originX, originY),
      g: project(dims.w, dims.d, dims.h, originX, originY),
      h: project(0, dims.d, dims.h, originX, originY),
    };
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.15;
    ctx.setLineDash([6, 5]);
    [["a","b","c","d","a"], ["e","f","g","h","e"], ["a","e"], ["b","f"], ["c","g"], ["d","h"]].forEach((path) => {
      ctx.beginPath();
      ctx.moveTo(pts[path[0]].x, pts[path[0]].y);
      for (let i = 1; i < path.length; i += 1) {
        ctx.lineTo(pts[path[i]].x, pts[path[i]].y);
      }
      ctx.stroke();
    });
    ctx.restore();
  }

  function filmHeight(state) {
    const norm = Math.max(0, Math.min(1, (state.thickness - 10.0) / 290.0));
    return 10 + (22 + norm * 62) * payload.thicknessExaggeration;
  }

  function currentWavelength(t) {
    return 400 + ((Math.sin(t * 0.00065 * payload.animationSpeed) + 1) * 0.5) * 400;
  }

  function spectralColor(lambdaNm, alpha = 1.0) {
    const clamped = Math.max(400, Math.min(800, lambdaNm));
    const t = (clamped - 400) / 400;
    const hue = 260 - t * 260;
    return `hsla(${hue}, 92%, 68%, ${alpha})`;
  }

  function drawWave(originX, originY, width, lambdaNm, phase, amplitude = 10) {
    ctx.save();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = spectralColor(lambdaNm, 0.95);
    ctx.beginPath();
    for (let i = 0; i <= width; i += 5) {
      const x = originX + i;
      const y = originY + Math.sin((i / width) * 10.0 - phase) * amplitude;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawArrow(x0, y0, x1, y1, color, width = 2.0) {
    const angle = Math.atan2(y1 - y0, x1 - x0);
    const head = 9;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(angle - Math.PI / 6), y1 - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x1 - head * Math.cos(angle + Math.PI / 6), y1 - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawLabel(x, y, label, subtitle, accent) {
    const paddingX = 10;
    const paddingY = 8;
    ctx.save();
    ctx.font = "600 13px Inter, sans-serif";
    const w = Math.max(ctx.measureText(label).width, ctx.measureText(subtitle).width) + paddingX * 2;
    const h = 38;
    ctx.fillStyle = "rgba(15, 23, 42, 0.82)";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.0;
    const left = x - w / 2;
    const top = y - h / 2;
    roundRect(left, top, w, h, 12);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(label, left + paddingX, top + 15);
    ctx.font = "500 11px Inter, sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(subtitle, left + paddingX, top + 29);
    ctx.restore();
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawSpectrumBar(width, height, lambdaNm) {
    const x = 26;
    const y = height - 38;
    const barW = Math.max(180, Math.min(280, width * 0.30));
    const barH = 14;
    const grad = ctx.createLinearGradient(x, 0, x + barW, 0);
    [400, 470, 530, 590, 650, 800].forEach((wl) => {
      grad.addColorStop((wl - 400) / 400, spectralColor(wl, 1.0));
    });
    ctx.save();
    ctx.fillStyle = "rgba(15,23,42,0.55)";
    roundRect(x - 10, y - 16, barW + 20, 42, 14);
    ctx.fill();
    ctx.fillStyle = grad;
    roundRect(x, y, barW, barH, 8);
    ctx.fill();
    const cursorX = x + ((lambdaNm - 400) / 400) * barW;
    ctx.strokeStyle = "rgba(248,250,252,0.92)";
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(cursorX, y - 4);
    ctx.lineTo(cursorX, y + barH + 4);
    ctx.stroke();
    ctx.fillStyle = "rgba(248,250,252,0.90)";
    ctx.font = "600 12px Inter, sans-serif";
    ctx.fillText(`${Math.round(lambdaNm)} nm`, x, y - 6);
    ctx.restore();
  }

  function drawSubstrate(originX, originY, widthScale = 1.0) {
    const dims = {w: 190 * widthScale, d: 114 * widthScale, h: 28};
    drawCuboid(originX, originY, dims, {
      top: "rgba(71, 85, 105, 0.85)",
      front: "rgba(51, 65, 85, 0.98)",
      side: "rgba(30, 41, 59, 0.98)",
    }, {stroke: "rgba(226,232,240,0.13)"});
    return dims;
  }

  function drawSingleScene(stateList, cx, cy, panelWidth, panelHeight, t, overlayMode) {
    const substrate = drawSubstrate(cx - 96, cy + 76);
    const lambdaNm = currentWavelength(t);
    const wavePhase = t * 0.0085 * payload.animationSpeed;

    if (payload.showRays) {
      drawWave(cx - 165, cy - 64, 110, lambdaNm, wavePhase, 8);
      drawArrow(cx - 62, cy - 58, cx + 5, cy - 6, spectralColor(lambdaNm, 0.85), 2.0);
      drawArrow(cx + 22, cy - 15, cx - 12, cy - 44, "rgba(248,250,252,0.55)", 1.7);
    }

    const offsets = overlayMode
      ? [[-16, -4], [0, 0], [18, 7], [34, 13]]
      : [[0, 0]];

    const visibleStates = overlayMode ? stateList : stateList.slice(0, 1);
    visibleStates.forEach((state, index) => {
      const fill = {
        top: state.palette.top,
        front: state.palette.front,
        side: state.palette.side,
      };
      const height = filmHeight(state);
      const offset = offsets[index] || [0, 0];
      const dims = {w: substrate.w - 16, d: substrate.d - 14, h: height};
      const film = drawCuboid(
        cx - 88 + offset[0],
        cy + 72 + offset[1],
        dims,
        fill,
        {
          stroke: "rgba(255,255,255,0.22)",
          glow: {color: state.palette.glow, blur: state.emphasis ? 34 : 18},
          lineWidth: state.emphasis ? 1.2 : 0.95,
        }
      );
      if (payload.showUncertainty && state.ci_thickness && state.ci_thickness > 0) {
        const shellHeight = Math.max(8, height + Math.min(40, state.ci_thickness * 0.9));
        drawWireCuboid(
          cx - 88 + offset[0],
          cy + 72 + offset[1],
          {w: dims.w, d: dims.d, h: shellHeight},
          state.palette.shell
        );
      }
      const subtitle = `d ${state.thickness.toFixed(1)} nm · n ${state.n.toFixed(3)} · k ${state.k.toFixed(3)}`;
      drawLabel(film.centerTop.x, film.centerTop.y - height - 24, state.label, subtitle, state.palette.accent);
    });

    ctx.save();
    ctx.fillStyle = "rgba(226,232,240,0.75)";
    ctx.font = "500 12px Inter, sans-serif";
    ctx.fillText("Si substrate", cx - 24, cy + 144);
    ctx.restore();

    drawSpectrumBar(panelWidth, panelHeight, lambdaNm);
    wlLabel.textContent = `λ ${Math.round(lambdaNm)} nm`;
  }

  function drawBackground(width, height) {
    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, "rgba(15,23,42,0.40)");
    bg.addColorStop(1, "rgba(2,6,23,0.10)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.strokeStyle = "rgba(148,163,184,0.08)";
    ctx.lineWidth = 1.0;
    for (let i = 0; i < 8; i += 1) {
      const y = 40 + i * 42;
      ctx.beginPath();
      ctx.moveTo(16, y);
      ctx.lineTo(width - 16, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawPanels(width, height, t) {
    drawBackground(width, height);
    const states = payload.states;
    if (payload.layout === "Side-by-side") {
      const panels = states.slice(0, 3);
      const gutter = 14;
      const panelWidth = (width - gutter * (panels.length - 1)) / panels.length;
      panels.forEach((state, i) => {
        const x = i * (panelWidth + gutter);
        ctx.save();
        ctx.fillStyle = "rgba(15,23,42,0.32)";
        roundRect(x + 2, 8, panelWidth - 4, height - 18, 18);
        ctx.fill();
        ctx.restore();
        drawSingleScene([state], x + panelWidth / 2, height * 0.39, panelWidth, height, t, false);
      });
    } else {
      drawSingleScene(payload.states, width * 0.53, height * 0.42, width, height, t, true);
    }
  }

  function frame(t) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    drawPanels(width, height, t);
    requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);
})();
</script>
'''
    return template.replace("__PAYLOAD__", payload_json).replace("__ID__", component_id)
