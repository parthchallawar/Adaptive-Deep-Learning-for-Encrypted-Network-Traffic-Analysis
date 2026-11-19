/* Tiny SVG charts for the story page and the console. No dependencies, so the demo works
   offline. Charts.line and Charts.bars draw into an <svg> (with a viewBox) or into a <div>. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function surface(target, w, h) {
    let svg = target;
    if (target.tagName.toLowerCase() !== "svg") {
      target.innerHTML = "";
      svg = el("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet" }, target);
    } else {
      const vb = target.viewBox.baseVal;
      if (vb && vb.width) { w = vb.width; h = vb.height; }
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }
    return { svg, w, h };
  }

  function niceTicks(min, max, n) {
    const span = max - min || 1;
    const step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  const tip = (() => {
    let t = null;
    return {
      show(html, x, y) {
        if (!t) { t = document.createElement("div"); t.className = "tip"; document.body.appendChild(t); }
        t.innerHTML = html;
        t.style.left = Math.min(window.innerWidth - 220, x + 14) + "px";
        t.style.top = y + 14 + "px";
        t.style.opacity = 1;
      },
      hide() { if (t) t.style.opacity = 0; },
    };
  })();

  function frame(target, o) {
    const { svg, w, h } = surface(target, o.width || 720, o.height || 260);
    const m = Object.assign({ l: 46, r: 16, t: 14, b: 32 }, o.margin || {});
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const defs = el("defs", {}, svg);
    return { svg, w, h, m, pw, ph, defs };
  }

  function axes(f, xs, ys, o) {
    const g = el("g", { class: "grid" }, f.svg);
    const ax = el("g", { class: "ax" }, f.svg);
    const yt = o.y.ticks || niceTicks(ys.min, ys.max, 4);
    yt.forEach((v) => {
      const y = ys.map(v);
      el("line", { x1: f.m.l, x2: f.m.l + f.pw, y1: y, y2: y }, g);
      const t = el("text", { x: f.m.l - 8, y: y + 4, "text-anchor": "end" }, ax);
      t.textContent = (o.y.fmt || String)(v);
    });
    const xt = o.x.ticks || niceTicks(xs.min, xs.max, Math.max(3, Math.floor(f.pw / 90)));
    xt.forEach((v) => {
      const t = el("text", { x: xs.map(v), y: f.m.t + f.ph + 20, "text-anchor": "middle" }, ax);
      t.textContent = (o.x.fmt || String)(v);
    });
    el("line", { x1: f.m.l, x2: f.m.l + f.pw, y1: f.m.t + f.ph, y2: f.m.t + f.ph }, ax);
    if (o.x.label) { const t = el("text", { x: f.m.l + f.pw, y: f.h - 2, "text-anchor": "end" }, ax); t.textContent = o.x.label; }
    if (o.y.label) { const t = el("text", { x: f.m.l, y: f.m.t - 2 + 0, "text-anchor": "start", dy: -2 }, ax); t.textContent = o.y.label; }
  }

  function scale(d0, d1, r0, r1) {
    const s = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    return { min: d0, max: d1, map: s };
  }

  function extent(series, i, pad) {
    let lo = Infinity, hi = -Infinity;
    series.forEach((s) => s.points.forEach((p) => { if (Number.isFinite(p[i])) { lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]); } }));
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    const p = (hi - lo) * (pad || 0);
    return [lo - p, hi + p];
  }

  /* line({series:[{name,color,points,dash,width,area,step}], x:{min,max,fmt,label,ticks}, y:{...},
           hlines:[{y,color,label,dash}], vlines:[{x,color,label}], markers:[{x,y,color,r,label}]}) */
  function line(target, o) {
    o = Object.assign({ x: {}, y: {} }, o);
    const f = frame(target, o);
    const series = (o.series || []).filter((s) => s.points && s.points.length);
    const [x0, x1] = [o.x.min ?? extent(series, 0)[0], o.x.max ?? extent(series, 0)[1]];
    const ye = extent(series.concat((o.hlines || []).map((h) => ({ points: [[0, h.y]] }))), 1, 0.08);
    const [y0, y1] = [o.y.min ?? ye[0], o.y.max ?? ye[1]];
    const xs = scale(x0, x1, f.m.l, f.m.l + f.pw);
    const ys = scale(y0, y1, f.m.t + f.ph, f.m.t);
    axes(f, xs, ys, o);
    (o.bands || []).forEach((b) => {
      el("rect", { x: xs.map(b.x0), y: f.m.t, width: Math.max(0, xs.map(b.x1) - xs.map(b.x0)), height: f.ph, fill: b.color, opacity: b.opacity ?? 0.08 }, f.svg);
      if (b.label) { const t = el("text", { x: xs.map(b.x0) + 6, y: f.m.t + f.ph - 8, fill: b.text || "#98A2CC" }, f.svg); t.textContent = b.label; }
    });
    (o.hlines || []).forEach((h) => {
      const y = ys.map(h.y);
      el("line", { x1: f.m.l, x2: f.m.l + f.pw, y1: y, y2: y, stroke: h.color, "stroke-dasharray": h.dash || "5 5", "stroke-width": 1.4 }, f.svg);
      if (h.label) { const t = el("text", { x: f.m.l + f.pw - 4, y: y - 6, "text-anchor": "end", fill: h.color }, f.svg); t.textContent = h.label; }
    });
    const clipId = `c${Math.random().toString(36).slice(2)}`;
    el("rect", { x: f.m.l, y: f.m.t - 4, width: f.pw, height: f.ph + 8 }, el("clipPath", { id: clipId }, f.defs));
    const plot = el("g", { "clip-path": `url(#${clipId})` }, f.svg);
    series.forEach((s, i) => {
      const pts = s.points.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
      if (!pts.length) return;
      let d = "";
      pts.forEach((p, j) => {
        const X = xs.map(p[0]), Y = ys.map(p[1]);
        if (j === 0) d += `M${X},${Y}`;
        else if (s.step) d += `H${X}V${Y}`;
        else d += `L${X},${Y}`;
      });
      if (s.area) {
        const gid = `g${Math.random().toString(36).slice(2)}`;
        const lg = el("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, f.defs);
        el("stop", { offset: "0%", "stop-color": s.color, "stop-opacity": 0.32 }, lg);
        el("stop", { offset: "100%", "stop-color": s.color, "stop-opacity": 0 }, lg);
        el("path", { d: d + `L${xs.map(pts[pts.length - 1][0])},${f.m.t + f.ph}L${xs.map(pts[0][0])},${f.m.t + f.ph}Z`, fill: `url(#${gid})` }, plot);
      }
      const p = el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || null }, plot);
      if (s.glow) p.setAttribute("style", `filter: drop-shadow(0 0 6px ${s.color})`);
      if (s.dots) pts.forEach((q) => el("circle", { cx: xs.map(q[0]), cy: ys.map(q[1]), r: 2.6, fill: s.color }, plot));
      if (o.animate) {
        const len = p.getTotalLength ? p.getTotalLength() : 0;
        if (len) { p.style.strokeDasharray = len; p.style.strokeDashoffset = len; p.getBoundingClientRect(); p.style.transition = `stroke-dashoffset 1.6s cubic-bezier(.22,1,.36,1) ${i * 0.15}s`; p.style.strokeDashoffset = 0; }
      }
    });
    (o.vlines || []).forEach((v) => {
      const x = xs.map(v.x);
      el("line", { x1: x, x2: x, y1: f.m.t, y2: f.m.t + f.ph, stroke: v.color, "stroke-width": v.width || 1.4, "stroke-dasharray": v.dash || null }, f.svg);
      if (v.label) { const t = el("text", { x: x + 6, y: v.top ? f.m.t + 12 : f.m.t + f.ph - 6, fill: v.color }, f.svg); t.textContent = v.label; }
    });
    (o.markers || []).forEach((mk) => {
      if (!Number.isFinite(mk.y)) return;
      el("circle", { cx: xs.map(mk.x), cy: ys.map(mk.y), r: (mk.r || 6) + 6, fill: mk.color, opacity: 0.18 }, f.svg);
      el("circle", { cx: xs.map(mk.x), cy: ys.map(mk.y), r: mk.r || 6, fill: mk.color, stroke: "#04060E", "stroke-width": 2 }, f.svg);
      if (mk.label) { const t = el("text", { x: xs.map(mk.x) + 12, y: ys.map(mk.y) - 10, fill: mk.color }, f.svg); t.textContent = mk.label; }
    });
    if (o.tooltip !== false && series.length) hover(f, xs, ys, series, o);
    return { xs, ys, f };
  }

  function hover(f, xs, ys, series, o) {
    const guide = el("line", { y1: f.m.t, y2: f.m.t + f.ph, stroke: "rgba(233,237,255,.25)", opacity: 0 }, f.svg);
    const hit = el("rect", { x: f.m.l, y: f.m.t, width: f.pw, height: f.ph, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (e) => {
      const r = f.svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * f.w;
      const xv = xs.min + ((px - f.m.l) / f.pw) * (xs.max - xs.min);
      let best = null;
      series.forEach((s) => s.points.forEach((p) => { if (!best || Math.abs(p[0] - xv) < Math.abs(best - xv)) best = p[0]; }));
      if (best === null) return;
      guide.setAttribute("x1", xs.map(best)); guide.setAttribute("x2", xs.map(best)); guide.setAttribute("opacity", 1);
      const fx = o.x.tipFmt || o.x.fmt || ((v) => v);
      const fy = o.y.tipFmt || o.y.fmt || ((v) => (Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1)));
      const rows = series.map((s) => {
        const p = s.points.find((q) => q[0] === best);
        return p && Number.isFinite(p[1]) ? `<div><i style="display:inline-block;width:10px;height:3px;background:${s.color};margin-right:6px;vertical-align:middle"></i>${s.name || ""} <b>${fy(p[1])}</b></div>` : "";
      }).join("");
      tip.show(`<div style="color:#98A2CC">${o.x.tipLabel || ""}${fx(best)}</div>${rows}`, e.clientX, e.clientY);
    });
    hit.addEventListener("mouseleave", () => { guide.setAttribute("opacity", 0); tip.hide(); });
  }

  /* bars({keys:[...], groups:[{name,color,values:{key:v}}], x:{fmt,label}, y:{fmt,label}}) */
  function bars(target, o) {
    o = Object.assign({ x: {}, y: {} }, o);
    const f = frame(target, o);
    const keys = o.keys;
    let hi = 0;
    o.groups.forEach((g) => keys.forEach((k) => { hi = Math.max(hi, g.values[k] || 0); }));
    const ys = scale(0, o.y.max ?? (hi || 1) * 1.1, f.m.t + f.ph, f.m.t);
    const band = f.pw / Math.max(1, keys.length);
    const xs = { min: 0, max: keys.length, map: (i) => f.m.l + band * (i + 0.5) };
    axes(f, xs, ys, Object.assign({}, o, { x: Object.assign({}, o.x, { ticks: keys.map((_, i) => i).filter((i) => keys.length < 16 || i % 2 === 0), fmt: (i) => (o.x.fmt || String)(keys[i]) }) }));
    const bw = Math.max(2, (band * 0.72) / o.groups.length);
    o.groups.forEach((g, gi) => keys.forEach((k, i) => {
      const v = g.values[k] || 0;
      const x = f.m.l + band * i + band * 0.14 + gi * bw;
      const r = el("rect", { x, y: ys.map(v), width: bw - 1, height: Math.max(0, f.m.t + f.ph - ys.map(v)), rx: 3, fill: g.color, opacity: g.opacity ?? 0.9 }, f.svg);
      r.addEventListener("mousemove", (e) => tip.show(`${g.name}: <b>${(o.y.fmt || String)(v)}</b><div style="color:#98A2CC">${(o.x.tipLabel || "")}${(o.x.fmt || String)(k)}</div>`, e.clientX, e.clientY));
      r.addEventListener("mouseleave", tip.hide);
    }));
    return { f, ys };
  }

  /* A legend above a chart: before ``target`` when it is the chart's <svg>, else at the top of
     the container. Re-rendering replaces the previous legend. */
  function legend(target, items) {
    const d = document.createElement("div");
    d.className = "c-legend";
    d.innerHTML = items.map((i) => `<span><i style="background:${i.color}"></i>${i.name}</span>`).join("");
    if (target.tagName.toLowerCase() === "svg") {
      const prev = target.previousElementSibling;
      if (prev && prev.classList.contains("c-legend")) prev.remove();
      target.before(d);
    } else target.prepend(d);
  }

  window.Charts = { line, bars, legend, el, niceTicks, tip };
})();
