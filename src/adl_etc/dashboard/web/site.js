/* The story page: the packet flow field, scroll reveals, parallax, the decision scrubber, the
   budget simulator and the offline result charts. The console lives in console.js. */
(function () {
  "use strict";
  const C = { cyan: "#5BE3E0", violet: "#8C7BFF", rose: "#FF6F91", amber: "#F5C46B", text: "#E9EDFF", muted: "#98A2CC", faint: "#5F6A97" };
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = window.matchMedia("(pointer: fine)").matches;
  const DATA = window.ANYTIME_DATA || {};
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const mouse = { x: -9999, y: -9999, nx: 0, ny: 0 };

  /* ---------------------------------------------------------------- the hero flow field */
  // Each lane is one encrypted flow. Packets arrive as bars (up: client to server, down: back),
  // and the flow stops the moment it is decided: cyan when named, rose when flagged unknown.
  function flowField() {
    const cv = $("#flowfield");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let W = 0, H = 0, dpr = 1, lanes = [], visible = true, last = performance.now();
    const LANE_GAP = 26;

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.max(8, Math.floor(H / LANE_GAP));
      lanes = Array.from({ length: n }, (_, i) => newFlow(i, true));
    }

    function sampleK() { // packets until decision: roughly the measured spread (mean about 5.5)
      const r = Math.random();
      return Math.max(2, Math.min(30, Math.round(2 + (-Math.log(1 - r) * 3.6))));
    }

    function newFlow(i, warm) {
      const k = sampleK();
      const reject = Math.random() < 0.11;
      const speed = 110 + Math.random() * 120; // px per second: a decided flow is a short train
      // most flows start right of the headline, where they can be seen; some anywhere
      const x0 = Math.random() < 0.75 ? W * (0.42 + Math.random() * 0.5) : Math.random() * W * 0.9;
      return {
        y: (i + 0.5) * (H / Math.max(1, Math.floor(H / LANE_GAP))),
        x0: warm ? x0 - Math.random() * 120 : x0,
        speed, k, reject, packets: [], sent: 0, next: 0, decided: false, fade: 1,
        gap: 0.12 + Math.random() * 0.22,
        t: 0,
      };
    }

    function step(dt) {
      lanes.forEach((f, i) => {
        f.t += dt;
        const head = f.x0 + f.t * f.speed;
        if (!f.decided && f.t >= f.next) {
          const fwd = f.sent === 0 || Math.random() < 0.42;
          const size = fwd ? 0.15 + Math.random() * 0.45 : 0.35 + Math.random() * 0.65;
          f.packets.push({ x: head, dir: fwd ? -1 : 1, size, born: f.t });
          f.sent += 1;
          f.next = f.t + f.gap * (0.4 + Math.random() * 1.2);
          if (f.sent >= f.k) { f.decided = true; f.decidedAt = f.t; f.dx = head; }
        }
        if (f.decided) f.fade -= dt * 0.32;
        if (f.fade <= 0 || f.x0 + f.t * f.speed > W + 60) lanes[i] = newFlow(i, false);
      });
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const amp = Math.min(11, LANE_GAP * 0.42);
      lanes.forEach((f) => {
        const color = f.decided ? (f.reject ? C.rose : C.cyan) : "rgba(152,162,204,";
        // the lane's thread
        ctx.globalAlpha = 0.08 * f.fade;
        ctx.strokeStyle = f.decided ? color : "#98A2CC";
        ctx.lineWidth = 1;
        ctx.beginPath();
        const first = f.packets.length ? f.packets[0].x : 0;
        const lastX = f.decided ? f.dx : f.x0 + f.t * f.speed;
        ctx.moveTo(first, f.y); ctx.lineTo(lastX, f.y); ctx.stroke();
        f.packets.forEach((p) => {
          const dx = p.x - mouse.x, dy = f.y - mouse.y;
          const near = fine ? Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / 170) : 0;
          const h = amp * p.size * (1 + near * 0.9);
          const age = Math.min(1, (f.t - p.born) * 4);
          ctx.globalAlpha = (f.decided ? 0.95 : 0.34 + near * 0.5) * f.fade * age;
          ctx.fillStyle = f.decided ? color : `rgba(152,162,204,1)`;
          const w = 2.2;
          if (p.dir < 0) ctx.fillRect(p.x - w / 2, f.y - h, w, h);
          else ctx.fillRect(p.x - w / 2, f.y, w, h);
        });
        if (f.decided) { // the decision: a ring that blooms where the flow stopped
          const age = f.t - f.decidedAt;
          const r = 3 + Math.min(1, age * 2.2) * 9;
          ctx.globalAlpha = Math.max(0, 1 - age * 0.9) * f.fade;
          ctx.strokeStyle = color; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(f.dx + 8, f.y, r, 0, Math.PI * 2); ctx.stroke();
          ctx.globalAlpha = 0.9 * f.fade;
          ctx.fillStyle = color;
          ctx.shadowColor = color; ctx.shadowBlur = 12;
          ctx.beginPath(); ctx.arc(f.dx + 8, f.y, 2.6, 0, Math.PI * 2); ctx.fill();
          ctx.shadowBlur = 0;
        }
      });
      ctx.globalAlpha = 1;
    }

    function loop(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (visible) { step(dt); draw(); }
      requestAnimationFrame(loop);
    }

    resize();
    window.addEventListener("resize", () => { clearTimeout(resize._t); resize._t = setTimeout(resize, 150); });
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(cv);
    if (reduced) { for (let i = 0; i < 400; i++) step(0.05); draw(); }
    else requestAnimationFrame(loop);
  }

  /* --------------------------------------------------------------- pointer: light, tilt, magnet */
  function pointer() {
    const light = $(".cursor-light");
    let raf = 0;
    window.addEventListener("pointermove", (e) => {
      mouse.x = e.clientX; mouse.y = e.clientY;
      mouse.nx = e.clientX / window.innerWidth - 0.5; mouse.ny = e.clientY / window.innerHeight - 0.5;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        if (light) { light.style.setProperty("--mx", mouse.x + "px"); light.style.setProperty("--my", mouse.y + "px"); }
        if (!reduced) $$(".hero-chips .chip").forEach((c) => {
          const d = +c.dataset.depth || 10;
          c.style.translate = `${-mouse.nx * d}px ${-mouse.ny * d}px`;
        });
      });
    }, { passive: true });
    if (!fine || reduced) return;
    $$(".tilt").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform = `perspective(700px) rotateY(${x * 10}deg) rotateX(${-y * 10}deg)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
    $$(".magnetic").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        el.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * 0.22}px, ${(e.clientY - r.top - r.height / 2) * 0.3}px)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  }

  /* ------------------------------------------------------------------- scrolling: reveal, depth */
  function scrolling() {
    const nav = $("#nav");
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const sibs = $$(".reveal", e.target.parentElement).filter((s) => !s.classList.contains("in"));
      e.target.style.transitionDelay = `${Math.min(4, Math.max(0, sibs.indexOf(e.target))) * 90}ms`;
      e.target.classList.add("in");
      io.unobserve(e.target);
    }), { threshold: 0.14, rootMargin: "0px 0px -6% 0px" });
    $$(".reveal").forEach((el) => io.observe(el));

    const links = $$(".nav-links a");
    const sections = links.map((a) => $(a.getAttribute("href"))).filter(Boolean);
    const layers = $$(".stack .layer");
    const aurora = $(".aurora");
    let ticking = false;
    function onScroll() {
      ticking = false;
      const y = window.scrollY;
      nav.classList.toggle("scrolled", y > 40);
      let cur = null;
      sections.forEach((s) => { if (s.getBoundingClientRect().top < window.innerHeight * 0.4) cur = s.id; });
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + cur));
      if (reduced) return;
      if (aurora) aurora.style.transform = `translate3d(0, ${y * -0.06}px, 0)`;
      const stack = $("#stack");
      if (stack) {
        const r = stack.getBoundingClientRect();
        const prog = (window.innerHeight / 2 - (r.top + r.height / 2)) / window.innerHeight;
        layers.forEach((l) => {
          const d = +l.dataset.depth || 0;
          l.style.transform = `translate3d(${prog * d * 260}px, ${prog * d * -420}px, 0)`;
        });
      }
    }
    window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
    onScroll();
  }

  /* ------------------------------------------------------------------- the flowprint (vision) */
  function flowprint() {
    const svg = $("#flowprint");
    if (!svg) return;
    // a TLS-like opening: ClientHello, the server's certificate flight, then request and reply
    const pk = [[0, 1, 517], [24, -1, 1460], [1, -1, 1460], [1, -1, 1176], [31, 1, 80], [2, 1, 64], [18, -1, 255], [44, 1, 412], [37, -1, 1460], [0, -1, 1460], [1, -1, 1460], [2, -1, 986], [9, 1, 41], [61, 1, 330], [40, -1, 1460], [1, -1, 731]];
    const W = 560, H = 300, mid = H / 2, max = 1500;
    const xs = []; let x = 20;
    pk.forEach((p, i) => { x += i === 0 ? 0 : 14 + Math.log1p(p[0]) * 9; xs.push(x); });
    const sx = (W - 40) / (x - 20 || 1);
    const note = $("#fp-note");
    Charts.el("line", { x1: 10, x2: W - 10, y1: mid, y2: mid, stroke: "rgba(120,140,220,.3)" }, svg);
    const rects = pk.map((p, i) => {
      const h = (p[2] / max) * (mid - 18);
      const X = 20 + (xs[i] - 20) * sx;
      const r = Charts.el("rect", { x: X - 6, y: p[1] > 0 ? mid - h : mid, width: 12, height: h, rx: 3, fill: p[1] > 0 ? C.cyan : C.violet, opacity: 0.85 }, svg);
      if (!reduced) { r.style.transformOrigin = `${X}px ${mid}px`; r.style.transform = "scaleY(0)"; r.style.transition = `transform .9s cubic-bezier(.22,1,.36,1) ${i * 0.05}s`; }
      const label = `Packet ${i + 1}: ${p[2].toLocaleString()} bytes ${p[1] > 0 ? "to the server" : "from the server"}${i ? `, ${p[0]} ms after the last` : ""}`;
      r.setAttribute("tabindex", "0");
      r.setAttribute("aria-label", label);
      const on = () => { note.textContent = label; rects.forEach((o) => o.classList.toggle("dim", o !== r)); };
      r.addEventListener("pointerenter", on); r.addEventListener("focus", on);
      r.addEventListener("pointerleave", () => { note.textContent = "Hover a packet"; rects.forEach((o) => o.classList.remove("dim")); });
      return r;
    });
    new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { rects.forEach((r) => { r.style.transform = "scaleY(1)"; }); o.disconnect(); } }, { threshold: 0.3 }).observe(svg);
  }

  /* ------------------------------------------------------------------- the decision scrubber */
  const EXAMPLES = {
    known: {
      source: "Illustrative curves shaped like the measured ones: a familiar app, named after 6 packets.",
      p: [0.38, 0.58, 0.71, 0.82, 0.9, 0.946, 0.955, 0.962, 0.97, 0.972, 0.975, 0.978],
      u: [-1.6, -1.2, -1.35, -1.1, -1.25, -1.3, -1.2, -1.15, -1.3, -1.25, -1.2, -1.3],
      thetaC: 0.935, thetaR: Array(12).fill(1.4),
    },
    unknown: {
      source: "Illustrative curves: an app held out of training. It looks unfamiliar from packet 3 and is flagged at packet 4.",
      p: [0.3, 0.44, 0.52, 0.57, 0.6, 0.63, 0.66, 0.7, 0.72, 0.75, 0.77, 0.79],
      u: [0.2, 0.9, 1.25, 1.85, 2.1, 2.3, 2.2, 2.4, 2.5, 2.45, 2.6, 2.7],
      thetaC: 0.935, thetaR: Array(12).fill(1.4),
    },
  };
  let live = null;

  function decideAt(ex, k) {
    for (let j = 1; j <= k; j++) {
      const p = ex.p[j - 1], u = ex.u[j - 1], tr = ex.thetaR[j - 1];
      if (j >= 3 && u >= tr) return { k: j, rule: "reject" };
      if (j >= 2 && p >= ex.thetaC) return { k: j, rule: "commit" };
      if (j === ex.p.length && ex.final) return { k: j, rule: u >= tr ? "reject" : "commit" };
    }
    return null;
  }

  function scrubber() {
    const range = $("#dc-range");
    if (!range) return;
    let name = "known";
    const ex = () => (name === "live" && live ? live : EXAMPLES[name]);
    function render() {
      const e = ex(), n = e.p.length;
      range.max = n;
      const k = Math.min(+range.value, n);
      range.style.setProperty("--fill", ((k - 1) / Math.max(1, n - 1)) * 100 + "%");
      $("#dc-k").textContent = k;
      $("#dc-source").textContent = e.source;
      const dec = decideAt(e, k);
      const pts = (arr) => arr.slice(0, k).map((v, i) => [i + 1, v]);
      const stop = dec ? [{ x: dec.k, color: dec.rule === "commit" ? C.cyan : C.rose, label: dec.rule === "commit" ? "named here" : "flagged here", dash: "3 4" }] : [];
      Charts.line($("#decide-p"), {
        margin: { l: 40, r: 12, t: 10, b: 26 },
        x: { min: 1, max: n, ticks: Array.from({ length: n }, (_, i) => i + 1) },
        y: { min: 0, max: 1, ticks: [0, 0.5, 1], fmt: (v) => v.toFixed(1) },
        series: [{ name: "confidence", color: C.cyan, points: pts(e.p), width: 2.6, area: true, dots: true, glow: true }],
        hlines: [{ y: e.thetaC, color: C.cyan, label: "commit threshold" }],
        vlines: stop, tooltip: false,
      });
      const uMin = Math.min(...e.u, ...e.thetaR) - 0.4, uMax = Math.max(...e.u, ...e.thetaR) + 0.4;
      Charts.line($("#decide-u"), {
        margin: { l: 40, r: 12, t: 10, b: 26 },
        x: { min: 1, max: n, ticks: Array.from({ length: n }, (_, i) => i + 1) },
        y: { min: uMin, max: uMax, ticks: Charts.niceTicks(uMin, uMax, 3), fmt: (v) => v.toFixed(1) },
        series: [
          { name: "threshold", color: C.rose, points: e.thetaR.map((v, i) => [i + 1, v]), dash: "5 5", width: 1.4 },
          { name: "unknown score", color: C.violet, points: pts(e.u), width: 2.6, dots: true, glow: true },
        ],
        bands: [{ x0: 1, x1: 3, color: "#98A2CC", opacity: 0.05 }],
        vlines: stop, tooltip: false,
      });
      const active = dec ? dec.rule : "wait";
      $$(".rule").forEach((r) => r.classList.toggle("on", r.dataset.rule === active));
    }
    range.addEventListener("input", render);
    $$("#dc-examples button").forEach((b) => b.addEventListener("click", () => {
      name = b.dataset.ex;
      $$("#dc-examples button").forEach((o) => o.setAttribute("aria-checked", String(o === b)));
      range.value = 1; render(); play();
    }));
    let timer = null;
    function play() {
      if (reduced) { range.value = range.max; render(); return; }
      clearInterval(timer);
      range.value = 1; render();
      timer = setInterval(() => {
        const e = ex(); const dec = decideAt(e, e.p.length);
        const stopAt = dec ? Math.min(e.p.length, dec.k + 1) : e.p.length;
        if (+range.value >= stopAt) { clearInterval(timer); return; }
        range.value = +range.value + 1; render();
      }, 520);
    }
    range.addEventListener("pointerdown", () => clearInterval(timer));
    render();
    new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { play(); o.disconnect(); } }, { threshold: 0.45 }).observe(range);
    window.addEventListener("anytime:ready", loadLive, { once: true });
  }

  async function loadLive() { // a real decided flow, once the console has replayed some traffic
    try {
      const r = await fetch("/flows?limit=300");
      if (!r.ok) return;
      const rows = (await r.json()).flows.filter((f) => f.k >= 4 && f.k <= 14 && f.decision !== "undecidable");
      if (!rows.length) return;
      const pick = rows.find((f) => f.decision === "commit") || rows[0];
      const d = await (await fetch(`/flows/${pick.flow_id}`)).json();
      const ex = d.explain || null;
      if (!ex || !ex.p_safe.length) return;
      const n = ex.p_safe.length;
      const thR = (d.theta_reject || []).slice(0, n);
      while (thR.length < n) thR.push(d.decision.u_threshold ?? 1.4);
      live = {
        source: `Flow ${pick.flow_id} from a ${d.true_class ? "replayed week" : "capture"}: ${pick.decision.replace("_", " ")} at packet ${pick.k}${d.true_class ? `, true app ${d.true_class}` : ""}. Thresholds are the ones in force at its decision.`,
        p: ex.p_safe, u: ex.u, thetaC: d.decision.theta_commit, thetaR: thR, final: true,
      };
      const b = $('#dc-examples [data-ex="live"]');
      if (b) b.hidden = false;
    } catch (_) { /* the illustrations stay */ }
  }

  /* ------------------------------------------------------------------- the budget simulator */
  function simulator() {
    const range = $("#sim-k");
    if (!range || !DATA.fronts) return;
    const ss = DATA.fronts.safe_stop.slice().sort((a, b) => a[0] - b[0]);
    const pe = DATA.fronts.p_echo.slice().sort((a, b) => a[0] - b[0]);
    const lo = Math.max(ss[0][0], pe[0][0]), hi = Math.min(ss[ss.length - 1][0], pe[pe.length - 1][0]);
    range.min = Math.ceil(lo * 10) / 10; range.max = Math.floor(Math.min(12, hi) * 10) / 10;
    const at = (front, k) => {
      for (let i = 1; i < front.length; i++) if (front[i][0] >= k) {
        const [x0, y0] = front[i - 1], [x1, y1] = front[i];
        return y0 + ((k - x0) / (x1 - x0 || 1)) * (y1 - y0);
      }
      return front[front.length - 1][1];
    };
    const meter = $("#sim-meter");
    meter.innerHTML = Array.from({ length: 30 }, () => "<i></i>").join("");
    const bars = $$("i", meter);
    function render() {
      const k = +range.value;
      range.style.setProperty("--fill", ((k - range.min) / (range.max - range.min)) * 100 + "%");
      $("#sim-k-out").textContent = k.toFixed(1);
      const a = at(ss, k), b = at(pe, k);
      $("#sim-acc").textContent = (a * 100).toFixed(1) + "%";
      $("#sim-echo").textContent = (b * 100).toFixed(1) + "%";
      bars.forEach((el, i) => el.classList.toggle("on", i < Math.round(k)));
      Charts.line($("#sim-svg"), {
        margin: { l: 48, r: 18, t: 16, b: 34 },
        x: { min: 2.5, max: 12, label: "average packets read", fmt: (v) => v.toFixed(0), tipFmt: (v) => v.toFixed(2) + " packets" },
        y: { min: 0.68, max: 0.92, fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => (v * 100).toFixed(1) + "%" },
        series: [
          { name: "P-ECHO", color: C.muted, points: pe.filter((p) => p[0] <= 12), width: 1.8, dash: "6 5" },
          { name: "Safe-Stop", color: C.cyan, points: ss.filter((p) => p[0] <= 12), width: 2.8, area: true, glow: true },
        ],
        vlines: [{ x: k, color: "rgba(233,237,255,.35)" }],
        markers: [{ x: k, y: a, color: C.cyan }, { x: k, y: b, color: C.muted, r: 4 }],
      });
    }
    range.addEventListener("input", render);
    render();
  }

  /* ------------------------------------------------------------------- offline result charts */
  function resultCharts() {
    if (DATA.auroc_u && $("#auroc-svg")) {
      const ks = Object.keys(DATA.auroc_u).map(Number).sort((a, b) => a - b);
      Charts.line($("#auroc-svg"), {
        margin: { l: 44, r: 14, t: 14, b: 32 },
        x: { min: 1, max: 30, ticks: [1, 3, 5, 8, 12, 20, 30], label: "packets read", tipLabel: "packet ", fmt: String },
        y: { min: 0.4, max: 1, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fmt: (v) => v.toFixed(1) },
        series: [
          { name: "energy alone", color: C.muted, points: ks.map((k) => [k, DATA.auroc_energy[k]]), dash: "6 5", width: 1.8, dots: true },
          { name: "combined unknown score", color: C.violet, points: ks.map((k) => [k, DATA.auroc_u[k]]), width: 2.6, dots: true, glow: true },
        ],
        hlines: [{ y: 0.5, color: "rgba(152,162,204,.4)", label: "chance" }],
      });
      Charts.legend($("#auroc-svg"), [{ name: "combined unknown score", color: C.violet }, { name: "energy alone", color: C.muted }]);
    }
    if (DATA.weeks && $("#weeks-svg")) {
      const w = DATA.weeks.map((d) => [+d.week, d.acc, d.f1]);
      Charts.line($("#weeks-svg"), {
        margin: { l: 44, r: 14, t: 14, b: 32 },
        x: { min: 31, max: 52, ticks: [31, 35, 40, 45, 50], label: "ISO week of 2022", tipLabel: "week " },
        y: { min: 0.6, max: 0.95, fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => (v * 100).toFixed(1) + "%" },
        bands: [{ x0: 30.6, x1: 34.5, color: C.cyan, opacity: 0.07, label: "test", text: C.cyan }],
        series: [
          { name: "accuracy", color: C.cyan, points: w.map((d) => [d[0], d[1]]), width: 2.4, dots: true },
          { name: "macro-F1", color: C.rose, points: w.map((d) => [d[0], d[2]]), width: 2.4, dots: true },
        ],
      });
      Charts.legend($("#weeks-svg"), [{ name: "accuracy", color: C.cyan }, { name: "macro-F1 (rare apps count equally)", color: C.rose }]);
    }
  }

  /* ------------------------------------------------------------------- small things */
  function countUp() {
    if (reduced) return;
    $$("[data-count]").forEach((el) => {
      const target = +el.dataset.count, dec = +el.dataset.dec || 0, suffix = el.textContent.replace(/[\d.,]/g, "").trim();
      const t0 = performance.now() + 900;
      function tick(now) {
        const p = Math.min(1, Math.max(0, (now - t0) / 1400));
        const e = 1 - Math.pow(1 - p, 4);
        el.textContent = (target * e).toFixed(dec) + (suffix ? " " + suffix : "");
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  function copyCommand() {
    const b = $("#copy-cmd");
    if (!b) return;
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("#cmd-text").textContent); b.textContent = "Copied"; }
      catch (_) { b.textContent = "Select and copy"; }
      setTimeout(() => { b.textContent = "Copy command"; }, 1800);
    });
  }

  function health() {
    const pill = $("#status-pill");
    let announced = false;
    async function poll() {
      let s = "down", label = "Service offline";
      try {
        const r = await fetch("/health", { cache: "no-store" });
        const h = await r.json();
        s = h.status === "ready" ? "ready" : h.status === "error" ? "down" : "loading";
        label = s === "ready" ? `Model ${h.model_version} live` : s === "loading" ? "Loading model" : "Model failed to load";
        if (s === "ready" && !announced) { announced = true; window.dispatchEvent(new Event("anytime:ready")); }
      } catch (_) { /* offline: the story still works */ }
      pill.className = "status-pill " + s;
      pill.querySelector("b").textContent = label;
      setTimeout(poll, s === "ready" ? 15000 : 3000);
    }
    poll();
  }

  // ?still: a fixed-height hero, so the whole page can be captured or printed in one image
  if (new URLSearchParams(location.search).has("still")) document.documentElement.classList.add("still");

  document.addEventListener("DOMContentLoaded", () => {
    requestAnimationFrame(() => document.body.classList.add("loaded"));
    flowField(); pointer(); scrolling(); flowprint(); scrubber(); simulator(); resultCharts(); countUp(); copyCommand(); health();
  });
})();
