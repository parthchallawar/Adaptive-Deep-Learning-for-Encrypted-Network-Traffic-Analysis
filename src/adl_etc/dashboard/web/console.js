/* The live console (spec 017's seven pages). Every number comes from the service API; the active
   page polls once a second (correction S6). Opened with #console, closed with any story link. */
(function () {
  "use strict";
  const C = { cyan: "#5BE3E0", violet: "#8C7BFF", rose: "#FF6F91", amber: "#F5C46B", text: "#E9EDFF", muted: "#98A2CC", faint: "#5F6A97", blue: "#7FB8FF" };
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v === null || v === undefined || !Number.isFinite(+v) ? "–" : (v * 100).toFixed(d) + "%");
  const num = (v, d = 2) => (v === null || v === undefined || !Number.isFinite(+v) ? "–" : (+v).toFixed(d));
  const time = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const DATA = window.ANYTIME_DATA || {};

  async function api(path, params, opts) {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    const r = await fetch(path + qs, Object.assign({ cache: "no-store" }, opts || {}));
    let body = null;
    try { body = await r.json(); } catch (_) { body = null; }
    if (!r.ok) {
      const msg = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `${r.status} ${r.statusText}`;
      throw new Error(msg);
    }
    return body;
  }
  const post = (path, json) => api(path, null, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(json || {}) });

  function toast(msg, err) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show" + (err ? " err" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { t.className = "toast"; }, 3200);
  }

  const kpi = (value, label, tone) => `<div class="kpi ${tone || ""}"><b>${value}</b><span>${label}</span></div>`;
  const tag = (d) => {
    const cls = d === "commit" ? "commit" : d === "reject" ? "reject" : d === "undecidable" ? "undecidable" : "forced";
    const text = { commit: "named", reject: "unknown", forced_commit: "named at end", forced_reject: "unknown at end", undecidable: "no payload" }[d] || d;
    return `<span class="tag ${cls}">${text}</span>`;
  };

  /* ---------------------------------------------------------------- shell: open, tabs, health */
  const state = { tab: "monitor", lastId: 0, rows: [], timer: null, ready: false, inspect: null, expTab: null };

  function openConsole(open) {
    const el = $("#console");
    el.hidden = !open;
    document.body.classList.toggle("console-open", open);
    if (open) { selectTab(state.tab); } else { clearTimeout(state.timer); }
  }

  function route() {
    const h = location.hash;
    const [sub, arg] = (h.startsWith("#console/") ? h.slice(9) : "").split("/");
    if (sub && PAGES[sub]) state.tab = sub;
    if (sub === "inspector" && +arg) state.inspect = +arg;
    openConsole(h === "#console" || h.startsWith("#console/"));
  }

  function selectTab(name) {
    state.tab = name;
    $$(".c-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    $$(".c-page").forEach((p) => { p.hidden = p.dataset.page !== name; });
    tick(true);
  }

  async function tick(first) {
    clearTimeout(state.timer);
    if ($("#console").hidden) return;
    try {
      await health();
      if (state.ready || state.tab === "experiments") await PAGES[state.tab](first);
    } catch (e) {
      console.warn(e);
    }
    const slow = { experiments: 0, system: 3000, inspector: 0 }[state.tab];
    if (slow !== 0) state.timer = setTimeout(() => tick(false), slow || 1000);
  }

  async function health() {
    const banner = $("#c-banner");
    try {
      const h = await api("/health");
      state.ready = h.status === "ready";
      banner.hidden = state.ready;
      if (h.status === "loading") banner.textContent = "Loading the model. Pages fill in once it is ready.";
      if (h.status === "error") banner.textContent = `The model did not load: ${h.error}. Run python scripts/make_production_bundle.py, then restart the service.`;
      $("#c-health").innerHTML = `<b>${state.ready ? "Model " + esc(h.model_version) : esc(h.status)}</b><br>up ${Math.round(h.uptime_s / 60)} min<br>${h.db_pending_rows} rows waiting to be written`;
    } catch (_) {
      state.ready = false;
      banner.hidden = false;
      banner.textContent = "The service is not answering. Start it with python -m adl_etc.service, then reload this page.";
    }
  }

  /* ---------------------------------------------------------------- Monitor */
  let periodsLoaded = false;
  async function loadPeriods() {
    if (periodsLoaded) return;
    const p = await api("/ingest/periods");
    const sel = $("#rp-period");
    const groups = { test_id: "Test weeks (in distribution)", test_drift: "Drift weeks", val: "Validation weeks", train: "Training weeks", "": "Other weeks" };
    const by = {};
    p.periods.forEach((x) => { (by[x.split] = by[x.split] || []).push(x.period); });
    sel.innerHTML = Object.keys(groups).filter((g) => by[g]).map((g) => `<optgroup label="${groups[g]}">${by[g].map((w) => `<option value="${w}">${w.replace("WEEK-", "Week ").replace("-", " / ")}</option>`).join("")}</optgroup>`).join("");
    if (by.test_id) sel.value = by.test_id[0];
    if (!p.periods.length) sel.innerHTML = `<option value="">No shards found under data_root</option>`;
    periodsLoaded = true;
  }

  async function monitor(first) {
    if (first) await loadPeriods();
    const [s, jobs, fl, series] = await Promise.all([
      api("/stats", { window_s: 10 }),
      api("/jobs"),
      api("/flows", { since: state.lastId, limit: 500 }),
      api("/stats/series", { minutes: 10, bucket_s: 5 }),
    ]);
    $("#kpis").innerHTML = [
      kpi(num(s.flows_per_s, 1), "flows decided per second", "cyan"),
      kpi(num(s.mean_k, 2), "average packets read"),
      kpi(pct(s.commit_rate), "named before the end"),
      kpi(pct(s.reject_rate), "flagged unknown", "rose"),
      kpi(pct(s.accuracy), "right answer, last 10 s"),
      kpi(num(s.theta_commit, 3), "commit threshold in force", "violet"),
      kpi(num(s.events_per_s, 0), "packets scored per second"),
      kpi(num(s.batch_ms_p95, 0) + " ms", "slowest 5% of batches"),
    ].join("");
    $("#jobs").innerHTML = jobs.length ? jobs.slice(0, 8).map((j) => {
      const p = j.total ? Math.round((j.progress / j.total) * 100) : 0;
      const what = j.type === "shards" ? (j.detail.period || "").replace("WEEK-", "Week ") : j.detail.file || "capture";
      return `<li><div class="row"><b>${esc(what)}</b><span>${esc(j.status)}${j.status === "running" ? ` <button data-cancel="${j.id}" type="button">Stop</button>` : ""}</span></div><div class="bar"><i style="width:${p}%"></i></div>${j.progress.toLocaleString()} of ${j.total.toLocaleString()} flows${j.error ? ` <span class="bad">${esc(j.error)}</span>` : ""}</li>`;
    }).join("") : `<li class="muted">No replays yet. Start one on the left.</li>`;
    if (fl.flows.length) {
      state.lastId = fl.last_id;
      state.rows = fl.flows.concat(state.rows).slice(0, 500);
      renderFlows(fl.flows.length);
    } else if (first) renderFlows(0);
    const pts = series.series;
    const host = $("#mon-chart");
    if (pts.length) {
      Charts.line(host, {
        height: 220,
        x: { fmt: (t) => time(t).slice(0, 5), tipFmt: time, min: pts[0].t, max: pts[pts.length - 1].t },
        y: { min: 0, fmt: (v) => v.toFixed(0) },
        series: [
          { name: "average packets", color: C.cyan, points: pts.map((p) => [p.t, p.mean_k]), area: true, width: 2.4 },
          { name: "flagged unknown, % ÷ 2", color: C.rose, points: pts.map((p) => [p.t, (p.reject_rate || 0) * 50]), width: 2 },
        ],
      });
      Charts.legend(host, [{ name: "average packets read", color: C.cyan }, { name: "share flagged unknown (scale: 10 = 20%)", color: C.rose }]);
    } else host.innerHTML = `<p class="empty">Start a replay to see decisions arrive.</p>`;
  }

  function renderFlows(fresh) {
    const tb = $("#flows-tbl tbody");
    if (!state.rows.length) { tb.innerHTML = `<tr><td colspan="8" class="empty">No decisions yet.</td></tr>`; return; }
    tb.innerHTML = state.rows.slice(0, 200).map((r, i) => {
      const truth = r.true_class ? `<span class="${r.correct ? "ok" : "bad"}">${esc(r.true_class)}</span>` : `<span class="muted">not known</span>`;
      return `<tr data-flow="${r.flow_id}" class="${i < fresh ? "fresh" : ""}"><td>${r.flow_id}</td><td>${tag(r.decision)}</td><td>${esc(r.class_name || "–")}</td><td>${truth}</td><td>${r.k}</td><td>${num(r.p_safe, 3)}</td><td>${num(r.u, 2)}</td><td>${time(r.decided_at)}</td></tr>`;
    }).join("");
  }

  /* ---------------------------------------------------------------- Budget */
  let budgetInit = false;
  async function budget(first) {
    const b = await api("/budget", { history: 300 });
    if (first || !budgetInit) {
      budgetInit = true;
      $$("#bud-mode button").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.v === b.mode)));
      $("#bud-metric").value = b.metric;
      $("#bud-target").value = b.target; $("#bud-target-out").textContent = b.target;
      $("#bud-theta").value = b.manual_theta; $("#bud-theta-out").textContent = num(b.manual_theta, 3);
    }
    const modeName = { auto: "Auto: controller in charge", manual: "Manual threshold", quality_first: "Quality first" }[b.mode];
    $("#bud-state").innerHTML = `<h2>${modeName}</h2><div class="kpis">${[
      kpi(num(b.theta_commit, 3), "commit threshold in force", "violet"),
      kpi(num(b.target, 1), b.metric === "mean_k" ? "target average packets" : "target 95th percentile"),
      kpi(b.window_fill, `of ${b.window} decisions in this window`),
      kpi(b.windows_closed, "windows closed"),
      kpi(b.k_cap, "packet cap"),
      kpi(pct(b.f, 1), "unknown false-alarm target"),
    ].join("")}</div>`;
    const h = b.history.filter((r) => r.observed !== null);
    const host = $("#bud-chart");
    if (h.length) {
      Charts.line(host, {
        height: 230,
        x: { fmt: (v) => String(v), tipLabel: "window " },
        y: { min: 0, fmt: (v) => v.toFixed(0) },
        series: [
          { name: "target", color: C.faint, points: h.map((r) => [r.window_id, r.target]), step: true, dash: "6 5", width: 1.6 },
          { name: "realised", color: C.cyan, points: h.map((r) => [r.window_id, r.observed]), width: 2.4, area: true, dots: h.length < 60 },
        ],
      });
      Charts.legend(host, [{ name: "realised", color: C.cyan }, { name: "target", color: C.faint }]);
      Charts.line($("#theta-chart"), {
        height: 160,
        x: { tipLabel: "window " },
        y: { fmt: (v) => v.toFixed(2) },
        series: [{ name: "commit threshold", color: C.violet, points: b.history.map((r) => [r.window_id, r.theta_commit]), width: 2.2, step: true }],
      });
    } else {
      host.innerHTML = `<p class="empty">The controller acts once per window of ${b.window} decisions. Start a replay to fill one.</p>`;
      $("#theta-chart").innerHTML = "";
    }
    const front = (b.offline && b.offline.front) || (DATA.fronts && DATA.fronts.safe_stop) || [];
    if (front.length) {
      const pts = front.filter((p) => p[0] <= 14).sort((x, y) => x[0] - y[0]);
      const at = pts.reduce((best, p) => (Math.abs(p[0] - b.target) < Math.abs(best[0] - b.target) ? p : best), pts[0]);
      Charts.line($("#bud-front"), {
        height: 220,
        x: { min: Math.max(1, pts[0][0] - 0.2), max: 14, label: "average packets", fmt: (v) => v.toFixed(0), tipFmt: (v) => v.toFixed(2) },
        y: { fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => (v * 100).toFixed(1) + "%" },
        series: [{ name: "Safe-Stop accuracy", color: C.cyan, points: pts, width: 2.4, area: true }],
        markers: [{ x: at[0], y: at[1], color: C.violet, label: `target ${b.target}` }],
      });
    }
  }

  function wireBudget() {
    $$("#bud-mode button").forEach((x) => x.addEventListener("click", () => $$("#bud-mode button").forEach((y) => y.setAttribute("aria-checked", String(x === y)))));
    $("#bud-target").addEventListener("input", (e) => { $("#bud-target-out").textContent = e.target.value; });
    $("#bud-theta").addEventListener("input", (e) => { $("#bud-theta-out").textContent = (+e.target.value).toFixed(3); });
    $("#budget-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const mode = ($("#bud-mode [aria-checked=true]") || {}).dataset?.v || "auto";
      const body = { mode, metric: $("#bud-metric").value, target: +$("#bud-target").value };
      if (mode === "manual") body.theta_commit = +$("#bud-theta").value;
      try { await post("/budget", body); toast("Budget applied"); budgetInit = false; tick(false); }
      catch (err) { toast(err.message, true); }
    });
  }

  /* ---------------------------------------------------------------- Flow inspector */
  async function inspector() {
    const id = state.inspect;
    const body = $("#insp-body");
    if (!id) { body.innerHTML = `<div class="panel empty">Pick a flow: enter its id above, or select a row on the Monitor or Unknown page.</div>`; return; }
    $("#insp-id").value = id;
    let d;
    try { d = await api(`/flows/${id}`); }
    catch (err) { body.innerHTML = `<div class="panel empty">${esc(err.message)}</div>`; return; }
    const dec = d.decision || {};
    const ex = d.explain;
    const tr = d.trace;
    body.innerHTML = `
      <div class="kpis">
        ${kpi(tag(dec.decision), "decision")}
        ${kpi(dec.k ?? 0, "packets read")}
        ${kpi(esc(dec.class_name || "–"), "app named")}
        ${kpi(d.true_class ? `<span class="${dec.correct ? "ok" : "bad"}">${esc(d.true_class)}</span>` : "not known", "true app")}
        ${kpi(num(dec.p_safe, 3), "confidence at decision", "cyan")}
        ${kpi(num(dec.u, 2) + ` <small class="muted">/ ${num(dec.u_threshold, 2)}</small>`, "unknown score / threshold", "rose")}
      </div>
      <div class="panel"><h2>Packets</h2><p class="hint">Bars above the line go to the server, below come back; spacing follows the gap between packets. The marker is where the decision fell.</p><div id="insp-packets" class="chart small"></div></div>
      <div class="insp-grid">
        <div class="panel"><h2>Top three apps after each packet</h2><div id="insp-top" class="chart"></div></div>
        <div class="panel"><h2>Confidence against the commit threshold</h2><div id="insp-p" class="chart"></div></div>
        <div class="panel"><h2>Unknown score against its threshold</h2><div id="insp-u" class="chart"></div></div>
        <div class="panel"><h2>Surprise per packet</h2><p class="hint">How unexpected each packet was to the next-packet head (negative log-likelihood).</p><div id="insp-s" class="chart"></div></div>
      </div>`;
    const pk = d.packets || [];
    if (pk.length) {
      let x = 0;
      const pts = pk.map((p, i) => { x += i ? 1 + Math.log1p(p.ipt_ms) : 0; return { x, v: p.dir * p.size, i }; });
      const svg = Charts.line($("#insp-packets"), { height: 170, x: { min: -0.6, max: x + 0.6, ticks: [] }, y: { min: -1500, max: 1500, ticks: [-1500, 0, 1500], fmt: (v) => Math.abs(v) }, series: [], tooltip: false });
      const { xs, ys } = svg;
      pts.forEach((p) => {
        const decided = dec.k && p.i + 1 === dec.k;
        const r = Charts.el("rect", { x: xs.map(p.x) - 5, y: Math.min(ys.map(0), ys.map(p.v)), width: 10, height: Math.abs(ys.map(p.v) - ys.map(0)), rx: 2, fill: p.i + 1 <= (dec.k || 0) ? (p.v > 0 ? C.cyan : C.violet) : C.faint }, svg.f.svg);
        r.addEventListener("mousemove", (e) => Charts.tip.show(`Packet ${p.i + 1}: ${Math.abs(p.v)} bytes ${p.v > 0 ? "to the server" : "back"}, gap ${pk[p.i].ipt_ms} ms`, e.clientX, e.clientY));
        r.addEventListener("mouseleave", Charts.tip.hide);
        if (decided) Charts.el("line", { x1: xs.map(p.x) + 9, x2: xs.map(p.x) + 9, y1: 10, y2: 150, stroke: dec.decision && dec.decision.includes("reject") ? C.rose : C.cyan, "stroke-width": 2, "stroke-dasharray": "4 4" }, svg.f.svg);
      });
    }
    const kx = { min: 1, tipLabel: "after packet " };
    const vl = dec.k ? [{ x: dec.k, color: "rgba(233,237,255,.4)", label: "decision", dash: "3 4" }] : [];
    if (ex && ex.top.length) {
      const names = {};
      ex.top.forEach((row) => row.forEach((t) => { names[t.class] = (names[t.class] || 0) + t.p; }));
      const best = Object.keys(names).sort((a, b) => names[b] - names[a]).slice(0, 4);
      const colors = [C.cyan, C.violet, C.amber, C.blue];
      const host = $("#insp-top");
      Charts.line(host, {
        width: 520, height: 240, x: kx, y: { min: 0, max: 1, fmt: (v) => v.toFixed(1) }, vlines: vl,
        series: best.map((n, i) => ({ name: n, color: colors[i], width: 2.2, dots: true, points: ex.top.map((row, k) => { const t = row.find((r) => r.class === n); return [k + 1, t ? t.p : NaN]; }) })),
      });
      Charts.legend(host, best.map((n, i) => ({ name: n, color: colors[i] })));
      Charts.line($("#insp-p"), { width: 520, height: 240, x: kx, y: { min: 0, max: 1 }, vlines: vl, series: [{ name: "confidence", color: C.cyan, points: ex.p_safe.map((v, i) => [i + 1, v]), area: true, width: 2.4, dots: true }], hlines: dec.theta_commit ? [{ y: dec.theta_commit, color: C.cyan, label: "commit threshold" }] : [] });
      const thr = (d.theta_reject || []).slice(0, ex.u.length);
      Charts.line($("#insp-u"), { width: 520, height: 240, x: kx, vlines: vl.concat([{ x: 3, color: "rgba(152,162,204,.35)", dash: "2 4" }]), series: [
        { name: "threshold", color: C.rose, points: thr.map((v, i) => [i + 1, v]), dash: "5 5", width: 1.4 },
        { name: "unknown score", color: C.violet, points: ex.u.map((v, i) => [i + 1, v]), width: 2.4, dots: true }] });
      Charts.bars($("#insp-s"), { width: 520, height: 240, keys: ex.k.slice(1), groups: [{ name: "surprise", color: C.amber, values: Object.fromEntries(ex.k.slice(1).map((k, i) => [k, ex.pkt_nll[i + 1]])) }], y: { fmt: (v) => v.toFixed(1) }, x: { tipLabel: "packet " } });
    } else if (tr) {
      Charts.line($("#insp-p"), { width: 520, height: 240, x: kx, y: { min: 0, max: 1 }, series: [{ name: "confidence", color: C.cyan, points: tr.p_safe.map((v, i) => [i + 1, v]) }] });
      $("#insp-top").innerHTML = `<p class="empty">Per-class detail needs the real model loaded.</p>`;
    }
  }

  /* ---------------------------------------------------------------- Unknown */
  async function unknown() {
    const u = await api("/unknown", { limit: 200 });
    const keys = Array.from({ length: 30 }, (_, i) => i + 1);
    const hostH = $("#unk-hist");
    Charts.bars(hostH, { width: 520, height: 240, keys, groups: [
      { name: "all decisions", color: C.cyan, opacity: 0.55, values: u.decision_k_histogram },
      { name: "flagged unknown", color: C.rose, values: u.rejection_k_histogram },
    ], x: { tipLabel: "packet " }, y: { fmt: (v) => v.toFixed(0) } });
    Charts.legend(hostH, [{ name: "all decisions", color: C.cyan }, { name: "flagged unknown", color: C.rose }]);
    const au = u.offline_auroc || {};
    if (au.u) {
      const ks = Object.keys(au.u).map(Number).sort((a, b) => a - b);
      const hostA = $("#unk-auroc");
      const colors = { u: C.violet, energy_raw: C.muted, mahal: C.cyan, surprise: C.amber };
      const names = { u: "combined score", energy_raw: "energy", mahal: "distance to known apps", surprise: "surprise" };
      Charts.line(hostA, { width: 520, height: 240, x: { min: 1, max: 30, ticks: [1, 3, 5, 8, 12, 20, 30], tipLabel: "packet " }, y: { min: 0.4, max: 1, fmt: (v) => v.toFixed(1) },
        series: Object.keys(colors).filter((s) => au[s]).map((s) => ({ name: names[s], color: colors[s], width: s === "u" ? 2.8 : 1.6, dots: true, points: ks.map((k) => [k, au[s][k]]) })) });
      Charts.legend(hostA, Object.keys(colors).filter((s) => au[s]).map((s) => ({ name: names[s], color: colors[s] })));
    }
    $("#unk-tbl tbody").innerHTML = u.rejected.length ? u.rejected.map((r) => `<tr data-flow="${r.flow_id}"><td>${r.flow_id}</td><td>${r.k}</td><td>${num(r.u, 2)}</td><td>${num(r.u_threshold, 2)}</td><td>${r.true_class ? `<span class="${r.correct ? "ok" : "bad"}">${esc(r.true_class)}${r.correct ? " (held out)" : ""}</span>` : "–"}</td><td>${time(r.decided_at)}</td></tr>`).join("") : `<tr><td colspan="6" class="empty">Nothing flagged yet.</td></tr>`;
  }

  /* ---------------------------------------------------------------- Drift */
  const SIGNAL_NAMES = { mean_k: "average packets", commit_rate: "named before the end", reject_rate: "flagged unknown", mean_safe_at_commit: "confidence when naming", mean_pkt_nll: "next-packet surprise", prior_kl: "app mix shift (KL)", theta_commit: "commit threshold" };
  async function drift() {
    const d = await api("/drift", { history: 200 });
    if (!d.enabled) { $("#drift-state").innerHTML = `<p class="empty">No drift reference loaded.</p>`; return; }
    const levelText = { none: "Quiet", warning: "Warning: one signal", alarm: "Alarm: two or more signals" }[d.level];
    $("#drift-state").innerHTML = `<h2>Status</h2><p><span class="level ${d.level}">${levelText}</span></p>
      <div class="kpis">${[
        kpi(d.window_fill.toLocaleString(), `of ${d.window.toLocaleString()} decisions in this window`),
        kpi(d.windows_closed, `windows closed (first ${d.warmup} only warm up)`),
        kpi(num(d.h, 1), "alarm level h"),
        kpi(d.mode === "controller" ? "Auto" : "Fixed", d.mode === "controller" ? "reference: controller running" : "reference: fixed threshold"),
      ].join("")}</div>${d.reference_rescaled ? `<p class="hint">The window is smaller than the ${d.reference_window.toLocaleString()} decisions the reference was measured on, so its spread was scaled by the square root of the ratio. Treat alarms as approximate.</p>` : ""}`;
    const z = (d.events.length ? d.events[d.events.length - 1].z : {}) || {};
    $("#drift-bars").innerHTML = d.signals.map((s) => {
      const v = z[s];
      const c = d.cusum[s] || { pos: 0, neg: 0 };
      const w = Number.isFinite(v) ? Math.min(50, Math.abs(v) * 10) : 0;
      const left = v >= 0 ? 50 : 50 - w;
      const hot = Math.max(c.pos, c.neg) / (d.h || 5);
      return `<div class="zbar"><span>${SIGNAL_NAMES[s] || s}</span><span class="track"><i style="left:${left}%;width:${w}%;background:${hot > 0.6 ? C.rose : hot > 0.3 ? C.amber : C.cyan}"></i></span><span>${Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(1) + "σ" : "–"}</span></div>`;
    }).join("") + `<p class="hint">Bars: the last window's distance from validation. Colour: how close each signal's CUSUM is to alarming.</p>`;
    const hist = d.history;
    const host = $("#drift-chart");
    if (hist.length) {
      const palette = [C.cyan, C.violet, C.rose, C.amber, C.blue, C.muted];
      Charts.line(host, { height: 240, x: { tipLabel: "window " }, y: { fmt: (v) => v.toFixed(0) },
        series: d.signals.map((s, i) => ({ name: SIGNAL_NAMES[s] || s, color: palette[i % palette.length], width: 2, dots: hist.length < 40, points: hist.map((w) => [w.window, w.signals[s] ? w.signals[s].z : NaN]) })),
        vlines: hist.filter((w) => w.level === "alarm").slice(-5).map((w) => ({ x: w.window, color: "rgba(255,111,145,.5)", dash: "3 4" })) });
      Charts.legend(host, d.signals.map((s, i) => ({ name: SIGNAL_NAMES[s] || s, color: palette[i % palette.length] })));
    } else host.innerHTML = `<p class="empty">The monitor closes a window every ${d.window.toLocaleString()} decisions. Replay a late drift week, such as week 50, to watch it move.</p>`;
    if (DATA.weeks) {
      const w = DATA.weeks.map((x) => [+x.week, x.f1]);
      Charts.line($("#drift-weeks"), { height: 200, x: { min: 31, max: 52, tipLabel: "week " }, y: { fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => (v * 100).toFixed(1) + "%" }, bands: [{ x0: 30.6, x1: 34.5, color: C.cyan, label: "test" }], series: [{ name: "macro-F1", color: C.rose, points: w, width: 2.4, dots: true, area: true }] });
    }
  }

  /* ---------------------------------------------------------------- Experiments */
  let expData = null;
  function mdToHtml(md) {
    const lines = md.split(/\r?\n/);
    let html = "", i = 0;
    const inline = (s) => esc(s).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    while (i < lines.length) {
      const l = lines[i];
      if (/^\s*\|/.test(l)) {
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
        const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const body = rows.filter((r, j) => j !== 1 || !/^\s*\|\s*:?-+/.test(r));
        html += `<table><thead><tr>${cells(body[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body.slice(1).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
        continue;
      }
      const h = /^(#{1,3})\s+(.*)/.exec(l);
      if (h) html += `<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`;
      else if (l.trim()) html += `<p>${inline(l)}</p>`;
      i++;
    }
    return html;
  }
  async function experiments() {
    if (!expData) expData = await api("/experiments");
    const names = { phase4_table: "Phase 4: stopping, unknowns, budget", phase5_table: "Phase 5: drift and cost", phase3_table: "Phase 3: transformer training", main_table: "Baselines" };
    const keys = Object.keys(names).filter((k) => expData.tables[k]);
    if (!state.expTab) state.expTab = keys[0];
    $("#exp-tabs").innerHTML = keys.map((k) => `<button role="tab" data-exp="${k}" aria-selected="${k === state.expTab}">${names[k]}</button>`).join("");
    $("#exp-table").innerHTML = keys.length ? mdToHtml(expData.tables[state.expTab]) : `<p class="empty">No tables under results/summaries.</p>`;
    const crit = [];
    Object.values(expData.criteria || {}).forEach((group) => { if (group) Object.entries(group).forEach(([hyp, rows]) => (Array.isArray(rows) ? rows : []).forEach((r) => crit.push({ hyp, ...r }))); });
    $("#exp-criteria").innerHTML = `<h2>Pass bars set before the runs</h2>` + (crit.length ? `<div class="crit">${crit.map((c) => { const verdict = /no bar/.test(String(c.bar)) || c.met === null || c.met === undefined ? "Reported" : c.met ? "Met" : "Not met"; return `<div><span class="${verdict === "Met" ? "ok" : verdict === "Reported" ? "muted" : "bad"}">${verdict}</span><span>${esc(c.hyp.replace("_", " "))}: ${esc(c.criterion)}</span><span class="muted">${typeof c.value === "number" ? c.value.toFixed(3) : esc(c.value)} (bar ${esc(c.bar)})</span></div>`; }).join("")}</div>` : `<p class="empty">No criteria files found.</p>`);
  }

  /* ---------------------------------------------------------------- Model and system */
  async function system() {
    const [m, s] = await Promise.all([api("/model"), api("/system")]);
    const metrics = await fetch("/metrics", { cache: "no-store" }).then((r) => r.text()).catch(() => "");
    const a = m.active || {};
    const dl = (o) => `<dl class="dl">${Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
    $("#sys-model").innerHTML = `<h2>Active model</h2>` + dl({
      version: a.version, run: a.run, parameters: a.n_parameters ? a.n_parameters.toLocaleString() : "–", classes: a.n_classes,
      "commit threshold (calibrated to 6 packets)": num(a.theta_commit_calibrated, 4), "unknown false-alarm target": a.f,
      "unknown score weights": (a.unknown_weights || []).join(", "), "load time": a.load_seconds ? a.load_seconds + " s" : "–",
      "torch threads": a.torch_threads, "flows in flight": m.in_flight_flows,
    });
    const e = s.engine || {};
    $("#sys-engine").innerHTML = `<h2>Engine</h2>` + dl({
      "packets scored per second": num(e.events_per_s, 0), "batch time, median": num(e.batch_ms_p50, 1) + " ms", "batch time, slowest 5%": num(e.batch_ms_p95, 1) + " ms",
      "open flows": e.open_flows, "queued packets": e.queue_size, decisions: e.decisions, "dropped flows": e.dropped_flows, "ignored packets": e.ignored_events,
      "database rows written": s.db_rows_written, "database failures": s.db_failed_rows, platform: s.platform, CPUs: s.cpu_count, database: s.db_url,
    });
    $("#sys-metrics").textContent = metrics;
  }

  const PAGES = { monitor, budget, inspector, unknown, drift, experiments, system };

  /* ---------------------------------------------------------------- forms and wiring */
  function wire() {
    $$(".c-tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
    $("#close-console").addEventListener("click", (e) => { e.preventDefault(); history.pushState(null, "", "#hero"); openConsole(false); });
    window.addEventListener("hashchange", route);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#console").hidden) { history.pushState(null, "", "#hero"); openConsole(false); } });

    $("#replay-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const period = $("#rp-period").value;
      if (!period) { toast("Pick a week first", true); return; }
      try {
        const j = await post("/ingest/shards", { period, count: +$("#rp-count").value, rate: +$("#rp-rate").value, speed: +$("#rp-speed").value });
        toast(`Replaying ${j.total.toLocaleString()} flows from ${period.replace("WEEK-", "week ")}`);
        tick(false);
      } catch (err) { toast(err.message, true); }
    });
    $("#pcap-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = $("#pc-file").files[0];
      if (!f) { toast("Choose a pcap file first", true); return; }
      const fd = new FormData();
      fd.append("file", f); fd.append("speed", $("#pc-speed").value);
      try { await api("/ingest/pcap", null, { method: "POST", body: fd }); toast(`Uploaded ${f.name}`); tick(false); }
      catch (err) { toast(err.message, true); }
    });
    $("#jobs").addEventListener("click", async (e) => {
      const id = e.target.dataset && e.target.dataset.cancel;
      if (!id) return;
      try { await post(`/jobs/${id}/cancel`); toast("Replay stopped"); } catch (err) { toast(err.message, true); }
    });
    const openFlow = (e) => {
      const tr = e.target.closest("tr[data-flow]");
      if (!tr) return;
      state.inspect = +tr.dataset.flow;
      selectTab("inspector");
    };
    $("#flows-tbl").addEventListener("click", openFlow);
    $("#unk-tbl").addEventListener("click", openFlow);
    $("#insp-form").addEventListener("submit", (e) => { e.preventDefault(); state.inspect = +$("#insp-id").value || null; tick(true); });
    $("#exp-tabs").addEventListener("click", (e) => { const k = e.target.dataset.exp; if (k) { state.expTab = k; experiments(); } });
    $("#reload-btn").addEventListener("click", async () => {
      $("#reload-msg").textContent = "Reloading…";
      try { const r = await post("/model", {}); $("#reload-msg").textContent = `Loaded ${r.active.version}. ${r.in_flight_on_previous} flows finish on the previous model.`; }
      catch (err) { $("#reload-msg").textContent = err.message; }
    });
    wireBudget();
    route();
  }

  document.addEventListener("DOMContentLoaded", wire);
})();
