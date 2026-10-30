"""The fleet dashboard page.

One static page, served by the dashboard component, that reads the same JSON routes an
operator could `curl`: `/fleet`, `/coverage`, `/cost`, `/degraded`, and `POST /inject`.
It computes nothing the API does not already say. A page that derived its own numbers
would be a second implementation of the pipeline that could disagree with the first.

Kept as a string in a module rather than a package data file, so the wheel and the
image cannot ship without it. The script's hash goes into the Content-Security-Policy,
which lets the page run its own inline script and nothing else.
"""

# ruff: noqa: E501 - the page's own JavaScript and CSS, which are not Python lines.

from __future__ import annotations

import base64
import hashlib

_SCRIPT = r"""
const token = new URLSearchParams(location.hash.slice(1)).get("token");
const headers = token ? {Authorization: "Bearer " + token} : {};
const history = new Map();
const lastStatus = new Map();
const prevRul = new Map();
let selected = null;
let follow = true;

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers, ...(options.headers || {})}});
  if (!response.ok) throw new Error(path + " " + response.status);
  return response.json();
}

const el = (id) => document.getElementById(id);
const fmt = (x, d = 1) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
function node(tag, attrs = {}, text = "") {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text) n.textContent = text;
  return n;
}

function statusLabel(row) {
  if (row.status === "degraded") return "low confidence";
  if (row.status === "alerting") return "maintenance due";
  return "nominal";
}

function logEvent(kind, text) {
  const item = node("li", {class: kind});
  item.append(node("time", {}, new Date().toLocaleTimeString()), node("span", {}, text));
  el("events").prepend(item);
  while (el("events").children.length > 40) el("events").lastChild.remove();
}

function bandBar(row) {
  const scale = (v) => Math.max(0, Math.min(100, (v / 130) * 100));
  const bar = node("div", {class: "bar"});
  const band = node("div", {class: "band " + row.status});
  band.style.left = scale(row.lower) + "%";
  band.style.width = Math.max(0.5, scale(row.upper) - scale(row.lower)) + "%";
  const point = node("div", {class: "point"});
  point.style.left = scale(row.rul) + "%";
  bar.append(band, point);
  return bar;
}

function renderFleet(rows) {
  const body = el("fleet-rows");
  body.replaceChildren();
  for (const row of rows) {
    const tr = node("tr", {class: row.engine === selected ? "selected" : ""});
    tr.addEventListener("click", () => { selected = row.engine; follow = false; draw(); });
    const states = row.degraded.map((s) => s.state.replace("_", " ") + " · " + s.instrument).join(", ");
    tr.append(
      node("td", {class: "mono"}, row.engine),
      node("td", {class: "num"}, String(row.cycle)),
      node("td", {class: "num strong"}, fmt(row.rul)),
      node("td", {class: "num dim"}, "[" + fmt(row.lower) + ", " + fmt(row.upper) + "]"),
    );
    const bandCell = node("td");
    bandCell.append(bandBar(row));
    const statusCell = node("td");
    statusCell.append(node("span", {class: "chip " + row.status}, statusLabel(row)));
    if (states) statusCell.append(node("div", {class: "why"}, states));
    tr.append(bandCell, statusCell);
    body.append(tr);
  }
  el("n-engines").textContent = rows.length;
  el("n-alerting").textContent = rows.filter((r) => r.status === "alerting").length;
  el("n-degraded").textContent = rows.filter((r) => r.status === "degraded").length;
}

function track(rows) {
  let moved = null;
  for (const row of rows) {
    const series = history.get(row.engine) || [];
    const last = series[series.length - 1];
    if (!last || last.cycle !== row.cycle || last.rul !== row.rul) {
      series.push({cycle: row.cycle, lower: row.lower, rul: row.rul, upper: row.upper});
      if (series.length > 600) series.shift();
      history.set(row.engine, series);
    }
    if (prevRul.has(row.engine) && prevRul.get(row.engine) !== row.rul) moved = row.engine;
    prevRul.set(row.engine, row.rul);

    const before = lastStatus.get(row.engine);
    if (before !== undefined && before !== row.status) {
      if (row.status === "alerting") {
        logEvent("alerting", row.engine + ": maintenance due, lower bound " + fmt(row.lower) + " cycles");
      } else if (row.status === "degraded") {
        const why = row.degraded.map((s) => s.detail).join("; ");
        logEvent("degraded", row.engine + ": low confidence, " + why);
      } else {
        logEvent("nominal", row.engine + ": back to nominal");
      }
    }
    lastStatus.set(row.engine, row.status);
  }
  if (follow && moved) selected = moved;
  if (!selected && rows.length) selected = rows[0].engine;
}

function draw() {
  const svg = el("chart");
  svg.replaceChildren();
  const series = history.get(selected) || [];
  el("chart-title").textContent = selected ? selected : "waiting for the stream";
  if (series.length < 2) return;
  const W = 640, H = 240, pad = 34;
  const x0 = series[0].cycle, x1 = series[series.length - 1].cycle;
  const yMax = Math.max(130, ...series.map((p) => p.upper));
  const x = (i) => pad + (i / (series.length - 1)) * (W - 2 * pad);
  const y = (v) => H - pad - (Math.max(0, v) / yMax) * (H - 2 * pad);
  const ns = "http://www.w3.org/2000/svg";
  const make = (tag, attrs) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  for (const v of [0, 25, 50, 75, 100, 125]) {
    svg.append(make("line", {x1: pad, x2: W - pad, y1: y(v), y2: y(v), class: "grid"}));
    const label = make("text", {x: 4, y: y(v) + 4, class: "axis"});
    label.textContent = v;
    svg.append(label);
  }
  const upper = series.map((p, i) => x(i) + "," + y(p.upper));
  const lower = series.map((p, i) => x(i) + "," + y(p.lower)).reverse();
  svg.append(make("polygon", {points: upper.concat(lower).join(" "), class: "band-area"}));
  svg.append(make("polyline", {points: series.map((p, i) => x(i) + "," + y(p.rul)).join(" "), class: "rul-line"}));
  const first = make("text", {x: pad, y: H - 8, class: "axis"});
  first.textContent = "cycle " + x0;
  const last = make("text", {x: W - pad, y: H - 8, class: "axis end"});
  last.textContent = "cycle " + x1;
  svg.append(first, last);
  const now = series[series.length - 1];
  el("chart-now").textContent = "RUL " + fmt(now.rul) + " cycles, 90% interval [" + fmt(now.lower) + ", " + fmt(now.upper) + "]";
}

function populateEngines(rows) {
  const select = el("inject-engine");
  const current = select.value;
  const units = [...new Map(rows.map((r) => [r.unit, r.engine])).entries()];
  if (select.options.length === units.length) return;
  select.replaceChildren(...units.map(([unit, label]) => node("option", {value: unit}, label)));
  if (current) select.value = current;
}

async function tick() {
  try {
    const fleet = await api("/fleet");
    track(fleet.fleet);
    renderFleet(fleet.fleet);
    populateEngines(fleet.fleet);
    draw();
    el("conn").textContent = "live";
    el("conn").className = "chip nominal";
  } catch (error) {
    el("conn").textContent = "no data: " + error.message;
    el("conn").className = "chip degraded";
  }
}

async function coverage() {
  try {
    const c = await api("/coverage");
    el("coverage").textContent = c.measured_coverage === null
      ? "collecting (" + c.n_samples + ")"
      : fmt(c.measured_coverage * 100, 1) + "% vs " + fmt(c.target_coverage * 100, 0) + "% target";
  } catch (error) {
    el("coverage").textContent = "unavailable";
  }
}

async function costs() {
  const panel = await api("/cost");
  const list = el("cost-rows");
  const max = Math.max(...panel.policies.map((p) => p.cost_per_engine));
  for (const p of panel.policies) {
    const kind = p.policy === panel.best ? "best" : p.policy === "run_to_failure" ? "rtf" : "";
    const li = node("li", {class: kind});
    const track = node("span", {class: "track"});
    const fill = node("span", {class: "fill"});
    fill.style.width = Math.max(1, (p.cost_per_engine / max) * 100) + "%";
    track.append(fill);
    li.append(node("span", {class: "mono"}, p.policy.replaceAll("_", " ")), track, node("span", {class: "num"}, fmt(p.cost_per_engine, 2)));
    li.title = p.n_planned + " planned, " + p.n_failures + " failures";
    list.append(li);
  }
  const m = panel.cost_model;
  el("cost-model").textContent = "Cost per engine, planned maintenance = 1. Unplanned failure "
    + m.failure_cost_ratio + "x, horizon " + m.horizon + " cycles, " + m.n_episodes
    + " engines. Read from " + panel.source + ", not recomputed.";
}

async function instruments() {
  const info = await api("/degraded");
  el("inject-channel").replaceChildren(...info.instruments.map((name, i) => node("option", {value: i}, name)));
}

async function inject(fault) {
  const body = {
    engine: Number(el("inject-engine").value),
    fault,
    channel: Number(el("inject-channel").value),
    from_cycle: 0,
  };
  try {
    const result = await api("/inject", {method: "POST", body: JSON.stringify(body), headers: {"Content-Type": "application/json"}});
    const engine = el("inject-engine").selectedOptions[0]?.textContent || String(body.engine);
    const sensor = el("inject-channel").selectedOptions[0]?.textContent || "";
    logEvent("injected", fault === "none" ? engine + ": fault cleared" : engine + ": injected " + fault.replace("_", " ") + " on " + sensor);
    const armed = Object.keys(result.armed);
    el("armed").textContent = armed.length ? "armed on unit " + armed.join(", ") : "nothing armed";
  } catch (error) {
    logEvent("degraded", "injection refused: " + error.message);
  }
}

el("inject-go").addEventListener("click", () => inject(el("inject-fault").value));
el("inject-clear").addEventListener("click", () => inject("none"));
el("follow").addEventListener("click", () => { follow = true; });

costs().catch(() => { el("cost-model").textContent = "cost panel unavailable"; });
instruments().catch(() => {});
tick();
coverage();
setInterval(tick, 1000);
setInterval(coverage, 3000);
"""

_STYLE = """
:root { color-scheme: dark; --bg:#0d1117; --panel:#151b23; --line:#2a313c; --text:#e6edf3;
  --dim:#8b949e; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
header { display:flex; align-items:center; gap:16px; padding:14px 20px;
  border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:17px; margin:0; font-weight:600; }
header .sub { color:var(--dim); }
.stats { display:flex; gap:18px; margin-left:auto; flex-wrap:wrap; align-items:center; }
.stat b { font-variant-numeric:tabular-nums; font-size:16px; margin-right:4px; }
main { display:grid; grid-template-columns:minmax(0, 1.25fr) minmax(0, 1fr); gap:16px; padding:16px 20px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; min-width:0; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
  margin:0 0 10px; font-weight:600; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--dim); font-weight:500; font-size:12px; }
tbody tr { cursor:pointer; } tr.selected td { background:#1c2533; }
.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.strong { font-weight:600; } .dim { color:var(--dim); }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; }
.bar { position:relative; height:12px; min-width:120px; background:#0b0f14; border-radius:6px; margin-top:4px; }
.band { position:absolute; top:0; bottom:0; border-radius:6px; background:rgba(88,166,255,.45); }
.band.alerting { background:rgba(210,153,34,.6); } .band.degraded { background:rgba(248,81,73,.55); }
.point { position:absolute; top:-2px; width:2px; height:16px; background:var(--text); }
.chip { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; border:1px solid; white-space:nowrap; }
.chip.nominal { color:var(--ok); border-color:rgba(63,185,80,.5); }
.chip.alerting { color:var(--warn); border-color:rgba(210,153,34,.6); }
.chip.degraded { color:var(--bad); border-color:rgba(248,81,73,.6); }
.why { color:var(--bad); font-size:12px; margin-top:3px; }
svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--line); stroke-width:1; } .axis { fill:var(--dim); font-size:11px; }
.axis.end { text-anchor:end; }
.band-area { fill:rgba(88,166,255,.25); stroke:none; }
.rul-line { fill:none; stroke:var(--accent); stroke-width:2; }
#chart-now { color:var(--dim); margin-top:6px; font-variant-numeric:tabular-nums; }
.controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
select, button { background:#0b0f14; color:var(--text); border:1px solid var(--line);
  border-radius:6px; padding:6px 10px; font:inherit; }
button { cursor:pointer; } button.danger { border-color:rgba(248,81,73,.7); color:var(--bad); }
button:focus-visible, select:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
#follow { font-size:11px; padding:2px 8px; margin-left:8px; text-transform:none; letter-spacing:0; }
#armed { color:var(--dim); font-size:12px; margin-top:8px; }
#events { list-style:none; margin:0; padding:0; max-height:220px; overflow:auto; }
#events li { display:flex; gap:10px; padding:4px 0; border-bottom:1px solid var(--line); font-size:13px; }
#events time { color:var(--dim); font-variant-numeric:tabular-nums; white-space:nowrap; }
#events .alerting span { color:var(--warn); } #events .degraded span { color:var(--bad); }
#events .nominal span { color:var(--ok); } #events .injected span { color:var(--accent); }
#cost-rows { list-style:none; margin:0; padding:0; }
#cost-rows li { display:grid; grid-template-columns:170px 1fr 52px; gap:10px; align-items:center; padding:3px 0; }
#cost-rows .track { display:block; height:10px; background:#0b0f14; border-radius:5px; overflow:hidden; }
#cost-rows .fill { display:block; height:100%; background:var(--accent); }
#cost-rows li.best .fill { background:var(--ok); } #cost-rows li.rtf .fill { background:var(--bad); }
#cost-model { color:var(--dim); font-size:12px; margin-top:8px; }
.stack { display:grid; gap:16px; align-content:start; min-width:0; }
@media (max-width: 900px) { main { grid-template-columns:1fr; padding:12px 16px; } }
"""

_BODY = """
<header>
  <h1>Prognos</h1><span class="sub">live fleet: remaining useful life with 90% conformal intervals</span>
  <div class="stats">
    <span class="stat"><b id="n-engines">0</b>engines</span>
    <span class="stat"><b id="n-alerting">0</b>maintenance due</span>
    <span class="stat"><b id="n-degraded">0</b>low confidence</span>
    <span class="stat">coverage <b id="coverage">—</b></span>
    <span id="conn" class="chip">connecting</span>
  </div>
</header>
<main>
  <div class="stack">
    <section>
      <h2>Fleet, worst first</h2>
      <table>
        <thead><tr><th>Engine</th><th class="num">Cycle</th><th class="num">RUL</th>
          <th class="num">Interval</th><th>Band (0 to 130 cycles)</th><th>Status</th></tr></thead>
        <tbody id="fleet-rows"></tbody>
      </table>
    </section>
    <section>
      <h2>Maintenance cost by policy</h2>
      <ul id="cost-rows"></ul>
      <div id="cost-model"></div>
    </section>
  </div>
  <div class="stack">
    <section>
      <h2>RUL over time: <span id="chart-title"></span><button id="follow">follow live</button></h2>
      <svg id="chart" viewBox="0 0 640 240" role="img"
        aria-label="Remaining useful life and its interval over time"></svg>
      <div id="chart-now"></div>
    </section>
    <section>
      <h2>Inject a fault</h2>
      <div class="controls">
        <label>Engine <select id="inject-engine"></select></label>
        <label>Fault <select id="inject-fault">
          <option value="sensor_stuck">sensor stuck</option>
          <option value="sensor_dropout">sensor dropout</option>
          <option value="value_out_of_range">value out of range</option>
          <option value="sensor_drift">sensor drift</option>
        </select></label>
        <label>Sensor <select id="inject-channel"></select></label>
        <button id="inject-go" class="danger">Inject</button>
        <button id="inject-clear">Clear</button>
      </div>
      <div id="armed">nothing armed</div>
    </section>
    <section>
      <h2>Events</h2>
      <ul id="events"></ul>
    </section>
  </div>
</main>
"""

PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    f"<title>Prognos fleet</title><style>{_STYLE}</style></head>"
    f"<body>{_BODY}<script>{_SCRIPT}</script></body></html>"
)


def _hash(source: str) -> str:
    """A CSP source hash for one inline block."""
    digest = hashlib.sha256(source.encode()).digest()
    return "'sha256-" + base64.b64encode(digest).decode() + "'"


#: Only this page's own script and style may run, and it may only talk to its own origin.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    f"script-src {_hash(_SCRIPT)}; "
    f"style-src {_hash(_STYLE)}; "
    "connect-src 'self'; img-src 'self' data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

__all__ = ["CONTENT_SECURITY_POLICY", "PAGE"]
