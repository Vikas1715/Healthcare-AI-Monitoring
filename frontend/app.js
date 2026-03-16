/* ═══════════════════════════════════════════════════════════════
   app.js  —  Healthcare AI Monitoring  (Flask backend edition)
   All API calls point to the Flask routes defined in main.py
   ═══════════════════════════════════════════════════════════════ */

const API = "http://localhost:8000";

// Registry of Chart.js instances so we can destroy before redraw
const _charts = {};

// ── Tiny helpers ─────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function showToast(msg, type = "success") {
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

function mkChart(id, config) {
  destroyChart(id);
  const ctx = $(id);
  if (!ctx) return;
  _charts[id] = new Chart(ctx, config);
}

// Shared chart palette
const C = {
  green:  "rgba(0,212,170,0.85)",
  blue:   "rgba(14,165,233,0.85)",
  amber:  "rgba(245,158,11,0.85)",
  red:    "rgba(239,68,68,0.85)",
  dim:    "rgba(100,116,139,0.5)",
};

// Inject common defaults into every Chart config
function mkCfg(cfg) {
  cfg.options = cfg.options || {};
  cfg.options.responsive          = true;
  cfg.options.maintainAspectRatio = false;
  cfg.options.plugins = Object.assign({
    legend: { labels: { color: "#94a3b8", font: { family: "IBM Plex Sans", size: 12 } } },
    tooltip: { titleFont: { family: "IBM Plex Mono" }, bodyFont: { family: "IBM Plex Sans" } },
  }, cfg.options.plugins || {});

  if (cfg.type !== "pie" && cfg.type !== "doughnut") {
    cfg.options.scales = Object.assign({
      x: { ticks: { color: "#64748b" }, grid: { color: "rgba(30,45,69,0.6)" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "rgba(30,45,69,0.6)" } },
    }, cfg.options.scales || {});
  }
  return cfg;
}

function errHtml(msg) {
  return `<div class="card">
    <p style="color:var(--danger);">⚠ ${msg}</p>
    <p class="text-dim mt-8">Please upload a dataset first (Section ①), then retry.</p>
  </div>`;
}

// ── Navigation ────────────────────────────────────────────────────
function showSection(id) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".nav-links button").forEach(b => b.classList.remove("active"));
  $(id).classList.add("active");
  const idx = parseInt(id.replace("s", "")) - 1;
  document.querySelectorAll(".nav-links button")[idx].classList.add("active");

  // Lazy-load each section on first visit
  const loaders = {
    s2: loadMarginal,
    s3: () => loadJoint(),
    s4: () => loadConditional(),
    s5: loadMLE,
    s6: () => loadKL(),
    s7: loadMarkov,
    s8: loadHMM,
    s9: loadGenAI,
    s10: renderDL,
  };
  if (loaders[id]) loaders[id]();
}

// ══════════════════════════════════════════════════════════════════
// SECTION 1 — UPLOAD & EXPLORATION
// ══════════════════════════════════════════════════════════════════
function handleDrop(e) {
  e.preventDefault();
  $("upload-zone").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (!file) return;
  $("upload-status").innerHTML =
    `<div class="loader"><div class="spinner"></div> Uploading and analysing…</div>`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res  = await fetch(`${API}/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    renderExploration(data);
    showToast(`✓ Loaded ${data.summary.num_records} records`);
    $("upload-status").innerHTML =
      `<p class="text-dim" style="margin-top:8px;">✓ Dataset uploaded — <strong style="color:var(--accent)">${data.summary.num_records} records, ${data.summary.num_features} features</strong></p>`;
  } catch (err) {
    showToast(err.message, "error");
    $("upload-status").innerHTML =
      `<p style="color:var(--danger);margin-top:8px;">✗ ${err.message}</p>`;
  }
}

function renderExploration(data) {
  const s = data.summary;
  $("explore-results").classList.remove("hidden");

  // Stat pills
  $("stat-pills").innerHTML = [
    { value: s.num_records,    label: "Total Records" },
    { value: s.death_count,    label: "Death Events"  },
    { value: s.survival_count, label: "Survivors"     },
    { value: s.num_features,   label: "Features"      },
  ].map(p => `
    <div class="stat-pill">
      <div class="value">${p.value}</div>
      <div class="label">${p.label}</div>
    </div>`).join("");

  // Preview table
  if (data.preview && data.preview.length) {
    const cols = Object.keys(data.preview[0]);
    $("preview-table").querySelector("thead").innerHTML =
      `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
    $("preview-table").querySelector("tbody").innerHTML =
      data.preview.map(row =>
        `<tr>${cols.map(c => {
          let v = row[c];
          if (c === "DEATH_EVENT") {
            v = v == 1
              ? `<span class="badge badge-red">Death</span>`
              : `<span class="badge badge-green">Alive</span>`;
          }
          return `<td>${v !== null && v !== undefined ? v : "—"}</td>`;
        }).join("")}</tr>`
      ).join("");
  }

  // Stats table
  $("stats-table").querySelector("tbody").innerHTML =
    Object.entries(s.stats).map(([col, st]) =>
      `<tr><td>${col}</td><td>${st.mean}</td><td>${st.std}</td><td>${st.min}</td><td>${st.max}</td></tr>`
    ).join("");

  // Scatter charts
  renderScatter("chart-age", data.scatter.age,                "Age (years)");
  renderScatter("chart-ef",  data.scatter.ejection_fraction,  "Ejection Fraction (%)");
  renderScatter("chart-sc",  data.scatter.serum_creatinine,   "Serum Creatinine (mg/dL)");

  // Correlation heatmap
  renderHeatmap(data.correlation);
}

function renderScatter(id, scatterData, xLabel) {
  if (!scatterData) return;
  mkChart(id, mkCfg({
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Survived",
          data: scatterData.alive.map(v => ({ x: v, y: (Math.random() - 0.5) * 0.3 })),
          backgroundColor: C.green, pointRadius: 3,
        },
        {
          label: "Died",
          data: scatterData.dead.map(v => ({ x: v, y: 1 + (Math.random() - 0.5) * 0.3 })),
          backgroundColor: C.red, pointRadius: 3,
        },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: xLabel, color: "#64748b" }, ticks: { color: "#64748b" }, grid: { color: "rgba(30,45,69,0.6)" } },
        y: { ticks: { color: "#64748b", callback: v => v < 0.5 ? "Survived" : "Died" }, grid: { color: "rgba(30,45,69,0.6)" } },
      },
    },
  }));
}

function renderHeatmap(corr) {
  const labels = corr.labels;
  const matrix = corr.matrix;

  let html = `<table style="border-collapse:collapse;">
    <thead><tr><th style="min-width:80px;"></th>`;
  labels.forEach(l => {
    const abbr = l.replace(/_/g, " ").split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 4);
    html += `<th style="color:var(--text-muted);padding:2px 3px;font-size:0.6rem;" title="${l}">${abbr}</th>`;
  });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr><td style="color:var(--text-muted);padding:2px 6px;font-size:0.6rem;white-space:nowrap;">${labels[i].slice(0,12)}</td>`;
    row.forEach(val => {
      const abs = Math.abs(val);
      const r = val > 0 ? Math.round(abs * 80) : 0;
      const b = val < 0 ? Math.round(abs * 80) : 0;
      const bg = `rgba(${r},${40 + Math.round(abs * 40)},${b + 40},${0.3 + abs * 0.7})`;
      const color = abs > 0.4 ? "#fff" : "#94a3b8";
      html += `<td style="background:${bg};color:${color};text-align:center;padding:3px;width:32px;height:28px;border-radius:2px;font-size:0.58rem;font-family:'IBM Plex Mono';">${val.toFixed(2)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  $("corr-heatmap").innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════
// SECTION 2 — MARGINAL PROBABILITY   →  GET /marginal
// ══════════════════════════════════════════════════════════════════
async function loadMarginal() {
  const el = $("marginal-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Computing…</div>`;
  try {
    const r = await fetch(`${API}/marginal`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderMarginal(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderMarginal(d) {
  $("marginal-content").innerHTML = `
    <div class="grid-3">
      <div class="card text-center">
        <div class="card-title" style="justify-content:center;">P(Death = 1)</div>
        <div class="big-number">${(d.p_death * 100).toFixed(1)}%</div>
        <div class="text-dim mt-8">${d.deaths} of ${d.total} patients</div>
      </div>
      <div class="card text-center">
        <div class="card-title" style="justify-content:center;">P(Survival)</div>
        <div class="big-number" style="color:var(--accent2)">${(d.p_survival * 100).toFixed(1)}%</div>
        <div class="text-dim mt-8">${d.survival_count} of ${d.total} patients</div>
      </div>
      <div class="card">
        <div class="card-title">Outcome Distribution</div>
        <div class="chart-wrap"><canvas id="chart-marginal"></canvas></div>
      </div>
    </div>
    <div class="card mt-20">
      <div class="card-title">Formula</div>
      <div class="result-highlight">P(Death = 1) = deaths / total = ${d.deaths} / ${d.total} = <strong>${d.p_death}</strong></div>
    </div>
    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is Marginal Probability?</strong><br/>
        Marginal probability is the probability of a single event without any conditions — it is computed by summing (or integrating) the joint distribution over all values of all other variables. Here, <em>P(Death = 1) = ${d.p_death}</em>, meaning roughly <strong>${(d.p_death*100).toFixed(1)}%</strong> of patients in this cohort died during the follow-up period. This serves as the baseline mortality estimate for the entire population, used as a benchmark against which conditional and joint probabilities are compared.
      </div>
    </div>`;

  mkChart("chart-marginal", mkCfg({
    type: "doughnut",
    data: {
      labels: ["Death Event", "Survival"],
      datasets: [{ data: [d.p_death, d.p_survival], backgroundColor: [C.red, C.green], borderWidth: 0 }],
    },
    options: { cutout: "65%", plugins: { legend: { position: "bottom" } } },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 3 — JOINT PROBABILITY   →  GET /joint?age_threshold=60
// ══════════════════════════════════════════════════════════════════
async function loadJoint() {
  const age = parseInt($("joint-age").value) || 60;
  const el  = $("joint-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Computing…</div>`;
  try {
    const r = await fetch(`${API}/joint?age_threshold=${age}`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderJoint(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderJoint(d) {
  $("joint-content").innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Probability Comparison</div>
        <div class="chart-wrap"><canvas id="chart-joint"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Results</div>
        <table><thead><tr><th>Event</th><th>Probability</th></tr></thead>
        <tbody>
          <tr><td>P(Age &gt; ${d.age_threshold})</td><td class="mono">${d.p_age_gt_threshold}</td></tr>
          <tr><td>P(Death = 1)</td><td class="mono">${d.p_death}</td></tr>
          <tr><td><strong>P(Age &gt; ${d.age_threshold} ∧ Death = 1)</strong></td>
              <td class="mono" style="color:var(--accent)">${d.p_joint}</td></tr>
          <tr><td>Joint count</td><td class="mono">${d.joint_count} / ${d.total}</td></tr>
        </tbody></table>
        <div class="result-highlight mt-16">
          P(A ∩ B) = ${d.p_joint} &nbsp;≈ ${(d.p_joint * 100).toFixed(1)}% of patients
        </div>
      </div>
    </div>
    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is Joint Probability?</strong><br/>
        Joint probability P(A ∩ B) measures how likely two events occur <em>simultaneously</em>. Here, P(Age &gt; ${d.age_threshold} AND Death = 1) = <strong>${d.p_joint}</strong>. If events were independent, we would expect P(A) × P(B) = ${(d.p_age_gt_threshold * d.p_death).toFixed(4)}. The actual joint probability ${d.p_joint > d.p_age_gt_threshold * d.p_death ? "exceeds" : "is below"} this, indicating the two events are <strong>${d.p_joint > d.p_age_gt_threshold * d.p_death ? "positively associated" : "not strongly associated"}</strong>. Clinically, joint probability allows risk stratification — identifying which patient sub-groups carry the highest compound risk, enabling targeted interventions.
      </div>
    </div>`;

  mkChart("chart-joint", mkCfg({
    type: "bar",
    data: {
      labels: [`P(Age>${d.age_threshold})`, "P(Death=1)", "P(Joint A∩B)"],
      datasets: [{
        label: "Probability",
        data: [d.p_age_gt_threshold, d.p_death, d.p_joint],
        backgroundColor: [C.blue, C.red, C.amber],
        borderRadius: 6,
      }],
    },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 4 — CONDITIONAL   →  GET /conditional?condition=...
// ══════════════════════════════════════════════════════════════════
async function loadConditional() {
  const cond = $("cond-select").value;
  const el   = $("cond-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Computing…</div>`;
  try {
    const r = await fetch(`${API}/conditional?condition=${cond}`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderConditional(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderConditional(d) {
  const labels = {
    age_gt_60: "Age > 60", diabetes: "Diabetes",
    high_blood_pressure: "High BP", anaemia: "Anaemia", smoking: "Smoking",
  };
  const lift = (d.p_death_given_cond / d.p_overall_death).toFixed(2);
  const liftColor = d.p_death_given_cond > d.p_overall_death ? "var(--danger)" : "var(--accent)";

  $("cond-content").innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">P(Death | Condition) — All Conditions</div>
        <div class="chart-wrap chart-wrap-tall"><canvas id="chart-cond"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Selected: ${labels[d.condition] || d.condition}</div>
        <div class="stat-pill text-center mt-8">
          <div class="value">${(d.p_death_given_cond * 100).toFixed(1)}%</div>
          <div class="label">P(Death | ${labels[d.condition]})</div>
        </div>
        <div class="mt-16">
          <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Overall P(Death)</td><td class="mono">${d.p_overall_death}</td></tr>
            <tr><td>P(Death | Condition)</td><td class="mono" style="color:var(--accent)">${d.p_death_given_cond}</td></tr>
            <tr><td>Patients with condition</td><td class="mono">${d.condition_count} / ${d.total}</td></tr>
            <tr><td>Risk Lift</td><td class="mono" style="color:${liftColor}">${lift}×</td></tr>
          </tbody></table>
        </div>
      </div>
    </div>
    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is Conditional Probability?</strong><br/>
        P(B | A) = P(A ∩ B) / P(A) — the probability of B given that A has already occurred. Here, P(Death | ${labels[d.condition]}) = <strong>${d.p_death_given_cond}</strong>. The <em>Risk Lift</em> of ${lift}× means this condition ${parseFloat(lift) > 1 ? "increases" : "decreases"} the death probability by ${Math.abs((parseFloat(lift)-1)*100).toFixed(0)}% compared to the population baseline. Clinicians use conditional probabilities to triage high-risk patients and personalise treatment protocols.
      </div>
    </div>`;

  const compLabels = Object.keys(d.comparison).map(k => labels[k] || k);
  const compVals   = Object.values(d.comparison);

  mkChart("chart-cond", mkCfg({
    type: "bar",
    data: {
      labels: compLabels,
      datasets: [
        {
          label: "P(Death | Condition)",
          data: compVals,
          backgroundColor: compVals.map(v => v > d.p_overall_death ? C.red : C.blue),
          borderRadius: 6,
        },
        {
          label: "Overall P(Death)",
          data: compLabels.map(() => d.p_overall_death),
          type: "line",
          borderColor: C.amber,
          borderDash: [6, 3],
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 5 — MLE   →  GET /mle
// ══════════════════════════════════════════════════════════════════
async function loadMLE() {
  const el = $("mle-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Computing…</div>`;
  try {
    const r = await fetch(`${API}/mle`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderMLE(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderMLE(d) {
  const llAtMLE = (d.k * Math.log(d.p_mle) + (d.n - d.k) * Math.log(1 - d.p_mle)).toFixed(2);

  $("mle-content").innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Log-Likelihood Curve  ℓ(p)</div>
        <div class="chart-wrap chart-wrap-tall"><canvas id="chart-mle"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">MLE Result</div>
        <div class="text-center" style="padding:24px 0;">
          <div class="big-number">${d.p_mle}</div>
          <div class="text-dim mt-8">MLE estimate  p̂ = k / n</div>
        </div>
        <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Sample size (n)</td><td class="mono">${d.n}</td></tr>
          <tr><td>Death events (k)</td><td class="mono">${d.k}</td></tr>
          <tr><td>p̂ = k / n</td><td class="mono" style="color:var(--accent)">${d.p_mle}</td></tr>
          <tr><td>ℓ(p̂) — max log-likelihood</td><td class="mono">${llAtMLE}</td></tr>
        </tbody></table>
      </div>
    </div>
    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is Maximum Likelihood Estimation?</strong><br/>
        MLE finds the parameter value θ̂ that maximises the probability of observing the data we have. For a Bernoulli(p) model, the likelihood function is <em>L(p) = pᵏ(1−p)ⁿ⁻ᵏ</em>. Taking the log and differentiating gives the MLE: <strong>p̂ = k/n = ${d.k}/${d.n} = ${d.p_mle}</strong>. The log-likelihood curve above peaks exactly at p̂, confirming it is the global maximum. MLE is foundational to logistic regression, survival analysis, and most ML model training (cross-entropy loss is negative log-likelihood under a Bernoulli assumption).
      </div>
    </div>`;

  const step = 4;
  const pvs  = d.p_values.filter((_, i) => i % step === 0);
  const ll   = d.log_likelihood.filter((_, i) => i % step === 0);

  mkChart("chart-mle", mkCfg({
    type: "line",
    data: {
      labels: pvs.map(v => v.toFixed(2)),
      datasets: [{
        label: "Log-Likelihood ℓ(p)",
        data: ll,
        borderColor: C.green,
        backgroundColor: "rgba(0,212,170,0.08)",
        fill: true, tension: 0.4, pointRadius: 0,
      }],
    },
    options: {
      scales: {
        x: { title: { display: true, text: "p", color: "#64748b" }, ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "rgba(30,45,69,0.6)" } },
        y: { title: { display: true, text: "Log-Likelihood", color: "#64748b" }, ticks: { color: "#64748b" }, grid: { color: "rgba(30,45,69,0.6)" } },
      },
    },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 6 — KL DIVERGENCE   →  GET /kl-divergence?assumed_p=0.30
// ══════════════════════════════════════════════════════════════════
async function loadKL() {
  const assumed = parseFloat($("kl-assumed").value) || 0.30;
  const el = $("kl-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Computing…</div>`;
  try {
    const r = await fetch(`${API}/kl-divergence?assumed_p=${assumed}`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderKL(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderKL(d) {
  const kv = d.kl_divergence;
  const level = kv < 0.01
    ? { badge: "badge-green",  text: "Low — distributions closely match" }
    : kv < 0.05
      ? { badge: "badge-yellow", text: "Moderate — noticeable information gap" }
      : { badge: "badge-red",    text: "High — significant mismatch detected" };

  $("kl-content").innerHTML = `
    <div class="grid-2">
      <div class="card">
        <div class="card-title">KL Divergence vs Assumed P(Death)</div>
        <div class="chart-wrap chart-wrap-tall"><canvas id="chart-kl"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Result</div>
        <div class="text-center" style="padding:24px 0;">
          <div class="big-number" style="color:var(--accent3)">${kv}</div>
          <div class="text-dim mt-8">KL( P_observed ‖ P_assumed )</div>
        </div>
        <table><thead><tr><th>Distribution</th><th>P(Death)</th></tr></thead>
        <tbody>
          <tr><td>Observed (dataset)</td><td class="mono" style="color:var(--accent)">${d.p_observed}</td></tr>
          <tr><td>Assumed (hospital)</td><td class="mono" style="color:var(--accent3)">${d.p_assumed}</td></tr>
          <tr><td>KL Divergence</td><td class="mono" style="color:var(--accent3)">${kv}</td></tr>
        </tbody></table>
        <div class="mt-16"><span class="badge ${level.badge}">${level.text}</span></div>
      </div>
    </div>
    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is KL Divergence?</strong><br/>
        KL(P‖Q) = Σ P(x) log[P(x)/Q(x)] measures the information lost when distribution Q is used to approximate the true distribution P. A value of <strong>${kv}</strong> nats means the hospital's assumed rate (${d.p_assumed}) causes that much information loss relative to the true rate (${d.p_observed}). KL = 0 implies perfect agreement. High KL divergence in a clinical context means hospital protocols calibrated to the assumed probability will systematically under- or over-treat patients. The curve above shows KL as a function of the assumed p — it is always minimised at the true observed probability.
      </div>
    </div>`;

  const step = 4;
  const qs  = d.q_range.filter((_, i) => i % step === 0);
  const kls = d.kl_curve.filter((_, i) => i % step === 0);

  mkChart("chart-kl", mkCfg({
    type: "line",
    data: {
      labels: qs.map(v => v.toFixed(2)),
      datasets: [{
        label: "KL Divergence",
        data: kls,
        borderColor: C.amber,
        backgroundColor: "rgba(245,158,11,0.07)",
        fill: true, tension: 0.4, pointRadius: 0,
      }],
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Assumed P(Death)", color: "#64748b" }, ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "rgba(30,45,69,0.6)" } },
        y: { title: { display: true, text: "KL Divergence (nats)", color: "#64748b" }, ticks: { color: "#64748b" }, grid: { color: "rgba(30,45,69,0.6)" } },
      },
    },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 7 — MARKOV CHAIN   →  GET /markov/matrix  POST /markov/simulate
// ══════════════════════════════════════════════════════════════════
async function loadMarkov() {
  const el = $("markov-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Building model…</div>`;
  try {
    const r = await fetch(`${API}/markov/matrix`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderMarkov(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderMarkov(d) {
  const names = d.state_names;
  const T     = d.matrix;
  const stateColors = ["var(--accent)", "var(--accent3)", "var(--danger)"];

  const matrixRows = T.map((row, i) =>
    `<tr>
      <td style="color:${stateColors[i]};font-weight:600;">${names[i]}</td>
      ${row.map(v =>
        `<td style="text-align:center;font-family:'IBM Plex Mono';${v >= 0.5 ? 'color:var(--accent);font-weight:600;' : ''}">${v.toFixed(3)}</td>`
      ).join("")}
    </tr>`
  ).join("");

  $("markov-content").innerHTML = `
    <div class="grid-3 mb-20">
      ${Object.entries(d.state_distribution).map(([name, prob], i) => `
        <div class="stat-pill text-center">
          <div class="value" style="color:${stateColors[i]}">${(prob * 100).toFixed(1)}%</div>
          <div class="label">${name} (${d.state_counts[name]} patients)</div>
        </div>`).join("")}
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Transition Probability Matrix T[i→j]</div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>From \ To</th>${names.map(n => `<th style="text-align:center">${n}</th>`).join("")}</tr></thead>
            <tbody>${matrixRows}</tbody>
          </table>
        </div>
        <p class="text-dim mt-12">Each row sums to 1.0. Values ≥ 0.50 are highlighted.</p>
      </div>

      <div class="card">
        <div class="card-title">State Distribution</div>
        <div class="chart-wrap" style="height:230px;"><canvas id="chart-markov-dist"></canvas></div>
        <div class="markov-states mt-12">
          ${names.map((name, i) => {
            const cls = ["healthy","at-risk","critical"][i];
            return `<div class="state-node ${cls}" onclick="$('markov-start').value=${i}" title="Click to select as start state">
              <span style="font-size:1.4rem;">${["💚","🟡","🔴"][i]}</span>
              <span>${name}</span>
            </div>`;
          }).join('<div style="font-size:1.3rem;color:var(--text-dim)">→</div>')}
        </div>
      </div>
    </div>

    <div class="card mt-20">
      <div class="card-title">Patient Trajectory Simulation</div>
      <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">
        <label class="text-muted" style="font-size:0.88rem;">Start State:</label>
        <select id="markov-start">
          <option value="0">Healthy</option>
          <option value="1">At Risk</option>
          <option value="2">Critical</option>
        </select>
        <label class="text-muted" style="font-size:0.88rem;">Steps:</label>
        <input type="number" id="markov-steps" value="10" min="5" max="50" style="width:80px;" />
        <button class="btn btn-primary" onclick="runMarkovSim()">▶ Simulate</button>
      </div>
      <div id="markov-sim-result"></div>
    </div>

    <div class="card mt-20">
      <div class="explain-box">
        <strong>What is a Markov Chain?</strong><br/>
        A Markov chain models a system transitioning between states where the <em>next state depends only on the current state</em> (memoryless / Markov property). The transition matrix T defines P(next state = j | current state = i). In healthcare, states represent patient health levels: Healthy → At Risk → Critical. The matrix is estimated from the dataset by assigning each patient a state (based on ejection fraction and serum creatinine) and counting transitions between consecutive time-sorted records. Markov models are used to project long-term disease progression, estimate care costs, and evaluate preventive interventions.
      </div>
    </div>`;

  mkChart("chart-markov-dist", mkCfg({
    type: "bar",
    data: {
      labels: names,
      datasets: [{
        label: "Patient Count",
        data: Object.values(d.state_counts),
        backgroundColor: [C.green, C.amber, C.red],
        borderRadius: 6,
      }],
    },
  }));
}

async function runMarkovSim() {
  const state = parseInt($("markov-start").value);
  const steps = parseInt($("markov-steps").value) || 10;
  const el    = $("markov-sim-result");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Simulating…</div>`;
  try {
    const r = await fetch(`${API}/markov/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ current_state: state, steps }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);

    const traj  = d.trajectory;
    const idxs  = traj.indices;
    const named = traj.trajectory;
    const cls   = ["badge-green", "badge-yellow", "badge-red"];
    const next  = d.next_state_prediction;

    const strip = named.map((name, i) =>
      `<span class="traj-node ${cls[idxs[i]]}">${i}: ${name}</span>`
    ).join(" → ");

    el.innerHTML = `
      <div class="result-highlight mb-16">
        Next state → <strong>${next.next_state}</strong> &nbsp;|&nbsp;
        P(Healthy)=${next.probabilities.Healthy}
        P(At Risk)=${next.probabilities["At Risk"]}
        P(Critical)=${next.probabilities.Critical}
      </div>
      <div class="card-title">Simulated ${steps}-step Patient Trajectory</div>
      <div class="trajectory-strip mt-8">${strip}</div>`;
  } catch (e) {
    el.innerHTML = errHtml(e.message);
  }
}

// ══════════════════════════════════════════════════════════════════
// SECTION 8 — HMM   →  GET /hmm/predict
// ══════════════════════════════════════════════════════════════════
async function loadHMM() {
  const el = $("hmm-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Running HMM…</div>`;
  try {
    const r = await fetch(`${API}/hmm/predict`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderHMM(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderHMM(d) {
  const stateColors = ["var(--accent)", "var(--accent3)", "var(--danger)"];
  const total = Object.values(d.state_frequencies).reduce((a, b) => a + b, 0);

  const emRows = d.emission_matrix.map((row, i) =>
    `<tr>
      <td style="color:${stateColors[i]};font-weight:600;">${d.hidden_states[i]}</td>
      ${row.map(v => `<td style="text-align:center;font-family:'IBM Plex Mono'">${v.toFixed(2)}</td>`).join("")}
    </tr>`
  ).join("");

  const predRows = d.sample_predictions.map(p =>
    `<tr>
      <td class="mono">${p.index}</td>
      <td>${p.observation}</td>
      <td style="color:${p.hidden_state === "Healthy" ? "var(--accent)" : p.hidden_state === "At Risk" ? "var(--accent3)" : "var(--danger)"};font-weight:600;">${p.hidden_state}</td>
    </tr>`
  ).join("");

  $("hmm-content").innerHTML = `
    <div class="grid-2 mb-20">
      <div class="card">
        <div class="card-title">Emission Probability Matrix  P(obs | hidden state)</div>
        <table>
          <thead><tr><th>Hidden State</th>${d.obs_names.map(o => `<th style="text-align:center">${o}</th>`).join("")}</tr></thead>
          <tbody>${emRows}</tbody>
        </table>
        <div class="explain-box mt-16">
          <strong>Hidden vs Observed States</strong><br/>
          <em>Hidden states</em> — the true underlying health condition (Healthy / At Risk / Critical) — cannot be directly measured. We can only observe <em>symptoms</em> (Normal / Mild / Severe). The emission matrix gives P(observed symptom | true hidden state). Viterbi decoding then reconstructs the most likely sequence of hidden states given the observed symptom sequence.<br/><br/>
          <strong>Method:</strong> ${d.method}
        </div>
      </div>
      <div class="card">
        <div class="card-title">Decoded Hidden State Distribution (${total} patients)</div>
        <div class="chart-wrap"><canvas id="chart-hmm"></canvas></div>
        <table class="mt-16">
          <thead><tr><th>State</th><th>Count</th><th>Share</th></tr></thead>
          <tbody>${d.hidden_states.map((s, i) => `
            <tr>
              <td style="color:${stateColors[i]};font-weight:600;">${s}</td>
              <td class="mono">${d.state_frequencies[s]}</td>
              <td class="mono">${((d.state_frequencies[s] / total) * 100).toFixed(1)}%</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Sample Viterbi Decoded Predictions (first ${d.sample_predictions.length} patients)</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Observed Symptoms</th><th>Decoded Hidden State</th></tr></thead>
          <tbody>${predRows}</tbody>
        </table>
      </div>
    </div>

    <div class="card mt-20">
      <div class="explain-box">
        <strong>Why use an HMM for patient monitoring?</strong><br/>
        In clinical settings, the true patient condition is hidden — doctors observe lab values, symptoms, and vital signs, not the underlying disease state directly. The HMM models this: a patient's <em>hidden</em> health state generates <em>observable</em> symptoms with known probabilities (emission matrix), while health states evolve over time following the Markov transition matrix. This framework is used in: continuous glucose monitoring, ICU patient deterioration alerts, ECG rhythm analysis, and longitudinal disease progression modelling.
      </div>
    </div>`;

  mkChart("chart-hmm", mkCfg({
    type: "doughnut",
    data: {
      labels: d.hidden_states,
      datasets: [{ data: Object.values(d.state_frequencies), backgroundColor: [C.green, C.amber, C.red], borderWidth: 0 }],
    },
    options: { cutout: "58%", plugins: { legend: { position: "bottom" } } },
  }));
}

// ══════════════════════════════════════════════════════════════════
// SECTION 9 — GENERATIVE AI   →  GET /generative/synthetic
// ══════════════════════════════════════════════════════════════════
async function loadGenAI() {
  const el = $("genai-content");
  el.innerHTML = `<div class="loader"><div class="spinner"></div> Generating synthetic data…</div>`;
  try {
    const r = await fetch(`${API}/generative/synthetic?n_samples=20`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    renderGenAI(d);
  } catch (e) { el.innerHTML = errHtml(e.message); }
}

function renderGenAI(d) {
  const apps = [
    { icon: "🖼️", title: "Synthetic Medical Image Generation",
      body: "GANs (StyleGAN, DCGAN) generate realistic X-ray, MRI, and CT images from noise vectors. Synthetic images augment rare-disease training sets while preserving patient privacy. Conditional GANs can generate pathology-specific images on demand — e.g., synthesising diabetic retinopathy grades." },
    { icon: "💊", title: "Drug Discovery & Molecule Design",
      body: "Variational autoencoders (VAEs) and diffusion models generate novel molecular structures with desired pharmacological properties. RNN-based models (REINVENT, ChemFormer) optimise lead compounds iteratively. AlphaFold 3 predicts protein-ligand structures, cutting lab screening timelines from years to weeks." },
    { icon: "📋", title: "Clinical Report Generation",
      body: "Fine-tuned LLMs (GPT-4, Med-PaLM 2, Llama-Med) convert structured imaging data and lab values into natural-language radiology and pathology reports, auto-code ICD diagnoses, and draft discharge summaries — reducing clinician documentation load by up to 40%." },
    { icon: "📊", title: "Patient Data Augmentation",
      body: "SMOTE, CTGAN, and TVAE synthesise minority-class patient records to fix class imbalance in predictive models. Synthetic EHR data can be shared between hospitals for federated learning without violating HIPAA/GDPR, enabling multi-institutional model training." },
  ];

  const previewCols = Object.keys(d.synthetic[0] || {});

  $("genai-content").innerHTML = `
    <div class="grid-2 mb-20">
      ${apps.map(a => `
        <div class="card">
          <div style="font-size:2.2rem;margin-bottom:8px;">${a.icon}</div>
          <div class="card-title">${a.title}</div>
          <p style="font-size:0.86rem;color:var(--text-muted);line-height:1.75;">${a.body}</p>
        </div>`).join("")}
    </div>

    <div class="card mb-20">
      <div class="card-title">Statistical Comparison: Real vs Synthetic Data</div>
      <table>
        <thead><tr><th>Feature</th><th>Real Mean</th><th>Real Std</th><th style="color:var(--accent)">Synth Mean</th><th style="color:var(--accent)">Synth Std</th></tr></thead>
        <tbody>
          ${Object.entries(d.real_stats).map(([feat, rs]) => {
            const ss = d.synth_stats[feat] || {};
            return `<tr>
              <td>${feat}</td>
              <td class="mono">${rs.mean}</td><td class="mono">${rs.std}</td>
              <td class="mono" style="color:var(--accent)">${ss.mean || "—"}</td>
              <td class="mono" style="color:var(--accent)">${ss.std || "—"}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
      <p class="text-dim mt-12">Synthetic statistics should approximate real statistics. Small deviations are normal at n=${d.n_samples} samples. Larger samples converge more closely.</p>
    </div>

    <div class="card">
      <div class="card-title">Generated Synthetic Patient Records (first 8 of ${d.n_samples})</div>
      <div class="table-wrap">
        <table>
          <thead><tr>${previewCols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
          <tbody>
            ${d.synthetic.slice(0, 8).map(row =>
              `<tr>${previewCols.map(c => `<td class="mono" style="font-size:0.78rem;">${row[c] !== null && row[c] !== undefined ? row[c] : "—"}</td>`).join("")}</tr>`
            ).join("")}
          </tbody>
        </table>
      </div>
      <div class="explain-box mt-16">
        <strong>How is synthetic data generated here?</strong><br/>
        Each continuous feature (age, ejection fraction, serum creatinine) is sampled from a Gaussian N(μ, σ) fitted to the real column. Binary/categorical features are sampled with the observed class proportions. More advanced methods — CTGAN, TVAE — learn the full joint distribution including inter-feature correlations. The generated records above contain no real patient information yet statistically resemble the original cohort.
      </div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════════
// SECTION 10 — DEEP LEARNING  (static render, no API call)
// ══════════════════════════════════════════════════════════════════
function renderDL() {
  const layers_lstm = [
    { name: "Input Layer",            desc: "Sequential time-series  (T timesteps × F features)",              color: "var(--accent2)" },
    { name: "LSTM Layer 1 — 128 units", desc: "Forget / Input / Output gates learn long-range temporal patterns", color: "var(--accent)"  },
    { name: "Dropout — 0.30",          desc: "Regularisation: randomly zero 30% of activations per batch",      color: "var(--dim)"     },
    { name: "LSTM Layer 2 — 64 units",  desc: "Hierarchical temporal feature extraction",                        color: "var(--accent)"  },
    { name: "Dense — 32, ReLU",        desc: "Non-linear projection to compact representation",                  color: "var(--accent3)" },
    { name: "Output — 1, Sigmoid",     desc: "P(critical deterioration within 24h) ∈ [0, 1]",                   color: "var(--danger)"  },
  ];

  const layers_gru = [
    { name: "Input Layer",            desc: "ECG signal or heart-rate stream  (T × 1)",                         color: "var(--accent2)" },
    { name: "GRU Layer 1 — 128 units", desc: "Update + Reset gates; fewer params than LSTM, faster training",   color: "var(--accent)"  },
    { name: "Dropout — 0.30",          desc: "Regularisation layer",                                             color: "var(--dim)"     },
    { name: "GRU Layer 2 — 64 units",  desc: "Deeper gated recurrent processing",                               color: "var(--accent)"  },
    { name: "GlobalAvgPool1D",         desc: "Collapse temporal axis → fixed-length vector",                     color: "var(--accent3)" },
    { name: "Output — 1, Sigmoid",     desc: "P(arrhythmia) or mortality risk score",                           color: "var(--danger)"  },
  ];

  const layers_transformer = [
    { name: "Input Embeddings + Positional Encoding", desc: "Dense vectors + positional info injected into every token", color: "var(--accent2)" },
    { name: "Multi-Head Self-Attention ×6",           desc: "Each head attends to different clinical relationships; global context captured in O(1) depth", color: "var(--accent)" },
    { name: "Feed-Forward Network (×6)",              desc: "Position-wise dense layers with GELU activation",           color: "var(--accent3)" },
    { name: "Layer Norm + Residual (×12)",            desc: "Stable gradients through deep stacks",                      color: "var(--dim)"    },
    { name: "Classification Head",                    desc: "CLS token → mortality / ICD code / readmission prediction", color: "var(--danger)" },
  ];

  const arch = layers =>
    layers.map((l, i, arr) => `
      <div class="arch-layer">
        <div class="arch-layer-name" style="color:${l.color}">${l.name}</div>
        <div class="arch-layer-desc">${l.desc}</div>
      </div>
      ${i < arr.length - 1 ? '<div class="arch-arrow">↓</div>' : ''}`
    ).join("");

  $("dl-content").innerHTML = `
    <div class="grid-2 mb-20">
      <div class="card">
        <div class="card-title">LSTM — ICU Early Warning System</div>
        <div class="arch-diagram">${arch(layers_lstm)}</div>
        <div class="explain-box mt-16">
          <strong>LSTM (Long Short-Term Memory)</strong><br/>
          LSTMs solve the vanishing-gradient problem of vanilla RNNs through three learned gates: <em>forget</em> (discard irrelevant history), <em>input</em> (write new information to cell state), and <em>output</em> (expose cell state to hidden state). This allows gradients to flow through hundreds of timesteps. Healthcare applications: hourly vital-sign streams → 24h mortality prediction; medication event logs → adverse drug event forecasting; continuous glucose monitoring → hypoglycaemia alerts.
        </div>
      </div>
      <div class="card">
        <div class="card-title">GRU — Real-Time ECG / Arrhythmia Detection</div>
        <div class="arch-diagram">${arch(layers_gru)}</div>
        <div class="explain-box mt-16">
          <strong>GRU (Gated Recurrent Unit)</strong><br/>
          GRUs merge the cell and hidden state into one vector and use only two gates (update + reset), yielding 25–33% fewer parameters than LSTM while achieving comparable accuracy on most healthcare time-series benchmarks. They are preferred on wearable devices where battery and compute are constrained. At 500 Hz ECG sampling, a GRU processes 30-second windows in &lt;200ms latency — sufficient for real-time atrial fibrillation detection with &gt;97% sensitivity.
        </div>
      </div>
    </div>

    <div class="card mb-20">
      <div class="card-title">Transformer — Clinical NLP &amp; Longitudinal EHR Analysis</div>
      <div class="grid-2">
        <div class="arch-diagram">${arch(layers_transformer)}</div>
        <div class="explain-box" style="align-self:flex-start;">
          <strong>Why Transformers dominate modern clinical AI:</strong><br/><br/>
          Unlike RNNs, transformers process all time steps <em>in parallel</em> via self-attention, enabling training on year-long patient histories in minutes. Key models:<br/><br/>
          • <strong>ClinicalBERT / BioBERT</strong>: Pre-trained on MIMIC-III clinical notes; fine-tuned for ICD-10 coding (F1 &gt; 0.90), named entity recognition, and 30-day readmission prediction.<br/><br/>
          • <strong>Temporal Fusion Transformer (TFT)</strong>: SOTA on multi-horizon vital-sign forecasting; powers sepsis early warning with 6h lead time.<br/><br/>
          • <strong>Med-PaLM 2</strong>: Achieves "expert" level on USMLE; assists in differential diagnosis and drug interaction queries.<br/><br/>
          • <strong>Perceiver IO</strong>: Single architecture handles ECG + imaging + EHR text simultaneously for holistic risk scoring.
        </div>
      </div>
    </div>

    <div class="grid-3">
      <div class="card text-center">
        <div style="font-size:2.5rem;margin-bottom:8px;">📈</div>
        <div class="card-title" style="justify-content:center;">Time-Series Vitals</div>
        <p style="font-size:0.85rem;color:var(--text-muted);line-height:1.7;">Heart rate, SpO₂, BP monitored continuously. LSTMs/Transformers learn temporal deterioration patterns to predict sepsis, cardiac arrest, or respiratory failure hours ahead.</p>
      </div>
      <div class="card text-center">
        <div style="font-size:2.5rem;margin-bottom:8px;">⚡</div>
        <div class="card-title" style="justify-content:center;">ECG Signal Analysis</div>
        <p style="font-size:0.85rem;color:var(--text-muted);line-height:1.7;">12-lead ECG at 500 Hz = 5,000 samples/sec. 1D-CNNs + GRUs detect arrhythmias, STEMI, QT prolongation with cardiologist-level accuracy (&gt;97% AUC).</p>
      </div>
      <div class="card text-center">
        <div style="font-size:2.5rem;margin-bottom:8px;">🏥</div>
        <div class="card-title" style="justify-content:center;">EHR Integration</div>
        <p style="font-size:0.85rem;color:var(--text-muted);line-height:1.7;">Transformers process unstructured clinical notes + structured labs jointly. Downstream tasks: ICD coding, mortality prediction, drug dosing, and adverse event detection.</p>
      </div>
    </div>`;
}
