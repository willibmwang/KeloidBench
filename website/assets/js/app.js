"use strict";

const COLORS = {
  ink: "#102635",
  teal: "#3d806e",
  tealLight: "#8bd4bd",
  coral: "#bf684f",
  coralLight: "#eda78f",
  gold: "#d7a34b",
  grid: "rgba(16,38,53,.16)",
};

const state = { data: null, profile: null, confidenceThreshold: 0.55, fold: null };
const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function titleCase(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function fmt(value, digits = 3) { return Number(value).toFixed(digits); }
function pct(value, digits = 1) { return `${(Number(value) * 100).toFixed(digits)}%`; }

function svg(tag, attributes = {}, text = "") {
  const attrs = Object.entries(attributes).map(([key, value]) => `${key}="${escapeHtml(value)}"`).join(" ");
  return `<${tag} ${attrs}>${text}</${tag}>`;
}

function setTooltip(event, html) {
  const tooltip = $("#tooltip");
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function moveTooltip(event) {
  const tooltip = $("#tooltip");
  const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 12);
  const y = Math.min(event.clientY + 14, window.innerHeight - tooltip.offsetHeight - 12);
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip() { $("#tooltip").style.display = "none"; }

function populateHeadline() {
  const h = state.data.headline;
  $("#metric-studies").textContent = h.transferable_studies;
  $("#metric-f1").textContent = fmt(h.loso_macro_f1);
  $("#metric-accuracy").textContent = h.pooled_accuracy_count;
  $("#metric-selective").textContent = fmt(h.selective_macro_f1);
  $("#metric-coverage").textContent = pct(h.selective_coverage);
}

function buildStories() {
  const order = ["confident_keloid", "confident_unaffected", "abstention", "transfer_failure"];
  $("#story-grid").innerHTML = order.map((slug) => {
    const item = state.data.curated_examples[slug];
    return `<button class="story-card" type="button" data-slug="${slug}">
      <span class="story-icon">${escapeHtml(item.icon)}</span><span class="story-arrow">↗</span>
      <h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.description)}</p>
    </button>`;
  }).join("");
  document.querySelectorAll(".story-card").forEach((button) => {
    button.addEventListener("click", () => {
      const item = state.data.curated_examples[button.dataset.slug];
      selectProfile(item.sample_id, true);
    });
  });
}

function populateProfiles() {
  const profiles = [...state.data.profiles].sort((a, b) => {
    if (Boolean(a.curated_slug) !== Boolean(b.curated_slug)) return a.curated_slug ? -1 : 1;
    return `${a.accession}-${a.sample_id}`.localeCompare(`${b.accession}-${b.sample_id}`);
  });
  $("#profile-select").innerHTML = profiles.map((profile) => {
    const star = profile.curated_slug ? "★ " : "";
    return `<option value="${escapeHtml(profile.sample_id)}">${star}${escapeHtml(profile.sample_id)} · ${escapeHtml(profile.accession)}</option>`;
  }).join("");
  $("#profile-select").addEventListener("change", (event) => selectProfile(event.target.value, false));
}

function selectProfile(sampleId, scrollToDemo = false) {
  const profile = state.data.profiles.find((item) => item.sample_id === sampleId);
  if (!profile) return;
  state.profile = profile;
  $("#profile-select").value = sampleId;
  document.querySelectorAll(".story-card").forEach((card) => card.classList.toggle("active", card.dataset.slug === profile.curated_slug));
  updateDecision();
  renderPrograms(profile);
  renderPca(profile);
  renderContributions(profile);
  if (scrollToDemo) document.querySelector(".profile-toolbar").scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateDecision() {
  const profile = state.profile;
  if (!profile) return;
  const represented = profile.accession !== "GSE173900";
  const decision = profile.confidence >= state.confidenceThreshold ? profile.pred_label : "abstain";
  $("#profile-name").textContent = profile.sample_id;
  $("#profile-meta").textContent = `${profile.accession} · ${titleCase(profile.modality)} · ${profile.platform_id}`;

  const transfer = $("#transfer-badge");
  transfer.className = `status-badge ${represented ? "" : "warning"}`;
  transfer.textContent = represented ? "represented transfer subset" : "non-transferable platform";

  $("#probability-value").textContent = pct(profile.prob_keloid);
  $("#probability-gauge").style.setProperty("--angle", `${profile.prob_keloid * 360}deg`);
  $("#confidence-value").textContent = fmt(profile.confidence);
  $("#reference-value").textContent = titleCase(profile.reference_label);
  $("#class-threshold-value").textContent = fmt(profile.decision_threshold);

  const label = $("#decision-label");
  label.className = `decision-label ${decision === "keloid" ? "keloid" : decision === "abstain" ? "abstain" : ""}`;
  label.textContent = decision === "abstain" ? "ABSTAIN · CLASS WITHHELD" : `${titleCase(decision)}-LIKE · RESEARCH TRIAGE`;

  const explanation = $("#decision-explanation");
  if (decision === "abstain") {
    explanation.textContent = `Confidence ${fmt(profile.confidence)} does not meet the selected ${fmt(state.confidenceThreshold, 2)} policy. The probability remains visible, but no class is returned.`;
  } else if (!represented) {
    explanation.textContent = "The model returns a class, but the platform transfer warning should take precedence over the prediction.";
  } else {
    explanation.textContent = `Confidence ${fmt(profile.confidence)} meets the selected ${fmt(state.confidenceThreshold, 2)} policy.`;
  }

  const source = state.data.sources[profile.accession];
  const paperLink = $("#paper-link");
  if (source) {
    paperLink.href = source.paper_url;
    paperLink.classList.remove("hidden");
  } else {
    paperLink.classList.add("hidden");
  }
}

function renderPrograms(profile) {
  const rows = [...profile.programs].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const width = 880;
  const labelWidth = 225;
  const plotStart = 250;
  const plotEnd = 860;
  const center = (plotStart + plotEnd) / 2;
  const rowHeight = 31;
  const top = 34;
  const height = top + rows.length * rowHeight + 22;
  const maxValue = Math.max(.35, ...rows.flatMap((row) => [Math.abs(row.score), Math.abs(row.keloid_median), Math.abs(row.unaffected_median)]));
  const scale = (value) => center + value / maxValue * ((plotEnd - plotStart) / 2);
  let content = svg("line", { x1: center, x2: center, y1: 18, y2: height - 13, class: "chart-axis" });
  content += svg("text", { x: plotStart, y: 12, class: "chart-small" }, "lower activity");
  content += svg("text", { x: plotEnd, y: 12, "text-anchor": "end", class: "chart-small" }, "higher activity");

  rows.forEach((row, index) => {
    const y = top + index * rowHeight;
    content += svg("line", { x1: 0, x2: width, y1: y + 16, y2: y + 16, stroke: "rgba(16,38,53,.08)" });
    content += svg("text", { x: 0, y: y + 4, class: "chart-label" }, escapeHtml(row.display_name));
    const series = [
      { value: row.keloid_median, color: COLORS.coral, y: y - 7, height: 3 },
      { value: row.unaffected_median, color: COLORS.teal, y: y - 1, height: 3 },
      { value: row.score, color: COLORS.ink, y: y + 6, height: 7 },
    ];
    series.forEach((item) => {
      const x = scale(item.value);
      content += svg("rect", { x: Math.min(center, x), y: item.y, width: Math.max(1.5, Math.abs(x - center)), height: item.height, rx: 1.5, fill: item.color });
    });
    content += svg("text", { x: labelWidth, y: y + 4, "text-anchor": "end", class: "chart-small" }, `${row.score >= 0 ? "+" : ""}${fmt(row.score, 2)}`);
  });
  $("#program-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Program scores for ${escapeHtml(profile.sample_id)}">${content}</svg>`;
}

function renderPca(profile) {
  const points = state.data.pca;
  const width = 880, height = 330, pad = 28;
  const xs = points.map((point) => point.pc1), ys = points.map((point) => point.pc2);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value) => pad + (value - minX) / (maxX - minX || 1) * (width - pad * 2);
  const y = (value) => height - pad - (value - minY) / (maxY - minY || 1) * (height - pad * 2);
  let content = svg("line", { x1: pad, x2: width - pad, y1: height - pad, y2: height - pad, class: "chart-axis" });
  content += svg("line", { x1: pad, x2: pad, y1: pad, y2: height - pad, class: "chart-axis" });
  content += svg("text", { x: width - pad, y: height - 7, "text-anchor": "end", class: "chart-small" }, "program-space PC1");
  content += svg("text", { x: pad + 4, y: 17, class: "chart-small" }, "PC2");
  points.forEach((point) => {
    const selected = point.sample_id === profile.sample_id;
    const fill = point.reference_label === "keloid" ? COLORS.coral : COLORS.teal;
    const shape = point.transfer_group === "non-transferable platform" ? "rect" : "circle";
    const attrs = shape === "circle"
      ? { cx: x(point.pc1), cy: y(point.pc2), r: selected ? 7 : 4, fill, opacity: selected ? 1 : .58 }
      : { x: x(point.pc1) - (selected ? 7 : 4), y: y(point.pc2) - (selected ? 7 : 4), width: selected ? 14 : 8, height: selected ? 14 : 8, fill: COLORS.gold, opacity: selected ? 1 : .65 };
    const label = `${point.sample_id}<br>${point.accession} · ${titleCase(point.reference_label)}<br>${point.transfer_group}`;
    content += `<${shape} ${Object.entries({ ...attrs, class: "pca-point", "data-sample": point.sample_id, "data-tooltip": label }).map(([key, value]) => `${key}="${escapeHtml(value)}"`).join(" ")}></${shape}>`;
    if (selected) content += svg("circle", { cx: x(point.pc1), cy: y(point.pc2), r: 11, fill: "none", stroke: COLORS.ink, "stroke-width": 2 });
  });
  $("#pca-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PCA map of public program profiles">${content}</svg>`;
  document.querySelectorAll(".pca-point").forEach((point) => {
    point.addEventListener("mouseenter", (event) => setTooltip(event, point.dataset.tooltip));
    point.addEventListener("mousemove", moveTooltip);
    point.addEventListener("mouseleave", hideTooltip);
    point.addEventListener("click", () => selectProfile(point.dataset.sample, false));
  });
}

function renderContributions(profile) {
  const contributions = profile.contributions;
  const max = Math.max(.01, ...contributions.map((item) => Math.abs(item.contribution)));
  $("#contribution-chart").innerHTML = contributions.map((item) => {
    const value = Number(item.contribution);
    const width = Math.abs(value) / max * 49;
    const direction = value >= 0 ? "positive" : "negative";
    return `<div class="contribution-row"><span class="contribution-name">${escapeHtml(item.display_name)}</span>
      <div class="contribution-track"><span class="contribution-bar ${direction}" style="width:${width}%"></span></div>
      <span class="contribution-value">${value >= 0 ? "+" : ""}${fmt(value)}</span></div>`;
  }).join("");
}

function buildEvidence() {
  const accessions = Object.keys(state.data.folds).sort((a, b) => {
    if (a === "GSE173900") return 1;
    if (b === "GSE173900") return -1;
    return a.localeCompare(b);
  });
  $("#fold-select").innerHTML = accessions.map((accession) => `<option value="${accession}">${accession}${accession === "GSE173900" ? " · known transfer failure" : ""}</option>`).join("");
  $("#fold-select").addEventListener("change", (event) => selectFold(event.target.value));
  selectFold(accessions.includes("GSE181316") ? "GSE181316" : accessions[0]);
}

function selectFold(accession) {
  const fold = state.data.folds[accession];
  state.fold = fold;
  $("#fold-select").value = accession;
  const transfer = $("#fold-transfer");
  const warning = fold.transfer_group === "non-transferable platform";
  transfer.className = `status-badge ${warning ? "warning" : ""}`;
  transfer.textContent = fold.transfer_group;
  $("#fold-f1").textContent = fmt(fold.macro_f1);
  $("#fold-accuracy").textContent = fmt(fold.accuracy);
  $("#fold-coverage").textContent = pct(fold.coverage_055, 0);
  $("#fold-count").textContent = fold.predictions.length;
  renderFoldChart(fold);
  renderPredictionTable(fold);
}

function renderFoldChart(fold) {
  const predictions = fold.predictions;
  const width = 1100, height = 390, left = 54, right = 20, top = 23, bottom = 42;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const x = (index) => left + (index + .5) / predictions.length * plotWidth;
  const y = (value) => top + (1 - value) * plotHeight;
  let content = svg("rect", { x: left, y: y(.55), width: plotWidth, height: y(.45) - y(.55), fill: COLORS.gold, opacity: .17 });
  for (let tick = 0; tick <= 1.001; tick += .25) {
    content += svg("line", { x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: "chart-axis" });
    content += svg("text", { x: left - 10, y: y(tick) + 3, "text-anchor": "end", class: "chart-label" }, tick.toFixed(2));
  }
  content += svg("line", { x1: left, x2: width - right, y1: y(fold.threshold), y2: y(fold.threshold), stroke: "rgba(255,255,255,.8)", "stroke-dasharray": "6 5" });
  content += svg("text", { x: width - right, y: y(fold.threshold) - 7, "text-anchor": "end", class: "chart-small" }, `class threshold ${fmt(fold.threshold)}`);
  predictions.forEach((item, index) => {
    const fill = item.true_label === "keloid" ? COLORS.coralLight : COLORS.tealLight;
    const stroke = item.selective_decision === "abstain" ? COLORS.gold : "none";
    const tooltip = `${item.sample_id}<br>Reference: ${titleCase(item.true_label)}<br>P(keloid): ${fmt(item.prob_keloid)}<br>Decision: ${titleCase(item.selective_decision)}`;
    content += svg("circle", { cx: x(index), cy: y(item.prob_keloid), r: 6, fill, stroke, "stroke-width": 3, class: "fold-point", "data-tooltip": tooltip });
  });
  content += svg("text", { x: left + plotWidth / 2, y: height - 7, "text-anchor": "middle", class: "chart-label" }, "held-out profiles sorted by P(keloid)");
  content += svg("text", { x: 13, y: top + plotHeight / 2, transform: `rotate(-90 13 ${top + plotHeight / 2})`, "text-anchor": "middle", class: "chart-label" }, "P(keloid)");
  $("#fold-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Held-out probabilities for ${escapeHtml(fold.accession)}">${content}</svg>`;
  document.querySelectorAll(".fold-point").forEach((point) => {
    point.addEventListener("mouseenter", (event) => setTooltip(event, point.dataset.tooltip));
    point.addEventListener("mousemove", moveTooltip);
    point.addEventListener("mouseleave", hideTooltip);
  });
}

function renderPredictionTable(fold) {
  $("#prediction-table tbody").innerHTML = fold.predictions.map((item) => `<tr>
    <td>${escapeHtml(item.sample_id)}</td><td>${escapeHtml(titleCase(item.true_label))}</td>
    <td>${fmt(item.prob_keloid)}</td><td>${escapeHtml(titleCase(item.pred_label))}</td>
    <td>${fmt(item.confidence)}</td><td>${escapeHtml(titleCase(item.selective_decision))}</td>
  </tr>`).join("");
}

function buildSources() {
  const sources = Object.entries(state.data.sources).sort(([a], [b]) => a.localeCompare(b));
  $("#source-grid").innerHTML = sources.map(([accession, source]) => `<article class="source-card">
    <span class="accession">${escapeHtml(accession)}</span><h4>${escapeHtml(source.study_title)}</h4>
    <p>${escapeHtml(source.study_design)}</p><p>${escapeHtml(source.paper_citation)}</p>
    <div class="source-links"><a href="${escapeHtml(source.paper_url)}" target="_blank" rel="noreferrer">Original paper ↗</a><a href="${escapeHtml(source.geo_url)}" target="_blank" rel="noreferrer">GEO record ↗</a></div>
  </article>`).join("");
}

function wireControls() {
  const threshold = $("#confidence-threshold");
  threshold.addEventListener("input", (event) => {
    state.confidenceThreshold = Number(event.target.value);
    $("#threshold-output").textContent = state.confidenceThreshold.toFixed(2);
    updateDecision();
  });
}

async function init() {
  try {
    const response = await fetch("assets/data/demo-data.json");
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    state.data = await response.json();
    populateHeadline();
    buildStories();
    populateProfiles();
    buildEvidence();
    buildSources();
    wireControls();
    const first = state.data.curated_examples.confident_keloid.sample_id;
    selectProfile(first, false);
  } catch (error) {
    console.error(error);
    $("#story-grid").innerHTML = `<div class="error-message">The versioned demo data could not be loaded. Serve this directory over HTTP or refresh the GitHub Pages deployment.</div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
