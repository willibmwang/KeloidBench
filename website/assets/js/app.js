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

const state = { data: null, profile: null, confidenceThreshold: 0.55, fold: null, uploadFile: null };
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
    if (Boolean(a.uploaded) !== Boolean(b.uploaded)) return a.uploaded ? -1 : 1;
    if (Boolean(a.curated_slug) !== Boolean(b.curated_slug)) return a.curated_slug ? -1 : 1;
    return `${a.accession}-${a.sample_id}`.localeCompare(`${b.accession}-${b.sample_id}`);
  });
  $("#profile-select").innerHTML = profiles.map((profile) => {
    const marker = profile.uploaded ? "⬆ " : profile.curated_slug ? "★ " : "";
    const name = profile.display_name || profile.sample_id;
    return `<option value="${escapeHtml(profile.sample_id)}">${marker}${escapeHtml(name)} · ${escapeHtml(profile.accession)}</option>`;
  }).join("");
  $("#profile-select").onchange = (event) => selectProfile(event.target.value, false);
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
  document.querySelectorAll("#upload-results-body tr").forEach((row) => row.classList.toggle("selected", row.dataset.sample === sampleId));
  if (scrollToDemo) document.querySelector(".profile-toolbar").scrollIntoView({ behavior: "smooth", block: "start" });
}

function decisionForProfile(profile) {
  if (profile.quality_forced_abstain) return "abstain";
  return profile.confidence >= state.confidenceThreshold ? profile.pred_label : "abstain";
}

function updateDecision() {
  const profile = state.profile;
  if (!profile) return;
  const transferStatus = profile.transfer_status || (profile.accession === "GSE173900" ? "non-transferable" : "represented");
  const represented = transferStatus === "represented";
  const decision = decisionForProfile(profile);
  $("#profile-name").textContent = profile.display_name || profile.sample_id;
  $("#profile-meta").textContent = `${profile.accession} · ${titleCase(profile.modality)} · ${profile.platform_id}`;

  const transfer = $("#transfer-badge");
  transfer.className = `status-badge ${represented ? "" : "warning"}`;
  transfer.textContent = represented ? "represented transfer subset" : transferStatus === "non-transferable" ? "non-transferable platform" : "uploaded · transfer unverified";

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
    explanation.textContent = profile.quality_forced_abstain
      ? `Only ${profile.live_programs_usable}/${profile.live_programs_expected} required model programs passed gene-coverage QC. The score remains visible, but the class is withheld.`
      : `Confidence ${fmt(profile.confidence)} does not meet the selected ${fmt(state.confidenceThreshold, 2)} policy. The probability remains visible, but no class is returned.`;
  } else if (transferStatus === "non-transferable") {
    explanation.textContent = "The model returns a class, but the platform transfer warning should take precedence over the prediction.";
  } else if (!represented) {
    explanation.textContent = `This uploaded cohort was not part of transfer evaluation. Confidence ${fmt(profile.confidence)} describes the model score, not external validation.`;
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
  $("#program-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Program scores for ${escapeHtml(profile.display_name || profile.sample_id)}">${content}</svg>`;
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
    const fill = point.reference_label === "keloid" ? COLORS.coral : point.reference_label === "non_keloid" ? COLORS.teal : COLORS.gold;
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

function normalizeGeneSymbol(value) {
  const gene = String(value || "").trim().replace(/[^A-Za-z0-9_.-]/g, "").toUpperCase();
  if (!gene || ["NA", "NAN", "NULL", "GENE", "SYMBOL", "GENE_SYMBOL"].includes(gene)) return null;
  return gene;
}

function detectDelimiter(text) {
  const line = text.replace(/^\uFEFF/, "").split(/\r?\n/).find((item) => item.trim() && !item.trim().startsWith("!")) || "";
  const tabs = (line.match(/\t/g) || []).length;
  const commas = (line.match(/,/g) || []).length;
  return tabs > commas ? "\t" : ",";
}

function parseDelimited(text, delimiter) {
  const rows = [];
  let row = [], field = "", quoted = false;
  const input = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (character === '"') {
      if (quoted && input[index + 1] === '"') { field += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === delimiter && !quoted) {
      row.push(field); field = "";
    } else if (character === "\n" && !quoted) {
      row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = "";
    } else field += character;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows.filter((values) => values.some((value) => String(value).trim()) && !String(values[0] || "").trim().startsWith("!") && !String(values[0] || "").trim().startsWith("#"));
}

function uniqueNames(values) {
  const counts = new Map();
  return values.map((value, index) => {
    const base = String(value || "").trim() || `sample_${index + 1}`;
    const count = (counts.get(base) || 0) + 1;
    counts.set(base, count);
    return count === 1 ? base : `${base}_${count}`;
  });
}

function numericValue(value) {
  const number = Number(String(value ?? "").trim());
  return Number.isFinite(number) ? number : 0;
}

function finalizeGeneAccumulator(accumulator) {
  const genes = {};
  accumulator.forEach((value, gene) => { genes[gene] = value.sum / value.count; });
  return genes;
}

function parseExpressionMatrix(text, requestedOrientation = "auto") {
  const delimiter = detectDelimiter(text);
  const rows = parseDelimited(text, delimiter);
  if (rows.length < 2 || rows[0].length < 2) throw new Error("The file needs a header row and at least one data row.");
  const headers = rows[0].map((value) => String(value).trim());
  const programGenes = new Set(Object.values(state.data.browser_scorer.program_sets).flatMap((program) => [...program.positive, ...program.negative]));
  const firstHeader = String(headers[0] || "").trim().toLowerCase().replace(/[^a-z_]/g, "");
  const headerMatches = headers.slice(1).map(normalizeGeneSymbol).filter((gene) => gene && programGenes.has(gene)).length;
  const firstColumnMatches = rows.slice(1, 501).map((row) => normalizeGeneSymbol(row[0])).filter((gene) => gene && programGenes.has(gene)).length;
  let orientation = requestedOrientation;
  if (orientation === "auto") {
    if (["gene", "genes", "symbol", "gene_symbol", "hgnc", "id_ref"].includes(firstHeader)) orientation = "genes_rows";
    else if (["sample", "sample_id", "samples", "profile", "profile_id"].includes(firstHeader)) orientation = "samples_rows";
    else orientation = firstColumnMatches > headerMatches ? "genes_rows" : "samples_rows";
  }
  const samples = [];
  if (orientation === "samples_rows") {
    const geneColumns = headers.slice(1).map(normalizeGeneSymbol);
    const sampleNames = uniqueNames(rows.slice(1).map((row) => row[0]));
    rows.slice(1).forEach((row, sampleIndex) => {
      const accumulator = new Map();
      geneColumns.forEach((gene, geneIndex) => {
        if (!gene) return;
        const current = accumulator.get(gene) || { sum: 0, count: 0 };
        current.sum += numericValue(row[geneIndex + 1]); current.count += 1; accumulator.set(gene, current);
      });
      samples.push({ name: sampleNames[sampleIndex], genes: finalizeGeneAccumulator(accumulator) });
    });
  } else {
    const sampleNames = uniqueNames(headers.slice(1));
    const accumulators = sampleNames.map(() => new Map());
    rows.slice(1).forEach((row) => {
      const gene = normalizeGeneSymbol(row[0]);
      if (!gene) return;
      accumulators.forEach((accumulator, sampleIndex) => {
        const current = accumulator.get(gene) || { sum: 0, count: 0 };
        current.sum += numericValue(row[sampleIndex + 1]); current.count += 1; accumulator.set(gene, current);
      });
    });
    sampleNames.forEach((name, index) => samples.push({ name, genes: finalizeGeneAccumulator(accumulators[index]) }));
  }
  if (!samples.length) throw new Error("No samples could be read from this matrix.");
  if (samples.length > 200) throw new Error("This browser demo accepts up to 200 samples at once. Split larger cohorts into smaller files.");
  return { samples, orientation, delimiter: delimiter === "\t" ? "TSV" : "CSV" };
}

function percentileRanks(geneValues) {
  const entries = Object.entries(geneValues).filter(([, value]) => Number.isFinite(value)).sort((a, b) => a[1] - b[1]);
  const ranks = {};
  const count = entries.length;
  let index = 0;
  while (index < count) {
    let end = index + 1;
    while (end < count && entries[end][1] === entries[index][1]) end += 1;
    const percentile = ((index + 1 + end) / 2) / count;
    for (let cursor = index; cursor < end; cursor += 1) ranks[entries[cursor][0]] = percentile;
    index = end;
  }
  return ranks;
}

function meanRank(ranks, genes) {
  const values = genes.filter((gene) => Object.hasOwn(ranks, gene)).map((gene) => ranks[gene]);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function referenceMedian(programName, referenceLabel) {
  const values = state.data.profiles.filter((profile) => !profile.uploaded && profile.reference_label === referenceLabel).map((profile) => profile.programs.find((program) => program.program === programName)?.score).filter(Number.isFinite);
  return median(values);
}

function scoreUploadedSample(sample, accession, batchId, layout) {
  const scorer = state.data.browser_scorer;
  const ranks = percentileRanks(sample.genes);
  if (!Object.keys(ranks).length) throw new Error(`Sample ${sample.name} has no numeric gene values.`);
  const scoreByProgram = {}, coverageByProgram = {};
  Object.entries(scorer.program_sets).forEach(([name, program]) => {
    const expected = [...new Set([...program.positive, ...program.negative])];
    const present = expected.filter((gene) => Object.hasOwn(ranks, gene));
    const usable = present.length >= program.min_genes_present;
    const positive = meanRank(ranks, program.positive);
    const negative = meanRank(ranks, program.negative);
    let score = 0;
    if (usable) {
      if (positive !== null && negative !== null) score = positive - negative;
      else if (positive !== null) score = positive;
      else if (negative !== null) score = -negative;
    }
    scoreByProgram[name] = score;
    coverageByProgram[name] = { present: present.length, expected: expected.length, usable };
  });
  const features = scorer.feature_columns.map((name) => scoreByProgram[name] ?? 0);
  const standardized = features.map((value, index) => (value - scorer.scaler_mean[index]) / (scorer.scaler_scale[index] || 1));
  const margin = scorer.intercept + standardized.reduce((sum, value, index) => sum + value * scorer.coefficients[index], 0);
  const probKeloid = 1 / (1 + Math.exp(-(scorer.positive_direction * margin)));
  const confidence = Math.max(probKeloid, 1 - probKeloid);
  const predLabel = probKeloid >= scorer.class_threshold ? "keloid" : "non_keloid";
  const liveUsable = scorer.feature_columns.filter((name) => coverageByProgram[name].usable).length;
  const expectedGenes = new Set(Object.values(scorer.program_sets).flatMap((program) => [...program.positive, ...program.negative]));
  const genesPresent = [...expectedGenes].filter((gene) => Object.hasOwn(sample.genes, gene)).length;
  const programs = Object.keys(scorer.program_sets).map((name) => ({
    program: name,
    display_name: state.data.program_labels[name] || titleCase(name),
    score: scoreByProgram[name],
    direction: scoreByProgram[name] > 0 ? "positive" : scoreByProgram[name] < 0 ? "negative" : "neutral",
    keloid_median: referenceMedian(name, "keloid"),
    unaffected_median: referenceMedian(name, "non_keloid"),
  }));
  const contributions = scorer.feature_columns.map((feature, index) => ({
    feature,
    display_name: state.data.program_labels[feature] || titleCase(feature),
    value: features[index],
    contribution: scorer.positive_direction * standardized[index] * scorer.coefficients[index],
  })).sort((a, b) => b.contribution - a.contribution);
  const pca = scorer.pca;
  const pcaFeatures = pca.feature_columns.map((name) => scoreByProgram[name] ?? 0);
  const pcaScaled = pcaFeatures.map((value, index) => (value - pca.scaler_mean[index]) / (pca.scaler_scale[index] || 1) - pca.center[index]);
  const coordinate = pca.components.map((component) => component.reduce((sum, weight, index) => sum + weight * pcaScaled[index], 0));
  const sampleId = `uploaded::${batchId}::${sample.name}`;
  const profile = {
    sample_id: sampleId, display_name: sample.name, accession, reference_label: "unknown",
    modality: "uploaded expression", platform_id: layout, uploaded: true, transfer_status: "unverified",
    prob_keloid: probKeloid, confidence, pred_label: predLabel,
    decision_threshold: scorer.class_threshold, confidence_threshold: state.confidenceThreshold,
    quality_forced_abstain: liveUsable < scorer.feature_columns.length,
    live_programs_usable: liveUsable, live_programs_expected: scorer.feature_columns.length,
    program_genes_present: genesPresent, program_genes_expected: expectedGenes.size,
    programs, contributions,
  };
  const point = { sample_id: sampleId, pc1: coordinate[0], pc2: coordinate[1], reference_label: "unknown", accession, transfer_group: "uploaded · transfer unverified", uploaded: true };
  return { profile, point };
}

function setUploadStatus(message, kind = "") {
  const status = $("#upload-status");
  status.className = `upload-status ${kind}`.trim();
  status.textContent = message;
}

function renderUploadResults() {
  const profiles = state.data.profiles.filter((profile) => profile.uploaded);
  const results = $("#upload-results");
  if (!profiles.length) { results.classList.add("hidden"); return; }
  results.classList.remove("hidden");
  $("#upload-results-title").textContent = `${profiles.length} sample${profiles.length === 1 ? "" : "s"} analyzed`;
  $("#upload-results-body").innerHTML = profiles.map((profile) => {
    const decision = decisionForProfile(profile);
    const label = decision === "abstain" ? "Not sure" : `${titleCase(decision)}-like`;
    return `<tr data-sample="${escapeHtml(profile.sample_id)}"><td>${escapeHtml(profile.display_name)}</td><td>${pct(profile.prob_keloid)}</td><td>${fmt(profile.confidence)}</td><td><span class="upload-result-label ${decision.replaceAll("_", "-")}">${label}</span></td><td>${profile.program_genes_present}/${profile.program_genes_expected} genes · ${profile.live_programs_usable}/${profile.live_programs_expected} model programs</td></tr>`;
  }).join("");
  document.querySelectorAll("#upload-results-body tr").forEach((row) => row.addEventListener("click", () => selectProfile(row.dataset.sample, true)));
}

function setUploadFile(file) {
  if (!file) return;
  if (file.size > 30 * 1024 * 1024) {
    state.uploadFile = null; $("#analyze-upload").disabled = true;
    setUploadStatus("That file is larger than the 30 MB browser-demo limit.", "error"); return;
  }
  state.uploadFile = file;
  $("#file-name").textContent = file.name;
  $("#analyze-upload").disabled = false;
  setUploadStatus(`${file.name} is ready. Nothing has left your device.`);
}

async function analyzeUpload() {
  if (!state.uploadFile) return;
  const button = $("#analyze-upload");
  button.disabled = true; button.textContent = "Analyzing…";
  setUploadStatus("Reading the matrix and scoring biological programs…");
  await new Promise((resolve) => setTimeout(resolve, 30));
  try {
    const parsed = parseExpressionMatrix(await state.uploadFile.text(), $("#matrix-orientation").value);
    const accession = String($("#upload-accession").value || "UPLOADED").trim().replace(/[^A-Za-z0-9_.-]/g, "").toUpperCase() || "UPLOADED";
    const batchId = Date.now();
    const newProfiles = [], newPoints = [];
    for (let index = 0; index < parsed.samples.length; index += 1) {
      const layout = `${parsed.delimiter} · ${parsed.orientation === "genes_rows" ? "genes in rows" : "samples in rows"}`;
      const scored = scoreUploadedSample(parsed.samples[index], accession, batchId, layout);
      newProfiles.push(scored.profile); newPoints.push(scored.point);
      if (index && index % 20 === 0) await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    state.data.profiles = [...state.data.profiles.filter((profile) => !profile.uploaded), ...newProfiles];
    state.data.pca = [...state.data.pca.filter((point) => !point.uploaded), ...newPoints];
    populateProfiles(); renderUploadResults();
    const uploaded = state.data.profiles.filter((profile) => profile.uploaded);
    const repositoryCount = state.data.profiles.filter((profile) => !profile.uploaded).length;
    $("#pca-count").textContent = `${repositoryCount} repository profiles + ${uploaded.length} uploaded`;
    selectProfile(uploaded[0].sample_id, false);
    const lowCoverage = uploaded.filter((profile) => profile.quality_forced_abstain).length;
    const warning = lowCoverage ? ` ${lowCoverage} sample${lowCoverage === 1 ? " was" : "s were"} withheld because required program coverage was incomplete.` : " All required model programs passed coverage QC.";
    setUploadStatus(`Analyzed ${uploaded.length} sample${uploaded.length === 1 ? "" : "s"} locally. Detected ${parsed.orientation === "genes_rows" ? "genes in rows" : "samples in rows"}.${warning}`, lowCoverage ? "warning" : "");
    $("#upload-results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    console.error(error); setUploadStatus(error.message || "The matrix could not be analyzed.", "error");
  } finally {
    button.disabled = false; button.textContent = "Analyze in browser";
  }
}

function wireUpload() {
  const input = $("#expression-file"), drop = $("#file-drop");
  input.addEventListener("change", () => setUploadFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (event) => setUploadFile(event.dataTransfer.files[0]));
  $("#analyze-upload").addEventListener("click", analyzeUpload);
}

function wireControls() {
  const threshold = $("#confidence-threshold");
  threshold.addEventListener("input", (event) => {
    state.confidenceThreshold = Number(event.target.value);
    $("#threshold-output").textContent = state.confidenceThreshold.toFixed(2);
    updateDecision();
    renderUploadResults();
  });
}

async function init() {
  try {
    const response = await fetch("assets/data/demo-data.json?v=20260802-upload");
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    state.data = await response.json();
    buildStories();
    populateProfiles();
    buildEvidence();
    buildSources();
    wireControls();
    wireUpload();
    const first = state.data.curated_examples.confident_keloid.sample_id;
    selectProfile(first, false);
  } catch (error) {
    console.error(error);
    $("#story-grid").innerHTML = `<div class="error-message">The versioned demo data could not be loaded. Serve this directory over HTTP or refresh the GitHub Pages deployment.</div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
