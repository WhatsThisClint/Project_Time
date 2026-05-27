const form = document.querySelector("#runForm");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const statusEl = document.querySelector("#status");
const previewEl = document.querySelector("#preview");
const forecastEl = document.querySelector("#forecastTable");
const metricsEl = document.querySelector("#metrics");
const chartEl = document.querySelector("#chart");
const dateColumn = document.querySelector("#dateColumn");
const valueColumn = document.querySelector("#valueColumn");
const idColumn = document.querySelector("#idColumn");
const frequencyInput = document.querySelector("#frequency");
const downloadModel = document.querySelector("#downloadModel");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileName.textContent = file.name;
  statusEl.textContent = "Reading file...";
  const body = new FormData();
  body.append("file", file);
  try {
    const data = await postForm("/api/preview", body);
    fillColumns(data.columns, data.inferred);
    renderTable(previewEl, data.preview);
    statusEl.textContent = `${data.rows} rows loaded`;
  } catch (error) {
    statusEl.textContent = error.message;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = "Choose a file first";
    return;
  }
  const body = new FormData(form);
  body.set("file", file);
  statusEl.textContent = "Running pipeline...";
  try {
    const data = await postForm("/api/run", body);
    renderMetrics(data.report, data.metrics);
    renderTable(previewEl, data.cleaned_preview);
    renderTable(forecastEl, data.forecasts);
    renderChart(data.cleaned_preview, data.forecasts);
    statusEl.textContent = `Done: ${data.report.series_count} series, ${data.forecasts.length} forecast rows`;
    if (data.report.frequency) frequencyInput.placeholder = data.report.frequency;
  } catch (error) {
    statusEl.textContent = error.message;
  }
});

downloadModel.addEventListener("click", async () => {
  statusEl.textContent = "Caching TimesFM model...";
  try {
    const body = new FormData();
    const data = await postForm("/api/download-timesfm", body);
    statusEl.textContent = `Cached ${data.model}`;
  } catch (error) {
    statusEl.textContent = error.message;
  }
});

async function postForm(url, body) {
  const response = await fetch(url, { method: "POST", body });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function fillColumns(columns, inferred) {
  fillSelect(dateColumn, columns, inferred.date_column);
  fillSelect(valueColumn, columns, inferred.value_column);
  fillSelect(idColumn, ["", ...columns], inferred.id_column || "");
}

function fillSelect(select, columns, selected) {
  select.innerHTML = "";
  columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column || "None";
    option.selected = column === selected;
    select.appendChild(option);
  });
}

function renderMetrics(report, metrics) {
  const firstMetric = metrics && metrics.length ? metrics[0] : {};
  const cards = [
    ["Rows", report.output_rows ?? 0],
    ["Series", report.series_count ?? 0],
    ["Missing", report.missing_after_fill ?? 0],
    ["RMSE", firstMetric.rmse != null ? round(firstMetric.rmse) : "n/a"],
  ];
  metricsEl.innerHTML = cards
    .map(([label, value]) => `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
}

function renderTable(target, rows) {
  if (!rows || rows.length === 0) {
    target.innerHTML = '<div class="empty-state">No rows yet</div>';
    return;
  }
  const columns = Object.keys(rows[0]);
  const header = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows
    .slice(0, 200)
    .map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(format(row[column]))}</td>`).join("")}</tr>`)
    .join("");
  target.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderChart(history, forecast) {
  const seriesId = forecast?.[0]?.series_id || history?.[0]?.series_id;
  const hist = (history || []).filter((row) => !seriesId || row.series_id === seriesId).slice(-80);
  const fc = (forecast || []).filter((row) => !seriesId || row.series_id === seriesId);
  const values = [
    ...hist.map((row) => Number(row.value)).filter(Number.isFinite),
    ...fc.map((row) => Number(row.forecast)).filter(Number.isFinite),
  ];
  if (!values.length) {
    chartEl.innerHTML = "";
    return;
  }

  const width = 900;
  const height = 360;
  const pad = 42;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = [...hist.map((row) => Number(row.value)), ...fc.map((row) => Number(row.forecast))];
  const x = (i) => pad + (i / Math.max(points.length - 1, 1)) * (width - pad * 2);
  const y = (value) => height - pad - ((value - min) / span) * (height - pad * 2);
  const histPath = pathFor(hist.map((row) => Number(row.value)), 0, x, y);
  const forecastPath = pathFor(fc.map((row) => Number(row.forecast)), hist.length, x, y);
  const splitX = x(Math.max(hist.length - 1, 0));

  chartEl.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fff"></rect>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#ccd6df"></line>
    <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#ccd6df"></line>
    <line x1="${splitX}" y1="${pad}" x2="${splitX}" y2="${height - pad}" stroke="#9badbd" stroke-dasharray="6 6"></line>
    <path d="${histPath}" fill="none" stroke="#172026" stroke-width="3"></path>
    <path d="${forecastPath}" fill="none" stroke="#0b6bcb" stroke-width="3"></path>
    <text x="${pad}" y="24" fill="#64717d" font-size="13">${escapeHtml(seriesId || "series")}</text>
    <text x="${width - pad}" y="24" text-anchor="end" fill="#64717d" font-size="13">history / forecast</text>
  `;
}

function pathFor(values, offset, x, y) {
  return values
    .map((value, i) => `${i === 0 ? "M" : "L"} ${x(i + offset).toFixed(2)} ${y(value).toFixed(2)}`)
    .join(" ");
}

function round(value) {
  return Number(value).toFixed(3);
}

function format(value) {
  if (value == null) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
