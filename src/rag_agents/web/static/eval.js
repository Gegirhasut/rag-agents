// Страница «Качество» (/eval): графики прогона на Chart.js (CDN), данные — из #eval-data.
(function () {
  "use strict";
  const dataEl = document.getElementById("eval-data");

  function localTimes() {
    document.querySelectorAll("time[data-fmt]").forEach((el) => {
      const d = new Date(el.getAttribute("datetime"));
      if (!isNaN(d)) el.textContent = d.toLocaleString("ru-RU",
        { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    });
  }

  function init() {
    localTimes();
    if (!dataEl || !window.Chart) return;
    const D = JSON.parse(dataEl.textContent);
    const meta = JSON.parse(document.getElementById("eval-meta").textContent);
    const root = document.querySelector(".evalp");
    const css = getComputedStyle(root);
    const v = (name) => css.getPropertyValue(name).trim();
    const text = getComputedStyle(document.body).getPropertyValue("--bs-secondary-color") || "#6c757d";
    const grid = getComputedStyle(document.body).getPropertyValue("--bs-border-color-translucent") || "rgba(0,0,0,.1)";
    const GROUP = { retrieval: v("--c-retrieval"), answer: v("--c-answer"), citations: v("--c-citations"), refusal: v("--c-refusal") };
    const SERIES = [v("--c-retrieval"), "#7aa2ff", v("--c-answer"), v("--c-citations"), v("--c-refusal")];
    const VERDICT = { "лучше": v("--c-up"), "хуже": v("--c-down"), "шум": v("--c-noise") };
    Chart.defaults.color = text;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    const unit = { min: 0, max: 1, ticks: { stepSize: 0.2 }, grid: { color: grid } };
    const alpha = (hex, a) => hex + Math.round(a * 255).toString(16).padStart(2, "0");

    // 1. Все метрики: горизонтальные бары, цвет — группа метрики
    new Chart(document.getElementById("ch-metrics"), {
      type: "bar",
      data: {
        labels: D.metrics.map((m) => m.label),
        datasets: [{
          data: D.metrics.map((m) => m.mean),
          backgroundColor: D.metrics.map((m) => GROUP[m.group]),
          borderRadius: 3, barPercentage: 0.8,
        }],
      },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${c.raw.toFixed(3)} (n=${D.metrics[c.dataIndex].n})` } } },
        scales: { x: unit, y: { grid: { display: false }, ticks: { autoSkip: false } } },
      },
    });

    // 2. По категориям: сгруппированные бары
    new Chart(document.getElementById("ch-categories"), {
      type: "bar",
      data: {
        labels: D.categories.labels,
        datasets: D.categories.series.map((s, i) => ({
          label: s.label, data: s.data, backgroundColor: SERIES[i % SERIES.length], borderRadius: 3,
        })),
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => c.raw == null ? ` ${c.dataset.label}: —` : ` ${c.dataset.label}: ${c.raw.toFixed(2)}` } } },
        scales: { y: unit, x: { grid: { display: false } } },
      },
    });

    // 3. Порог отказа: лучший dense-score вопроса, «в корпусе» (y=1) и «вне корпуса» (y=0)
    const pts = D.threshold;
    const jitter = (id) => { let h = 0; for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 997; return (h / 997 - 0.5) * 0.5; };
    const row = (p) => (p.expects_refusal ? 0 : 1) + jitter(p.id);
    const tauLine = { id: "tau", afterDatasetsDraw(chart) {
      const x = chart.scales.x.getPixelForValue(tau.value);
      const { top, bottom } = chart.chartArea; const ctx = chart.ctx;
      ctx.save(); ctx.strokeStyle = v("--c-down"); ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      ctx.fillStyle = v("--c-down"); ctx.font = "600 11px sans-serif"; ctx.fillText("τ", x + 4, top + 12); ctx.restore();
    } };
    const tau = document.getElementById("tau");
    const thr = new Chart(document.getElementById("ch-threshold"), {
      type: "scatter",
      data: { datasets: [
        { label: "в корпусе: ответ есть", data: pts.filter((p) => !p.expects_refusal).map((p) => ({ x: p.score, y: row(p), p })),
          backgroundColor: alpha(v("--c-retrieval"), 0.75), pointRadius: 5,
          pointStyle: pts.filter((p) => !p.expects_refusal).map((p) => p.refused ? "crossRot" : "circle"),
          borderColor: v("--c-retrieval") },
        { label: "вне корпуса / инъекция: нужен отказ", data: pts.filter((p) => p.expects_refusal).map((p) => ({ x: p.score, y: row(p), p })),
          backgroundColor: alpha(v("--c-refusal"), 0.75), pointRadius: 5,
          pointStyle: pts.filter((p) => p.expects_refusal).map((p) => p.refused ? "circle" : "crossRot"),
          borderColor: v("--c-refusal") },
      ] },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true } },
          tooltip: { callbacks: { label: (c) => ` ${c.raw.p.id} · ${c.raw.p.category} · score ${c.raw.p.score.toFixed(3)} · ${c.raw.p.refused ? "отказал" : "ответил"}` } } },
        scales: {
          x: { title: { display: true, text: "лучший косинусный score найденного чанка" }, grid: { color: grid } },
          y: { min: -0.5, max: 1.5, grid: { display: false },
            ticks: { stepSize: 1, callback: (val) => ({ 0: "нет ответа", 1: "есть ответ" })[val] ?? "" } },
        },
      },
      plugins: [tauLine],
    });
    const tauVal = document.getElementById("tau-val"), tauStats = document.getElementById("tau-stats");
    function updateTau() {
      const t = Number(tau.value);
      tauVal.textContent = t.toFixed(3);
      const ooc = pts.filter((p) => p.expects_refusal), inc = pts.filter((p) => !p.expects_refusal);
      const caught = ooc.filter((p) => p.score < t).length, lost = inc.filter((p) => p.score < t).length;
      tauStats.innerHTML =
        `<span class="tau-stat">отказ до LLM для «нет ответа»: <b>${caught} из ${ooc.length}</b></span>` +
        `<span class="tau-stat text-danger">ложный отказ на вопросах с ответом: <b>${lost} из ${inc.length}</b></span>`;
      thr.update("none");
    }
    tau.addEventListener("input", updateTau);
    updateTau();

    // 4. Динамика по прогонам (клик — открыть прогон)
    const trendRuns = D.trend.runs;
    new Chart(document.getElementById("ch-trend"), {
      type: "line",
      data: {
        labels: trendRuns.map((r) => new Date(r.ts).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })),
        datasets: D.trend.series.map((s, i) => ({
          label: s.label, data: s.data, borderColor: SERIES[i % SERIES.length], backgroundColor: SERIES[i % SERIES.length],
          spanGaps: true, tension: 0.25,
          pointRadius: trendRuns.map((r) => (r.current ? 6 : 3)),
        })),
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { boxWidth: 12 } },
          tooltip: { callbacks: { title: (it) => `${trendRuns[it[0].dataIndex].config} · ${it[0].label}` } } },
        scales: { y: unit, x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true } } },
        onClick: (_e, els) => {
          if (els.length) location.href = `/eval/agents/${meta.agent}/runs/${trendRuns[els[0].index].id}`;
        },
      },
    });

    // 5. Сравнение: forest plot — CI плавающим баром, средняя разница точкой
    const diffEl = document.getElementById("ch-diff");
    if (diffEl && D.diff.length) {
      const zero = { id: "zero", beforeDatasetsDraw(chart) {
        const x = chart.scales.x.getPixelForValue(0); const { top, bottom } = chart.chartArea; const ctx = chart.ctx;
        ctx.save(); ctx.strokeStyle = text; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.restore();
      } };
      new Chart(diffEl, {
        data: {
          labels: D.diff.map((d) => d.label),
          datasets: [
            { type: "bar", label: "95 % CI", data: D.diff.map((d) => [d.lo, d.hi]),
              backgroundColor: D.diff.map((d) => alpha(VERDICT[d.verdict], 0.35)), borderColor: D.diff.map((d) => VERDICT[d.verdict]),
              borderWidth: 1, borderSkipped: false, borderRadius: 3, barPercentage: 0.55 },
            // line без линии: при indexAxis "y" значение ложится на ось x, точка — по центру строки
            { type: "line", label: "Δ", data: D.diff.map((d) => d.diff), showLine: false,
              backgroundColor: D.diff.map((d) => VERDICT[d.verdict]), borderColor: D.diff.map((d) => VERDICT[d.verdict]),
              pointRadius: 4, pointHoverRadius: 6 },
          ],
        },
        options: {
          indexAxis: "y", maintainAspectRatio: false,
          plugins: { legend: { display: false },
            tooltip: { callbacks: { label: (c) => { const d = D.diff[c.dataIndex];
              return ` Δ ${d.diff >= 0 ? "+" : ""}${d.diff.toFixed(3)}  [${d.lo.toFixed(3)}; ${d.hi.toFixed(3)}] — ${d.verdict}`; } } } },
          scales: { x: { grid: { color: grid }, title: { display: true, text: "этот − база" } },
            y: { grid: { display: false }, ticks: { autoSkip: false } } },
        },
        plugins: [zero],
      });
    }

    // 6. Латентность по вопросам: ожидание первого токена + стрим
    const lat = D.latency;
    new Chart(document.getElementById("ch-latency"), {
      type: "bar",
      data: {
        labels: lat.map((r) => r.id),
        datasets: [
          { label: "до первого токена", data: lat.map((r) => (r.ttft ?? r.total) / 1000), backgroundColor: v("--c-answer"), stack: "t" },
          { label: "стрим ответа", data: lat.map((r) => r.ttft == null ? 0 : (r.total - r.ttft) / 1000), backgroundColor: alpha(v("--c-answer"), 0.35), stack: "t" },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw.toFixed(1)} с` } } },
        scales: { x: { stacked: true, ticks: { display: false }, grid: { display: false } },
          y: { stacked: true, grid: { color: grid }, title: { display: true, text: "секунды" } } },
      },
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
