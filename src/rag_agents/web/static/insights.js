// Страница «Аналитика»: локальное время, раскрытие трейсов, график (Chart.js с CDN).
(function () {
  const fmt = {
    time: { hour: "2-digit", minute: "2-digit", second: "2-digit" },
    datetime: { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" },
  };

  function localTimes(root) {
    root.querySelectorAll("time[data-fmt]").forEach((el) => {
      const d = new Date(el.getAttribute("datetime"));
      if (!isNaN(d)) el.textContent = d.toLocaleString("ru-RU", fmt[el.dataset.fmt] || fmt.datetime);
    });
  }

  let chart = null;
  function seriesChart(root) {
    const canvas = root.querySelector("#series-chart");
    const data = root.querySelector("#series-data");
    if (!canvas || !data || !window.Chart) return;
    const rows = JSON.parse(data.textContent);
    const css = getComputedStyle(document.body);
    const grid = css.getPropertyValue("--bs-border-color-translucent") || "rgba(0,0,0,.1)";
    const text = css.getPropertyValue("--bs-secondary-color") || "#6c757d";
    const hourly = rows.length > 1 && new Date(rows[1].ts) - new Date(rows[0].ts) < 86400000;
    const label = (ts) => new Date(ts).toLocaleString("ru-RU",
      hourly ? { hour: "2-digit", minute: "2-digit" } : { day: "2-digit", month: "2-digit" });
    if (chart) chart.destroy();
    chart = new Chart(canvas, {
      data: {
        labels: rows.map((r) => label(r.ts)),
        datasets: [
          { type: "bar", label: "Вопросы (вызовы LLM)", data: rows.map((r) => r.questions),
            backgroundColor: "rgba(79,125,243,.55)", borderRadius: 3, yAxisID: "y" },
          { type: "line", label: "Стоимость, $", data: rows.map((r) => r.cost_usd),
            borderColor: "#19a974", backgroundColor: "#19a974", cubicInterpolationMode: "monotone", pointRadius: 2, yAxisID: "cost" },
        ],
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: text, boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => c.dataset.yAxisID === "cost"
            ? ` $${Number(c.raw).toFixed(4)}` : ` ${c.raw} вопр.` } },
        },
        scales: {
          x: { ticks: { color: text, maxRotation: 0, autoSkip: true }, grid: { display: false } },
          y: { beginAtZero: true, ticks: { color: text, precision: 0 }, grid: { color: grid } },
          cost: { position: "right", beginAtZero: true, grid: { display: false },
            ticks: { color: text, callback: (v) => "$" + Number(v).toFixed(3) } },
        },
      },
    });
  }

  function init(root) {
    localTimes(root);
    seriesChart(root);
  }

  // Клик по строке трейса: первый раз HTMX грузит водопад (hx-trigger="click once"), дальше — toggle
  document.addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const row = e.target.closest(".trace-row");
    if (!row) return;
    const target = document.getElementById(row.dataset.expand);
    const open = target.classList.toggle("d-none") === false;
    row.setAttribute("aria-expanded", String(open));
  });

  document.addEventListener("DOMContentLoaded", () => init(document));
  document.addEventListener("htmx:afterSwap", (e) => init(e.target));
  // Chart.js грузится с defer и может прийти позже фрагмента
  window.addEventListener("load", () => seriesChart(document));
})();
