// Страница «Под капотом»: живая схема (SSE trace:events), журнал, карта векторов Qdrant.
(() => {
  "use strict";
  const cfg = JSON.parse(document.getElementById("sys-config").textContent);
  const svg = document.getElementById("arch");
  const NS = "http://www.w3.org/2000/svg";
  const W = cfg.nodeW, H = cfg.nodeH;

  // ---------- схема ----------
  const nodes = {};
  svg.querySelectorAll("[data-node]").forEach((g) => {
    const [, x, y] = g.getAttribute("transform").match(/translate\(([-\d.]+),([-\d.]+)\)/).map(Number);
    nodes[g.dataset.node] = { g, x, y, group: [...g.classList].find((c) => c.startsWith("g-")) };
  });

  function edgePath(a, b) {
    const A = nodes[a], B = nodes[b];
    let p1, p2, horizontal = true;
    if (B.x >= A.x + W) { p1 = [A.x + W, A.y + H / 2]; p2 = [B.x, B.y + H / 2]; }
    else if (A.x >= B.x + W) { p1 = [A.x, A.y + H / 2]; p2 = [B.x + W, B.y + H / 2]; }
    else {
      horizontal = false;
      p1 = B.y > A.y ? [A.x + W / 2, A.y + H] : [A.x + W / 2, A.y];
      p2 = B.y > A.y ? [B.x + W / 2, B.y] : [B.x + W / 2, B.y + H];
    }
    const k = horizontal ? (p2[0] - p1[0]) * 0.5 : (p2[1] - p1[1]) * 0.5;
    const c1 = horizontal ? [p1[0] + k, p1[1]] : [p1[0], p1[1] + k];
    const c2 = horizontal ? [p2[0] - k, p2[1]] : [p2[0], p2[1] - k];
    return `M${p1} C${c1} ${c2} ${p2}`;
  }

  const edges = {};
  const edgesG = document.getElementById("edges");
  cfg.edges.forEach(([a, b]) => {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", edgePath(a, b));
    path.setAttribute("class", "edge");
    edgesG.appendChild(path);
    edges[`${a}|${b}`] = path;
  });

  function findEdge(src, dst) {
    if (edges[`${src}|${dst}`]) return { path: edges[`${src}|${dst}`], reverse: false };
    if (edges[`${dst}|${src}`]) return { path: edges[`${dst}|${src}`], reverse: true };
    return null;
  }

  function flash(el, cls, ms) {
    el.classList.remove(cls);
    void el.getBBox?.(); // перезапуск CSS-анимации
    el.classList.add(cls);
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove(cls), ms);
  }

  const dotsG = document.getElementById("dots");
  function animateDot(ev, duration) {
    const e = findEdge(ev.src, ev.dst);
    if (!e) return;
    const group = nodes[ev.src].group;
    e.path.classList.add(group);
    flash(e.path, "active", duration + 400);
    setTimeout(() => e.path.classList.remove(group), duration + 450);
    const len = e.path.getTotalLength();
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("r", 7);
    dot.setAttribute("class", `dot ${group}`);
    dotsG.appendChild(dot);
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / duration);
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      const pt = e.path.getPointAtLength(len * (e.reverse ? 1 - eased : eased));
      dot.setAttribute("cx", pt.x);
      dot.setAttribute("cy", pt.y);
      if (t < 1) requestAnimationFrame(step);
      else dot.remove();
    };
    requestAnimationFrame(step);
  }

  // ---------- очередь анимаций: события приходят пачками, показываем их по одному ----------
  const queue = [];
  let busy = false;
  const caption = document.getElementById("live-caption");
  function enqueue(ev) {
    queue.push(ev);
    if (!busy) drain();
  }
  function drain() {
    const ev = queue.shift();
    if (!ev) { busy = false; return; }
    busy = true;
    const fast = queue.length > 6; // ingest большой книги шлёт сотни событий
    const duration = fast ? 180 : 650;
    play(ev, duration);
    setTimeout(drain, fast ? 90 : 750);
  }
  function play(ev, duration) {
    caption.textContent = ev.label;
    if (nodes[ev.src]) flash(nodes[ev.src].g, "active", duration + 300);
    if (ev.dst && nodes[ev.dst]) {
      animateDot(ev, duration);
      setTimeout(() => flash(nodes[ev.dst].g, "active", 600), duration * 0.8);
    }
    logEvent(ev, true);
    mapOnEvent(ev);
  }

  // ---------- журнал ----------
  const log = document.getElementById("event-log");
  const TITLES = {};
  document.querySelectorAll(".node").forEach((g) => { TITLES[g.dataset.node] = g.querySelector(".node-title").textContent; });
  function logEvent(ev, fresh) {
    const li = document.createElement("li");
    li.className = (nodes[ev.src]?.group || "") + (fresh ? " fresh" : "");
    const time = new Date(ev.ts * 1000).toLocaleTimeString("ru-RU", { hour12: false }) +
      "." + String(Math.floor((ev.ts % 1) * 1000)).padStart(3, "0");
    const route = ev.dst ? `${TITLES[ev.src]} → ${TITLES[ev.dst]}` : TITLES[ev.src];
    li.innerHTML = `<div class="d-flex justify-content-between gap-2"><span class="ev-route"></span><span class="ev-time">${time}</span></div>
      <div class="ev-label"></div><div class="ev-kind"></div>`;
    li.querySelector(".ev-route").textContent = route;
    li.querySelector(".ev-label").textContent = ev.label;
    li.querySelector(".ev-kind").textContent = ev.kind;
    log.prepend(li);
    while (log.children.length > 200) log.lastChild.remove();
  }
  cfg.history.forEach((ev) => logEvent(ev, false));
  if (!cfg.history.length) {
    log.innerHTML = '<li class="text-body-secondary">Событий пока нет. Задайте вопрос или загрузите книгу.</li>';
  }

  // ---------- SSE ----------
  const liveDot = document.getElementById("live-dot");
  if (cfg.traceEnabled) {
    const es = new EventSource("/system/events");
    es.onopen = () => liveDot.classList.add("on");
    es.onerror = () => liveDot.classList.remove("on");
    es.addEventListener("trace", (m) => {
      if (log.firstElementChild?.classList.contains("text-body-secondary")) log.innerHTML = "";
      enqueue(JSON.parse(m.data));
    });
  } else {
    caption.textContent = "trace_enabled=false: живые события выключены";
  }

  // ---------- клик по узлу → описание технологии ----------
  const info = document.getElementById("node-info");
  function showNode(id) {
    const card = document.getElementById(`tech-${id}`);
    if (!card) return;
    Object.values(nodes).forEach((n) => n.g.classList.toggle("selected", n.g.dataset.node === id));
    info.className = `card mt-3 ${nodes[id].group}`;
    info.innerHTML = card.innerHTML +
      '<div class="card-footer small"><a href="#glossary">весь справочник ↓</a></div>';
    info.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  Object.entries(nodes).forEach(([id, n]) => {
    n.g.addEventListener("click", () => showNode(id));
    n.g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showNode(id); } });
  });

  // ---------- живые подписи узлов из фрагмента статистики ----------
  document.body.addEventListener("htmx:afterSettle", () => {
    const el = document.getElementById("node-badges");
    if (!el) return;
    const badges = JSON.parse(el.textContent);
    svg.querySelectorAll("[data-badge]").forEach((t) => { t.textContent = badges[t.dataset.badge] ?? ""; });
  });

  // ---------- карта векторов ----------
  const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"];
  const OTHER = "#a3a29c";
  const canvas = document.getElementById("map");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const tip = document.getElementById("map-tip");
  const select = document.getElementById("map-agent");
  const meta = document.getElementById("map-meta");
  const legend = document.getElementById("map-legend");
  const detail = document.getElementById("point-detail");
  const form = document.getElementById("sys-ask");
  let map = null, query = null, hits = new Map(), screen = [];

  const docColor = (i) => (i < PALETTE.length ? PALETTE[i] : OTHER);

  async function loadMap() {
    const id = select.value;
    const name = select.selectedOptions[0].dataset.name;
    document.getElementById("map-agent-link").href = `/agents/${id}`;
    history.replaceState(null, "", `?agent=${id}`);
    if (form) {
      form.setAttribute("hx-post", `/agents/${id}/chats/new/messages`);
      form.querySelector("input").placeholder = `Вопрос агенту «${name}»`;
      htmx.process(form);
    }
    query = null; hits = new Map();
    meta.textContent = "строю проекцию…";
    let resp = await fetch(`/system/agents/${id}/vector-map`);
    // 202: карту строит воркер в фоне (векторы из Qdrant + PCA) — спрашиваем, пока не готова
    for (let waited = 0; resp.status === 202; waited += 2) {
      meta.textContent = `строю проекцию в фоне… ${waited} с`;
      await new Promise((r) => setTimeout(r, 2000));
      if (select.value !== id) return;  // пользователь уже выбрал другого агента
      resp = await fetch(`/system/agents/${id}/vector-map`);
    }
    if (!resp.ok) { meta.textContent = "не удалось загрузить карту"; return; }
    map = await resp.json();
    const ex = map.explained.map((e) => `${(e * 100).toFixed(1)}%`).join(" + ");
    meta.textContent = map.points.length
      ? `${map.points.length} точек · ${map.dim} измерений → 2 · оси PCA объясняют ${ex} разброса`
      : "у агента ещё нет векторов: загрузите документ";
    legend.innerHTML = "";
    map.docs.forEach((d, i) => {
      const s = document.createElement("span");
      s.innerHTML = `<i class="sw" style="background:${docColor(i)}"></i>`;
      s.append(`${d.title} (${d.points})`);
      legend.appendChild(s);
    });
    draw();
  }

  function project(vec) {
    return map.components.map((c) => {
      let s = 0;
      for (let i = 0; i < vec.length; i++) s += (vec[i] - map.mean[i]) * c[i];
      return s;
    });
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    canvas.width = cw * dpr; canvas.height = ch * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    screen = [];
    if (!map || !map.points.length) return;
    const xs = map.points.map((p) => p.x), ys = map.points.map((p) => p.y);
    if (query) { xs.push(query.x); ys.push(query.y); }
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const pad = 24;
    const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (cw - 2 * pad);
    const sy = (y) => ch - pad - ((y - y0) / (y1 - y0 || 1)) * (ch - 2 * pad);

    // оси через центр масс (0,0 в PCA-координатах)
    ctx.strokeStyle = "rgba(128,128,128,.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx(0), pad / 2); ctx.lineTo(sx(0), ch - pad / 2);
    ctx.moveTo(pad / 2, sy(0)); ctx.lineTo(cw - pad / 2, sy(0)); ctx.stroke();

    const dim = hits.size > 0;
    map.points.forEach((p) => {
      const X = sx(p.x), Y = sy(p.y);
      screen.push({ X, Y, p });
      ctx.globalAlpha = dim && !hits.has(p.id) ? 0.25 : 0.8;
      ctx.fillStyle = docColor(p.doc);
      ctx.beginPath(); ctx.arc(X, Y, 3.5, 0, 7); ctx.fill();
    });
    ctx.globalAlpha = 1;

    if (query) {
      const QX = sx(query.x), QY = sy(query.y);
      screen.filter((s) => hits.has(s.p.id)).forEach((s, i) => {
        ctx.strokeStyle = "rgba(20,20,20,.35)"; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(QX, QY); ctx.lineTo(s.X, s.Y); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = "#111"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(s.X, s.Y, 8, 0, 7); ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = "#111"; ctx.font = "600 11px system-ui";
        ctx.fillText(hits.get(s.p.id).score.toFixed(3), s.X + 10, s.Y - 8 + (i % 2) * 4);
      });
      star(QX, QY, 12, 5.5);
      ctx.fillStyle = "#111"; ctx.font = "600 12px system-ui"; ctx.fillText("вопрос", QX - 22, QY - 16);
    }
  }

  function star(x, y, R, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (Math.PI / 5) * i - Math.PI / 2, rad = i % 2 ? r : R;
      ctx.lineTo(x + rad * Math.cos(a), y + rad * Math.sin(a));
    }
    ctx.closePath();
    ctx.fillStyle = "#111"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.stroke(); ctx.fill(); ctx.lineWidth = 1;
  }

  function nearest(e) {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 100;
    screen.forEach((s) => {
      const d = (s.X - mx) ** 2 + (s.Y - my) ** 2;
      if (d < bd) { bd = d; best = s; }
    });
    return best;
  }

  canvas.addEventListener("mousemove", (e) => {
    const s = nearest(e);
    if (!s) { tip.classList.add("d-none"); return; }
    const doc = map.docs[s.p.doc];
    const hit = hits.get(s.p.id);
    tip.textContent = "";
    const b = document.createElement("b"); b.textContent = doc.title; tip.append(b);
    tip.append(document.createElement("br"), `${s.p.chapter || "без главы"} · чанк #${s.p.ord}`);
    if (hit) tip.append(document.createElement("br"), `score ${hit.score.toFixed(3)}`);
    tip.style.left = `${Math.min(s.X + 12, canvas.clientWidth - 280)}px`;
    tip.style.top = `${s.Y + 12}px`;
    tip.classList.remove("d-none");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.add("d-none"));
  canvas.addEventListener("click", async (e) => {
    const s = nearest(e);
    if (!s) return;
    const resp = await fetch(`/system/agents/${select.value}/points/${s.p.id}`);
    if (!resp.ok) return;
    const d = await resp.json();
    renderVector(`${d.book_title || ""} · ${d.chapter_title || "без главы"} · чанк #${d.ord}`, d.vector, d.norm, d.text);
  });

  // Вектор целиком: 1024 клетки, синий < 0 < оранжевый (дивергентная шкала)
  function renderVector(title, vec, norm, text) {
    detail.innerHTML = `<div class="fw-semibold mb-1"></div>
      <canvas class="strip" width="${vec.length}" height="1"></canvas>
      <div class="strip-scale"><span>dim 0</span><span>синий &lt; 0 · серый ≈ 0 · оранжевый &gt; 0</span><span>dim ${vec.length - 1}</span></div>
      <div class="my-1">длина вектора ‖v‖ = ${norm.toFixed(3)} (нормирован), min ${Math.min(...vec).toFixed(3)}, max ${Math.max(...vec).toFixed(3)}</div>
      <div class="mb-1 text-body-secondary">первые числа: [${vec.slice(0, 8).map((v) => v.toFixed(3)).join(", ")}, …]</div>`;
    detail.querySelector(".fw-semibold").textContent = title;
    if (text) {
      const t = document.createElement("div");
      t.className = "chunk-text small"; t.textContent = text;
      detail.appendChild(t);
    }
    const c = detail.querySelector("canvas").getContext("2d");
    const img = c.createImageData(vec.length, 1);
    const scale = Math.max(...vec.map(Math.abs)) || 1;
    const neg = [42, 120, 214], mid = [233, 232, 228], pos = [235, 104, 52];
    vec.forEach((v, i) => {
      const t = Math.min(1, Math.abs(v) / scale);
      const to = v < 0 ? neg : pos;
      for (let k = 0; k < 3; k++) img.data[i * 4 + k] = mid[k] + (to[k] - mid[k]) * t;
      img.data[i * 4 + 3] = 255;
    });
    c.putImageData(img, 0, 0);
  }

  function mapOnEvent(ev) {
    if (!map || ev.agent_id !== map.agent_id) return;
    if (ev.kind === "query.ask") { query = null; hits = new Map(); draw(); }
    if (ev.kind === "query.embedded" && ev.data.vector) {
      const [x, y] = project(ev.data.vector);
      query = { x, y };
      hits = new Map();
      draw();
      renderVector("★ вектор вашего вопроса", ev.data.vector, Math.hypot(...ev.data.vector), null);
    }
    if (ev.kind === "query.found" && ev.data.hits) {
      hits = new Map(ev.data.hits.map((h) => [h.id, h]));
      draw();
    }
    if (ev.kind === "ingest.done") loadMap();
  }

  select.addEventListener("change", loadMap);
  window.addEventListener("resize", draw);
  loadMap();
})();
