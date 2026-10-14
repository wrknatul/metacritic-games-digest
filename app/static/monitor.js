(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...children) => {
    const n = document.createElement(tag);
    Object.assign(n, props);
    for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(c ?? ""));
    return n;
  };
  const time = (iso) => (iso ? new Date(iso).toLocaleTimeString("ru-RU") : "—");
  const dateTime = (iso) => (iso ? new Date(iso).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—");
  const ago = (ts) => {
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    return s < 60 ? `${s} с` : `${Math.floor(s / 60)} мин ${s % 60} с`;
  };
  const PHASES = { discovery: "выбор игр", details: "карточки и отзывы", finished: "завершён" };
  const STATUS = { idle: "свободен", busy: "работает", error: "ошибка" };
  const RUN_STATUS = { running: "идёт", done: "готово", failed: "упал" };
  let state = JSON.parse($("initial").textContent);

  function tile(label, value, hint) {
    return el("div", { className: "tile" }, el("div", { className: "tile-value" }, String(value)), el("div", { className: "tile-label" }, label), hint ? el("div", { className: "tile-hint" }, hint) : "");
  }

  function render() {
    const st = state.stats || {};
    $("tiles").replaceChildren(
      tile("игр в базе", st.games_total ?? 0),
      tile("обработано сегодня", st.processed_today ?? 0, `день ${st.day || ""}, след. страница SEE ALL: ${st.next_page ?? 1}`),
      tile("запусков сегодня", st.runs_today ?? 0),
      tile("следующий запуск", state.next_run_at ? time(state.next_run_at) : "—", `каждые ${state.interval_minutes} мин`),
      tile("резюме отзывов", st.with_summaries ?? 0, "игр с резюме"),
      tile("летсплеев разобрано", st.letsplay_done ?? 0, st.letsplay_enabled === false ? "выключено" : `в очереди: ${st.letsplay_pending ?? 0}`),
      tile("LLM вызовов сегодня", `${st.llm_calls_today ?? 0} / ${st.llm_limit ?? "?"}`, `${st.llm_model || ""} · $${(st.llm_cost_today ?? 0).toFixed(4)}${st.llm_enabled ? "" : " · ключ не задан"}`)
    );

    const r = state.run;
    const runBox = $("run");
    if (r) {
      const pct = r.total ? Math.round(((r.done + r.failed) / r.total) * 100) : 0;
      const bar = el("div", { className: "bar" }, el("div", { className: "bar-fill", style: `width:${pct}%` }));
      runBox.replaceChildren(
        el("p", {}, `#${r.id} · ${r.trigger === "manual" ? "вручную" : "по расписанию"} · этап: ${PHASES[r.phase] || r.phase} · идёт ${ago(r.started_at)}`),
        bar,
        el("p", {}, `Карточки: ${r.done + r.failed} из ${r.total} (ошибок ${r.failed}, новых ${r.new}). Летсплеи разбирает отдельный воркер по очереди.`),
        el("p", { className: "muted small" }, (r.sources || []).join(" · "))
      );
    } else {
      const last = state.last_run;
      runBox.replaceChildren(el("p", { className: "muted" }, last ? `Сейчас запуска нет. Последний запуск #${last.id}: ${last.done} ок, ${last.failed} ошибок, новых ${last.new}.` : "Сейчас запуска нет."));
    }

    const workers = state.workers || [];
    $("workers").replaceChildren(
      ...(workers.length
        ? workers.map((w) =>
            el("tr", {},
              el("td", {}, w.name),
              el("td", {}, el("span", { className: `pill ${w.status}` }, STATUS[w.status] || w.status)),
              el("td", {}, w.game || "—"),
              el("td", {}, w.stage || (w.last_error && w.status === "error" ? w.last_error : "—")),
              el("td", {}, w.status === "busy" ? ago(w.since) : "—"),
              el("td", {}, `${w.done} / ${w.failed}`)))
        : [el("tr", {}, el("td", { colSpan: 6, className: "muted" }, "Воркеров нет."))])
    );

    $("runs").replaceChildren(
      ...(state.runs || []).map((x) =>
        el("tr", { title: x.error || (x.sources || []).join(" · ") },
          el("td", {}, String(x.id)),
          el("td", {}, x.trigger === "manual" ? "вручную" : "плановый"),
          el("td", {}, el("span", { className: `pill ${x.status}` }, RUN_STATUS[x.status] || x.status)),
          el("td", {}, dateTime(x.started_at)),
          el("td", {}, String(x.games_total)),
          el("td", {}, String(x.games_failed)),
          el("td", {}, String(x.games_new))))
    );

    $("log").replaceChildren(
      ...(state.log || []).slice().reverse().map((e) => el("li", { className: e.level }, el("span", { className: "muted" }, time(e.ts) + " "), e.msg))
    );

    const btn = $("run-btn");
    btn.disabled = !!state.running;
    btn.textContent = state.running ? "Обработка идёт…" : "Запустить обработку сейчас";
  }

  // SSE first; some proxies buffer event streams, so with no event in 5 s fall back to polling
  function poll() {
    $("conn").textContent = "обновление раз в 2 с";
    $("conn").className = "pill busy";
    const tick = async () => {
      try {
        const resp = await fetch("/api/monitor", { cache: "no-store" });
        if (resp.ok) { state = await resp.json(); render(); }
      } catch (e) {
        $("conn").textContent = "нет связи, повтор…";
        $("conn").className = "pill error";
      }
    };
    tick();
    setInterval(tick, 2000);
  }

  function connect() {
    let gotEvent = false;
    const es = new EventSource("/api/monitor/stream");
    es.onmessage = (ev) => {
      gotEvent = true;
      $("conn").textContent = "в реальном времени";
      $("conn").className = "pill busy";
      state = JSON.parse(ev.data);
      render();
    };
    es.onerror = () => { if (gotEvent) { $("conn").textContent = "переподключение…"; $("conn").className = "pill error"; } };
    setTimeout(() => { if (!gotEvent) { es.close(); poll(); } }, 5000);
  }

  $("run-btn").addEventListener("click", async () => {
    const headers = {};
    if (state.token_required) {
      const t = window.prompt("Токен администратора");
      if (!t) return;
      headers["X-Admin-Token"] = t;
    }
    $("run-btn").disabled = true;
    try {
      const resp = await fetch("/api/runs", { method: "POST", headers });
      const body = await resp.json();
      $("run-msg").textContent = resp.ok ? `Запуск #${body.run_id} начат.` : body.message || "Не удалось запустить";
    } catch (e) {
      $("run-msg").textContent = "Сервер не ответил: " + e;
      $("run-btn").disabled = false;
    }
  });

  render();
  connect();
})();
