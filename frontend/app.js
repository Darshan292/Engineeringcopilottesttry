/**
 * Engineering Copilot -- frontend controller.
 *
 * Vanilla ES modules, no build step, no framework. Tabs, samples, and
 * placeholders all come from GET /api/tools, so adding a fifth tool on the
 * backend makes it appear here with no frontend change.
 */

import { renderMarkdown, escapeHtml } from "./markdown.js";

const $ = (id) => document.getElementById(id);

const els = {
  tabs: $("tabs"),
  title: $("tool-title"),
  blurb: $("tool-blurb"),
  input: $("input"),
  runBtn: $("run-btn"),
  runLabel: $("run-label"),
  sampleBtn: $("sample-btn"),
  clearBtn: $("clear-btn"),
  emptySample: $("empty-sample"),
  charCount: $("char-count"),
  sampleNote: $("sample-note"),
  output: $("output"),
  stats: $("stats"),
  copyBtn: $("copy-btn"),
  downloadBtn: $("download-btn"),
  modelName: $("model-name"),
  statusDot: $("status-dot"),
  banner: $("banner"),
  bannerTitle: $("banner-title"),
  bannerText: $("banner-text"),
  bannerClose: $("banner-close"),
  themeToggle: $("theme-toggle"),
};

const MAX_INPUT_CHARS = 60000;

const state = {
  tools: [],
  activeTool: null,
  lastMarkdown: "",
  running: false,
  /** Per-tool draft + result, so switching tabs never loses work. */
  drafts: new Map(),
  results: new Map(),
};

// --- theme ----------------------------------------------------------------

function initTheme() {
  let stored = null;
  try {
    stored = localStorage.getItem("copilot-theme");
  } catch {
    // Private mode / blocked storage. Fall through to the media query.
  }
  const preferred =
    stored || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = preferred;

  els.themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("copilot-theme", next);
    } catch {
      /* not worth surfacing */
    }
  });
}

// --- banner ---------------------------------------------------------------

function showBanner(title, text, kind = "warn") {
  els.bannerTitle.textContent = title;
  els.bannerText.innerHTML = text; // caller-controlled, never model output
  els.banner.hidden = false;
  els.banner.classList.toggle("banner-bad", kind === "bad");
}

els.bannerClose.addEventListener("click", () => {
  els.banner.hidden = true;
});

// --- config ---------------------------------------------------------------

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();

    els.modelName.textContent = cfg.model || "unknown";
    els.modelName.parentElement.title = `Model: ${cfg.model}\nEndpoint: ${cfg.base_url}\nTemperature: ${cfg.temperature}\nMax tokens: ${cfg.max_tokens}`;

    if (!cfg.api_key_configured) {
      els.statusDot.className = "dot dot-bad";
      showBanner(
        "No GROQ_API_KEY set.",
        ' Every request will fail until you configure one. Run <code>cp .env.example .env</code>, ' +
          'paste a free key from <a href="https://console.groq.com/keys" target="_blank" rel="noopener">console.groq.com/keys</a>, ' +
          "then restart the server.",
        "bad"
      );
    } else if (cfg.model_warning) {
      els.statusDot.className = "dot dot-warn";
      showBanner("Model problem.", ` ${escapeHtml(cfg.model_warning)}`, "bad");
    } else {
      els.statusDot.className = "dot dot-ok";
    }
  } catch {
    els.modelName.textContent = "backend unreachable";
    els.statusDot.className = "dot dot-bad";
    showBanner(
      "Cannot reach the backend.",
      " Is uvicorn running? Start it with <code>./run.sh</code> or <code>uvicorn backend.main:app --reload</code>.",
      "bad"
    );
  }
}

// --- tabs -----------------------------------------------------------------

async function loadTools() {
  const res = await fetch("/api/tools");
  if (!res.ok) throw new Error(`GET /api/tools returned ${res.status}`);
  const data = await res.json();
  state.tools = data.tools || [];

  els.tabs.innerHTML = "";
  state.tools.forEach((tool, index) => {
    const btn = document.createElement("button");
    btn.className = "tab";
    btn.type = "button";
    btn.role = "tab";
    btn.textContent = tool.title;
    btn.dataset.toolId = tool.id;
    btn.setAttribute("aria-selected", String(index === 0));
    btn.addEventListener("click", () => selectTool(tool.id));
    els.tabs.appendChild(btn);
  });

  if (state.tools.length) selectTool(state.tools[0].id);
}

function currentTool() {
  return state.tools.find((t) => t.id === state.activeTool) || null;
}

function selectTool(toolId) {
  // Stash the current draft before switching away.
  if (state.activeTool) state.drafts.set(state.activeTool, els.input.value);

  state.activeTool = toolId;
  const tool = currentTool();
  if (!tool) return;

  for (const btn of els.tabs.children) {
    btn.setAttribute("aria-selected", String(btn.dataset.toolId === toolId));
  }

  els.title.textContent = tool.title;
  els.blurb.textContent = tool.blurb;
  els.input.placeholder = tool.placeholder || "Paste content here...";
  els.input.value = state.drafts.get(toolId) || "";
  updateCharCount();

  const sample = tool.sample || {};
  els.sampleNote.textContent = sample.label
    ? `Sample: ${sample.label} - ${sample.description || ""}`
    : "";
  els.sampleBtn.disabled = !sample.content;

  const prior = state.results.get(toolId);
  if (prior) {
    paintResult(prior);
  } else {
    resetOutput();
  }
}

// --- input ----------------------------------------------------------------

function updateCharCount() {
  const n = els.input.value.length;
  els.charCount.textContent = `${n.toLocaleString()} chars`;
  els.charCount.classList.toggle("over", n > MAX_INPUT_CHARS);
  els.runBtn.disabled = state.running || n === 0 || n > MAX_INPUT_CHARS;
}

els.input.addEventListener("input", updateCharCount);

els.input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    run();
  }
});

function loadSample() {
  const tool = currentTool();
  if (!tool || !tool.sample || !tool.sample.content) return;
  els.input.value = tool.sample.content;
  updateCharCount();
  els.input.focus();
  els.input.setSelectionRange(0, 0);
  els.input.scrollTop = 0;
}

els.sampleBtn.addEventListener("click", loadSample);
els.emptySample.addEventListener("click", loadSample);

els.clearBtn.addEventListener("click", () => {
  els.input.value = "";
  updateCharCount();
  els.input.focus();
});

// --- output ---------------------------------------------------------------

function resetOutput() {
  state.lastMarkdown = "";
  els.stats.textContent = "";
  els.copyBtn.disabled = true;
  els.downloadBtn.disabled = true;
  els.output.innerHTML =
    '<div class="empty-state">' +
    '<p class="empty-title">Nothing yet.</p>' +
    "<p>Paste something in and hit <strong>Run</strong>.</p>" +
    '<p class="empty-hint">No input handy? <button class="linklike" id="empty-sample-inline">Load the sample</button> for this tool.</p>' +
    "</div>";
  const inline = $("empty-sample-inline");
  if (inline) inline.addEventListener("click", loadSample);
}

function showThinking() {
  const tool = currentTool();
  els.output.innerHTML =
    '<div class="thinking"><span class="spinner"></span>' +
    `<span>Asking the model to ${escapeHtml((tool && tool.title) || "run").toLowerCase()}...</span></div>` +
    '<p class="thinking-hint">Typically 3-15 seconds. Longer inputs take longer.</p>';
  els.stats.textContent = "";
  els.copyBtn.disabled = true;
  els.downloadBtn.disabled = true;
}

function showError(message, hint) {
  els.output.innerHTML =
    '<div class="error-box"><h3>Request failed</h3>' +
    `<p>${escapeHtml(message)}</p>` +
    (hint ? `<p class="hint">${escapeHtml(hint)}</p>` : "") +
    "</div>";
  els.stats.textContent = "";
}

function formatStats(result) {
  const parts = [`${(result.elapsed_ms / 1000).toFixed(1)}s`];
  const usage = result.usage || {};
  if (usage.total_tokens) parts.push(`${usage.total_tokens.toLocaleString()} tok`);
  if (usage.prompt_tokens && usage.completion_tokens) {
    parts.push(`${usage.prompt_tokens.toLocaleString()} in / ${usage.completion_tokens.toLocaleString()} out`);
  }
  if (result.model) parts.push(result.model);
  return parts.join(" · ");
}

function paintResult(result) {
  state.lastMarkdown = result.markdown;

  const truncatedNote = result.truncated
    ? '<div class="truncated-note"><strong>Output was cut off</strong> - the model hit its token cap. ' +
      "Raise <code>GROQ_MAX_TOKENS</code> in <code>.env</code>, or run against a smaller input.</div>"
    : "";

  els.output.innerHTML = truncatedNote + renderMarkdown(result.markdown);
  els.stats.textContent = formatStats(result);
  els.copyBtn.disabled = false;
  els.downloadBtn.disabled = false;
  els.output.parentElement.scrollTop = 0;

  wireCodeCopyButtons();
  renderMermaidBlocks();
}

function wireCodeCopyButtons() {
  for (const btn of els.output.querySelectorAll("[data-copy-code]")) {
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code");
      if (!code) return;
      await copyText(code.textContent || "");
      const original = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = original;
      }, 1200);
    });
  }
}

/**
 * Mermaid is loaded lazily and only when a diagram actually appears, so the
 * app stays fully functional with no network access to the CDN.
 */
let mermaidPromise = null;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")
      .then((mod) => {
        const mermaid = mod.default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: document.documentElement.dataset.theme === "light" ? "default" : "dark",
        });
        return mermaid;
      })
      .catch((err) => {
        mermaidPromise = null; // allow a retry on the next render
        throw err;
      });
  }
  return mermaidPromise;
}

async function renderMermaidBlocks() {
  const nodes = [...els.output.querySelectorAll(".mermaid[data-mermaid-pending]")];
  if (!nodes.length) return;

  try {
    const mermaid = await loadMermaid();
    for (const node of nodes) {
      node.removeAttribute("data-mermaid-pending");
      const source = node.textContent || "";
      try {
        const { svg } = await mermaid.render(`mmd-${Math.random().toString(36).slice(2)}`, source);
        node.innerHTML = svg;
      } catch (err) {
        // A malformed diagram should not eat the surrounding document.
        node.innerHTML =
          '<div class="mermaid-error">Diagram failed to render:<br />' +
          `${escapeHtml(String(err && err.message ? err.message : err))}</div>` +
          `<pre><code>${escapeHtml(source)}</code></pre>`;
      }
    }
  } catch {
    for (const node of nodes) {
      node.removeAttribute("data-mermaid-pending");
      const source = node.textContent || "";
      node.innerHTML =
        '<div class="mermaid-error">Mermaid could not be loaded from the CDN (offline?). ' +
        "Showing the diagram source instead.</div>" +
        `<pre><code>${escapeHtml(source)}</code></pre>`;
    }
  }
}

// --- run ------------------------------------------------------------------

async function run() {
  if (state.running) return;
  const tool = currentTool();
  const text = els.input.value.trim();
  if (!tool || !text) return;

  state.running = true;
  els.runBtn.disabled = true;
  els.runLabel.textContent = "Running";
  showThinking();

  try {
    const res = await fetch(`/api/${tool.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: text }),
    });

    let payload;
    try {
      payload = await res.json();
    } catch {
      throw new Error(`Backend returned ${res.status} with an unreadable body.`);
    }

    if (!res.ok) {
      showError(payload.error || payload.detail || `Request failed (${res.status}).`, payload.hint);
      return;
    }

    state.results.set(tool.id, payload);
    paintResult(payload);
  } catch (err) {
    showError(
      String(err && err.message ? err.message : err),
      "If this says 'Failed to fetch', the backend is not running or the page was opened as a file:// URL."
    );
  } finally {
    state.running = false;
    els.runLabel.textContent = "Run";
    updateCharCount();
  }
}

els.runBtn.addEventListener("click", run);

// --- copy / download ------------------------------------------------------

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // clipboard API needs a secure context; http://localhost qualifies, but
    // http://192.168.x.x does not, so keep the old path around.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    document.body.removeChild(ta);
    return ok;
  }
}

els.copyBtn.addEventListener("click", async () => {
  if (!state.lastMarkdown) return;
  const ok = await copyText(state.lastMarkdown);
  els.copyBtn.textContent = ok ? "Copied" : "Copy failed";
  setTimeout(() => {
    els.copyBtn.textContent = "Copy";
  }, 1400);
});

els.downloadBtn.addEventListener("click", () => {
  if (!state.lastMarkdown) return;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const blob = new Blob([state.lastMarkdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.activeTool}-${stamp}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// --- boot -----------------------------------------------------------------

(async function boot() {
  initTheme();
  els.emptySample.addEventListener("click", loadSample);
  await Promise.all([loadConfig(), loadTools().catch(() => {})]);
  updateCharCount();
})();
