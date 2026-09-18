/**
 * A small GitHub-flavored Markdown renderer.
 *
 * Why hand-rolled: the only dependency budget here is "CDN script tags", and
 * the LLM output we render is untrusted text. Escaping the entire document
 * before any HTML is generated makes XSS structurally impossible rather than
 * something we have to remember -- no raw HTML passthrough exists at all.
 *
 * Supports: fenced code (with language + mermaid), ATX headings, tables with
 * alignment, nested ordered/unordered lists, blockquotes, horizontal rules,
 * bold/italic/strike/inline-code, and links with a protocol allowlist.
 *
 * Placeholders use Unicode private-use characters, which cannot appear in
 * model output as meaningful text and are stripped from input defensively.
 */

const SENTINEL = "";
const FENCE_TOKEN = (i) => `${SENTINEL}FENCE${i}${SENTINEL}`;
const FENCE_RE = new RegExp(`${SENTINEL}FENCE(\\d+)${SENTINEL}`);
const FENCE_RE_G = new RegExp(`${SENTINEL}FENCE(\\d+)${SENTINEL}`, "g");
const INLINE_CODE_TOKEN = (i) => `${SENTINEL}IC${i}${SENTINEL}`;
const INLINE_CODE_RE_G = new RegExp(`${SENTINEL}IC(\\d+)${SENTINEL}`, "g");
const PIPE_TOKEN = `${SENTINEL}PIPE${SENTINEL}`;

const SAFE_PROTOCOL = /^(https?:|mailto:|#|\/)/i;

// Block parsing runs AFTER HTML-escaping, so a blockquote marker reaches the
// parser as "&gt;", not ">". Match both so the renderer is safe to call on
// either stage of the pipeline.
const QUOTE_RE = /^\s*(?:&gt;|>)/;
const QUOTE_STRIP_RE = /^\s*(?:&gt;|>)\s?/;

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Links are the one place an attribute is emitted, so gate the protocol. */
function safeUrl(url) {
  const trimmed = url.trim();
  // Text is escaped before this runs, so quotes cannot break out of the
  // attribute. This check exists to block javascript:/data: navigation.
  return SAFE_PROTOCOL.test(trimmed) ? trimmed : null;
}

/** Inline formatting. Input MUST already be HTML-escaped. */
function renderInline(text) {
  const codes = [];
  let out = text.replace(/`([^`\n]+)`/g, (_m, code) => {
    codes.push(code);
    return INLINE_CODE_TOKEN(codes.length - 1);
  });

  // Images degrade to links -- we never load remote resources from model output.
  out = out.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (m, alt, url) => {
    const safe = safeUrl(url);
    return safe ? `<a href="${safe}" target="_blank" rel="noopener noreferrer">${alt || safe}</a>` : m;
  });

  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
    const safe = safeUrl(url);
    return safe ? `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>` : m;
  });

  out = out
    .replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s.,;:)!?])/g, "$1<em>$2</em>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>");

  return out.replace(INLINE_CODE_RE_G, (_m, i) => `<code>${codes[Number(i)]}</code>`);
}

// --- tables ---------------------------------------------------------------

function splitRow(line) {
  let s = line.trim().split("\\|").join(PIPE_TOKEN);
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((cell) => cell.trim().split(PIPE_TOKEN).join("|"));
}

function isTableDivider(line) {
  return /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(line) && line.includes("-");
}

function parseTable(lines, start) {
  const header = splitRow(lines[start]);
  const aligns = splitRow(lines[start + 1]).map((cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    if (left && right) return "center";
    if (right) return "right";
    return "";
  });

  const rows = [];
  let i = start + 2;
  while (i < lines.length && lines[i].trim().startsWith("|")) {
    rows.push(splitRow(lines[i]));
    i += 1;
  }

  const styleFor = (col) => (aligns[col] ? ` style="text-align:${aligns[col]}"` : "");
  const head = header.map((c, n) => `<th${styleFor(n)}>${renderInline(c)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = header.map((_h, n) => `<td${styleFor(n)}>${renderInline(row[n] ?? "")}</td>`);
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");

  return {
    html: `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`,
    next: i,
  };
}

// --- lists ----------------------------------------------------------------

const LIST_ITEM_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

function renderItems(items) {
  if (!items.length) return "";
  const base = Math.min(...items.map((it) => it.indent));
  const first = items.find((it) => it.indent === base);
  const ordered = Boolean(first && first.ordered);

  let html = ordered ? "<ol>" : "<ul>";
  let i = 0;
  while (i < items.length) {
    const item = items[i];
    const children = [];
    let j = i + 1;
    while (j < items.length && items[j].indent > item.indent) {
      children.push(items[j]);
      j += 1;
    }
    const body = renderInline(item.lines.join(" ").trim());
    html += `<li>${body}${renderItems(children)}</li>`;
    i = j;
  }
  return html + (ordered ? "</ol>" : "</ul>");
}

function parseList(lines, start) {
  const items = [];
  let i = start;

  while (i < lines.length) {
    const line = lines[i];
    const match = LIST_ITEM_RE.exec(line);

    if (match) {
      items.push({
        indent: match[1].replace(/\t/g, "  ").length,
        ordered: /\d/.test(match[2]),
        lines: [match[3]],
      });
      i += 1;
      continue;
    }

    // A blank line only continues the list if another item follows.
    if (!line.trim()) {
      const next = lines[i + 1];
      if (items.length && next && LIST_ITEM_RE.test(next)) {
        i += 1;
        continue;
      }
      break;
    }

    // Lazy continuation of the previous item's paragraph.
    if (items.length && /^\s{2,}\S/.test(line) && !FENCE_RE.test(line)) {
      items[items.length - 1].lines.push(line.trim());
      i += 1;
      continue;
    }

    break;
  }

  return { html: renderItems(items), next: i };
}

// --- block parser ---------------------------------------------------------

function startsNewBlock(line) {
  return (
    !line.trim() ||
    /^#{1,6}\s/.test(line) ||
    QUOTE_RE.test(line) ||
    LIST_ITEM_RE.test(line) ||
    line.trim().startsWith("|") ||
    FENCE_RE.test(line) ||
    /^\s*([-*_])\s*(\1\s*){2,}$/.test(line)
  );
}

function parseBlocks(lines) {
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    // A fence placeholder standing alone on its line.
    if (FENCE_RE.test(line.trim()) && line.trim().replace(FENCE_RE, "") === "") {
      out.push(line.trim());
      i += 1;
      continue;
    }

    const heading = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
      out.push("<hr />");
      i += 1;
      continue;
    }

    if (line.trim().startsWith("|") && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const { html, next } = parseTable(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    if (QUOTE_RE.test(line)) {
      const quoted = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        quoted.push(lines[i].replace(QUOTE_STRIP_RE, ""));
        i += 1;
      }
      out.push(`<blockquote>${parseBlocks(quoted)}</blockquote>`);
      continue;
    }

    if (LIST_ITEM_RE.test(line)) {
      const { html, next } = parseList(lines, i);
      // Guard against a pathological input that consumes no lines.
      if (next > i) {
        out.push(html);
        i = next;
        continue;
      }
    }

    // Paragraph: run until a blank line or the start of another block.
    const para = [line];
    i += 1;
    while (i < lines.length && !startsNewBlock(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(`<p>${renderInline(para.join("\n").trim()).split("\n").join("<br />")}</p>`);
  }

  return out.join("\n");
}

// --- entrypoint -----------------------------------------------------------

/**
 * @param {string} source raw markdown, assumed untrusted
 * @returns {string} HTML safe to assign to innerHTML
 */
export function renderMarkdown(source) {
  const fences = [];
  // Strip our own sentinel so crafted input cannot forge a placeholder.
  let text = String(source ?? "")
    .split(SENTINEL)
    .join("")
    .replace(/\r\n?/g, "\n");

  // Closed fences first.
  text = text.replace(/^[ \t]*```([^\n`]*)\n([\s\S]*?)\n?^[ \t]*```[ \t]*$/gm, (_m, lang, code) => {
    fences.push({ lang: lang.trim().toLowerCase(), code });
    return FENCE_TOKEN(fences.length - 1);
  });

  // A trailing unclosed fence happens whenever the model hits its token cap.
  // Render what we got rather than dumping stray backticks into the page.
  text = text.replace(/^[ \t]*```([^\n`]*)\n([\s\S]*)$/m, (_m, lang, code) => {
    fences.push({ lang: lang.trim().toLowerCase(), code, unclosed: true });
    return FENCE_TOKEN(fences.length - 1);
  });

  const escaped = escapeHtml(text);
  let html = parseBlocks(escaped.split("\n"));

  html = html.replace(FENCE_RE_G, (_m, index) => {
    const block = fences[Number(index)];
    if (!block) return "";
    const code = escapeHtml(block.code.replace(/\s+$/, ""));

    if (block.lang === "mermaid") {
      return `<div class="mermaid" data-mermaid-pending="1">${code}</div>`;
    }

    const label = block.unclosed
      ? `${block.lang || "code"} · truncated`
      : block.lang;

    return (
      `<div class="code-wrap">` +
      (label ? `<span class="code-lang">${escapeHtml(label)}</span>` : "") +
      `<button class="code-copy" type="button" data-copy-code>Copy</button>` +
      `<pre><code>${code}</code></pre></div>`
    );
  });

  return html;
}

export { escapeHtml };
