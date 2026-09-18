/**
 * Tests for the hand-rolled markdown renderer.
 *
 * Run with:  node --test tests/
 *
 * The security cases are the point of this file. The renderer's whole safety
 * argument is "everything is escaped before any HTML exists", so these assert
 * that no model output can inject a tag, an event handler, or a javascript: URL.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { renderMarkdown } from "../frontend/markdown.js";

// --- security -------------------------------------------------------------

test("raw HTML in model output is escaped, never executed", () => {
  const html = renderMarkdown('<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">');
  assert.ok(!html.includes("<script>"), "script tag leaked");
  assert.ok(!html.includes("<img"), "img tag leaked");
  assert.ok(!html.includes('onerror="alert(2)"'), "live event handler leaked");
  assert.ok(html.includes("onerror=&quot;"), "attribute should survive as inert text");
  assert.ok(html.includes("&lt;script&gt;"));
});

test("javascript: and data: links are not turned into anchors", () => {
  for (const bad of ["javascript:alert(1)", "data:text/html,<script>alert(1)</script>", "vbscript:x"]) {
    const html = renderMarkdown(`[click](${bad})`);
    assert.ok(!html.includes("<a href"), `anchor emitted for ${bad}`);
  }
});

test("http, https, mailto, anchor and relative links are allowed", () => {
  for (const good of ["https://example.com", "http://example.com", "mailto:a@b.co", "#section", "/docs"]) {
    assert.ok(renderMarkdown(`[x](${good})`).includes(`<a href="${good}"`), `blocked ${good}`);
  }
});

test("quotes in link text cannot break out of the href attribute", () => {
  const html = renderMarkdown('[a](https://e.com" onmouseover="alert(1))');
  assert.ok(!html.includes('onmouseover="alert(1)"'));
});

test("crafted sentinel characters in input cannot forge a code placeholder", () => {
  const html = renderMarkdown("FENCE0 plus text");
  assert.ok(!html.includes("<pre>"), "forged fence placeholder was honoured");
  assert.ok(html.includes("FENCE0"));
});

test("HTML inside a fenced code block stays inert", () => {
  const html = renderMarkdown("```html\n<script>alert(1)</script>\n```");
  assert.ok(html.includes("<pre><code>"));
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(!html.includes("<script>"));
});

// --- block elements -------------------------------------------------------

test("ATX headings render at the right level", () => {
  assert.ok(renderMarkdown("# One").includes("<h1>One</h1>"));
  assert.ok(renderMarkdown("### Three").includes("<h3>Three</h3>"));
  assert.ok(renderMarkdown("###### Six").includes("<h6>Six</h6>"));
  // Not a heading without the space.
  assert.ok(!renderMarkdown("#NotAHeading").includes("<h1>"));
});

test("fenced code keeps its language label and a copy button", () => {
  const html = renderMarkdown("```python\ndef f():\n    return 1\n```");
  assert.ok(html.includes('class="code-lang">python<'));
  assert.ok(html.includes("data-copy-code"));
  assert.ok(html.includes("def f():"));
});

test("an unclosed fence (hit the token cap) still renders as code", () => {
  const html = renderMarkdown("Some text\n\n```yaml\nopenapi: 3.1.0\ninfo:\n  title: cut off");
  assert.ok(html.includes("truncated"));
  assert.ok(html.includes("openapi: 3.1.0"));
  assert.ok(!html.includes("```"));
});

test("mermaid blocks become a mermaid container, not a pre", () => {
  const html = renderMarkdown("```mermaid\ngraph TD;\nA-->B;\n```");
  assert.ok(html.includes('class="mermaid"'));
  assert.ok(html.includes("data-mermaid-pending"));
  assert.ok(!html.includes("<pre>"));
});

test("tables render with headers, body cells and alignment", () => {
  const html = renderMarkdown(
    ["| Time | Event | Owner |", "| --- | :---: | ---: |", "| 02:16 | Pool full | [ONCALL] |"].join("\n")
  );
  assert.ok(html.includes("<table>"));
  assert.ok(html.includes("<th>Time</th>"));
  assert.ok(html.includes('<th style="text-align:center">Event</th>'));
  assert.ok(html.includes('<th style="text-align:right">Owner</th>'));
  assert.ok(html.includes("<td>02:16</td>"));
});

test("a short table row is padded rather than dropping the row", () => {
  const html = renderMarkdown(["| A | B |", "| --- | --- |", "| only |"].join("\n"));
  assert.ok(html.includes("<td>only</td>"));
  assert.equal((html.match(/<tr>/g) || []).length, 2);
});

test("escaped pipes stay inside their cell", () => {
  const html = renderMarkdown(["| Cmd |", "| --- |", "| a \\| b |"].join("\n"));
  assert.ok(html.includes("<td>a | b</td>"));
});

test("unordered and ordered lists nest by indentation", () => {
  const html = renderMarkdown(["- top", "  - nested", "- second"].join("\n"));
  assert.ok(html.includes("<ul>"));
  assert.ok(html.includes("<li>nested</li>"));
  assert.equal((html.match(/<ul>/g) || []).length, 2);

  const ordered = renderMarkdown(["1. first", "2. second"].join("\n"));
  assert.ok(ordered.includes("<ol>"));
  assert.equal((ordered.match(/<li>/g) || []).length, 2);
});

test("blockquotes render and can contain other blocks", () => {
  const html = renderMarkdown("> **Note:** something\n> - a bullet");
  assert.ok(html.includes("<blockquote>"));
  assert.ok(html.includes("<strong>Note:</strong>"));
  assert.ok(html.includes("<li>a bullet</li>"));
});

test("horizontal rules render", () => {
  for (const rule of ["---", "***", "___", "- - -"]) {
    assert.ok(renderMarkdown(rule).includes("<hr />"), `failed for ${rule}`);
  }
});

// --- inline ---------------------------------------------------------------

test("inline emphasis, code and strikethrough", () => {
  assert.ok(renderMarkdown("**bold**").includes("<strong>bold</strong>"));
  assert.ok(renderMarkdown("*ital*").includes("<em>ital</em>"));
  assert.ok(renderMarkdown("***both***").includes("<strong><em>both</em></strong>"));
  assert.ok(renderMarkdown("~~gone~~").includes("<del>gone</del>"));
  assert.ok(renderMarkdown("use `pytest -q`").includes("<code>pytest -q</code>"));
});

test("underscores inside identifiers are not italics", () => {
  const html = renderMarkdown("call `max_connections` and snake_case_name here");
  assert.ok(!html.includes("<em>"), "mangled an identifier into italics");
});

test("asterisks inside inline code are left alone", () => {
  const html = renderMarkdown("`a ** b`");
  assert.ok(html.includes("<code>a ** b</code>"));
  assert.ok(!html.includes("<strong>"));
});

// --- robustness -----------------------------------------------------------

test("empty and nullish input do not throw", () => {
  assert.equal(renderMarkdown(""), "");
  assert.equal(renderMarkdown(null), "");
  assert.equal(renderMarkdown(undefined), "");
});

test("CRLF input renders the same as LF", () => {
  assert.equal(renderMarkdown("# A\r\n\r\ntext"), renderMarkdown("# A\n\ntext"));
});

test("a realistic postmortem section renders every block type", () => {
  const doc = [
    "## Incident Summary",
    "",
    "| Field | Value |",
    "| --- | --- |",
    "| Severity | SEV2 |",
    "",
    "### Root Cause",
    "",
    "The reindex job pointed at the **primary**, not the replica.",
    "",
    "- Contributing factor one",
    "  - a nested detail",
    "- Contributing factor two",
    "",
    "```sql",
    "SELECT sku FROM stock_levels WHERE updated_at > $1;",
    "```",
    "",
    "> No alert existed on pool saturation.",
  ].join("\n");

  const html = renderMarkdown(doc);
  for (const needle of ["<h2>", "<h3>", "<table>", "<ul>", "<pre><code>", "<blockquote>", "<strong>primary</strong>"]) {
    assert.ok(html.includes(needle), `missing ${needle}`);
  }
});
