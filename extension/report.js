// report.js — renders text claim verdicts (top) and one or more region cards.
const API_BASE = "http://127.0.0.1:5001";
const params = new URLSearchParams(location.search);
const root = document.getElementById("root");

const TEXT_THEME = {
  SUPPORTS:          { accent: "#1dcea0", bg: "rgba(29,206,160,0.10)", icon: "✓", long: "Likely TRUE" },
  REFUTES:           { accent: "#ef4444", bg: "rgba(239,68,68,0.10)",  icon: "✕", long: "Likely FALSE" },
  "NOT ENOUGH INFO": { accent: "#f59e0b", bg: "rgba(245,158,11,0.10)", icon: "?", long: "Inconclusive" },
};

function reportImg(path) {
  if (!path) return null;
  const img = document.createElement("img");
  img.src = `${API_BASE}/report?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  return img;
}

function nliClass(label) {
  if (label === "SUPPORTS") return "supports";
  if (label === "REFUTES")  return "refutes";
  return "nei";
}

function renderTextClaim(tc, idx) {
  const r = tc.result;

  // No-claim card — LLM cleaning produced nothing worth fact-checking.
  if (tc.no_claim) {
    const card = document.createElement("div");
    card.className = "text-card";
    card.setAttribute("style", "--accent:#8aa1c8; --accent-bg:transparent;");
    card.innerHTML = `
      <div class="row">
        <div class="badge">— NO CLAIM</div>
        <div class="source-pill">${tc.source.toUpperCase()} TEXT</div>
      </div>
      <div class="statement" style="font-size:15px;color:#ecf2ff;">There are no claims.</div>
    `;
    return card;
  }

  if (!r || !r.ok) {
    // Couldn't verify; just show the text + reason.
    const card = document.createElement("div");
    card.className = "text-card";
    card.setAttribute("style", "--accent:#8aa1c8; --accent-bg:transparent;");
    card.innerHTML = `
      <div class="row">
        <div class="badge">— UNVERIFIED</div>
        <div class="source-pill">${tc.source.toUpperCase()} TEXT</div>
      </div>
      <div class="claim"><small>EXTRACTED</small>${escapeHtml(tc.cleaned || tc.raw)}</div>
      <div class="statement">${escapeHtml((r && r.error) || "No verification result")}</div>
    `;
    return card;
  }

  const verdict = r.final_prediction || "NOT ENOUGH INFO";
  const theme = TEXT_THEME[verdict] || TEXT_THEME["NOT ENOUGH INFO"];
  const confPct = (r.confidence * 100).toFixed(1);
  const verif = r.verification || {};

  const card = document.createElement("div");
  card.className = "text-card";
  card.setAttribute("style", `--accent:${theme.accent}; --accent-bg:${theme.bg};`);

  // header row
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <div class="badge"><span>${theme.icon}</span> ${theme.long}</div>
    <div class="conf">${confPct}% <small>CONFIDENCE</small></div>
    <div class="verified-pill ${verif.verified ? "" : "unverified"}">
      ${verif.verified ? "✓ VERIFIED" : "unverified"}
    </div>
    <div class="source-pill">${tc.source.toUpperCase()} TEXT</div>
  `;
  card.appendChild(row);

  // claim
  const claim = r.claim || tc.english || tc.cleaned || tc.raw;
  const claimEl = document.createElement("div");
  claimEl.className = "claim";
  claimEl.innerHTML = `<small>CLAIM</small>${escapeHtml(claim)}`;
  card.appendChild(claimEl);

  // statement
  const stmt = verif.statement || r.note || `Verdict: ${verdict}`;
  const stmtEl = document.createElement("div");
  stmtEl.className = "statement";
  stmtEl.textContent = stmt;
  card.appendChild(stmtEl);

  // LLM explanation (SUPPORTS / REFUTES verdicts)
  if (r.explanation) {
    const exp = document.createElement("div");
    exp.setAttribute("style",
      `margin-bottom:14px;padding:12px 14px;border-radius:10px;` +
      `border:1px solid ${theme.accent};background:${theme.bg};` +
      `color:#ecf2ff;font-size:13px;line-height:1.5;`);
    exp.innerHTML =
      `<div style="font-size:10px;letter-spacing:2px;color:${theme.accent};margin-bottom:4px;">WHY</div>` +
      escapeHtml(r.explanation);
    card.appendChild(exp);
  }

  // bar
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = `<div class="fill" style="width:${confPct}%"></div>`;
  card.appendChild(bar);

  // top 3 sources inline
  const ev = (r.evidence || []).slice(0, 3);
  for (const s of ev) {
    const probs = s.probabilities || {};
    const sPct = ((probs["SUPPORTS"] || 0) * 100).toFixed(0);
    const rPct = ((probs["REFUTES"]  || 0) * 100).toFixed(0);
    const nPct = ((probs["NOT ENOUGH INFO"] || 0) * 100).toFixed(0);
    const line = document.createElement("div");
    line.className = "src-line";
    line.innerHTML = `
      <span class="nli-chip ${nliClass(s.nli_label)}">${s.nli_label || "?"}</span>
      <a href="${escapeAttr(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.domain || "(unknown)")}</a>
      <span style="color:#8aa1c8;margin-left:auto;">
        S:${sPct}% · R:${rPct}% · N:${nPct}%
      </span>
    `;
    card.appendChild(line);
  }
  return card;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/`/g, "&#96;"); }

function renderRegion(region, index) {
  const wrap = document.createElement("section");
  wrap.className = "region";

  const title = document.createElement("h3");
  title.className = "region-title";
  const label = region.label || "?";
  const real = region.probs ? (region.probs[0] * 100).toFixed(1) : "?";
  const fake = region.probs ? (region.probs[1] * 100).toFixed(1) : "?";
  const isFake = label.toUpperCase() === "FAKE";
  title.innerHTML =
    `Region ${region.id ?? index + 1}` +
    (region.type ? ` · ${region.type}` : "") +
    ` — <span class="label${isFake ? " fake" : ""}">${label}</span>` +
    ` &nbsp; real ${real}% · fake ${fake}%`;
  wrap.appendChild(title);

  const grid = document.createElement("div");
  grid.className = "grid";

  if (region.out_path) {
    const c = document.createElement("div");
    c.className = "card";
    const h = document.createElement("h2");
    h.textContent = "Classification";
    c.appendChild(h);
    c.appendChild(reportImg(region.out_path));
    grid.appendChild(c);
  }
  if (region.seg_path) {
    const c = document.createElement("div");
    c.className = "card";
    const h = document.createElement("h2");
    h.textContent = "Edited Areas (segmentation)";
    c.appendChild(h);
    c.appendChild(reportImg(region.seg_path));
    grid.appendChild(c);
  }

  wrap.appendChild(grid);
  return wrap;
}

function renderSingle({ cls, seg }) {
  // Legacy image-flow path: one region with cls + optional seg.
  return renderRegion({ id: 1, out_path: cls, seg_path: seg }, 0);
}

async function main() {
  if (params.has("multi")) {
    const { realEyesRegions, realEyesTextClaims } =
      await chrome.storage.local.get(["realEyesRegions", "realEyesTextClaims"]);
    const regions = Array.isArray(realEyesRegions) ? realEyesRegions : [];
    const textClaims = Array.isArray(realEyesTextClaims) ? realEyesTextClaims : [];

    if (textClaims.length > 0) {
      const sec = document.createElement("section");
      sec.className = "text-section";
      const h = document.createElement("h2");
      h.className = "region-title";
      h.textContent = "Text claims";
      sec.appendChild(h);
      textClaims.forEach((tc, i) => sec.appendChild(renderTextClaim(tc, i)));
      root.appendChild(sec);
    }

    if (regions.length > 0) {
      regions.forEach((r, i) => root.appendChild(renderRegion(r, i)));
    } else if (textClaims.length === 0) {
      root.textContent = "Nothing to display.";
    }
  } else {
    root.appendChild(renderSingle({
      cls: params.get("cls"),
      seg: params.get("seg"),
    }));
  }
}

main();
