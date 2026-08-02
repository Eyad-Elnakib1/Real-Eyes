// verdict.js — renders the news-verification result.

const THEME = {
  SUPPORTS: {
    accent: "#1dcea0",
    bg:     "rgba(29,206,160,0.10)",
    glow:   "radial-gradient(800px 300px at 50% -10%, rgba(29,206,160,0.35), transparent 60%)",
    icon:   "✓",
    short:  "SUPPORTS",
    long:   "Likely TRUE",
  },
  REFUTES: {
    accent: "#ef4444",
    bg:     "rgba(239,68,68,0.10)",
    glow:   "radial-gradient(800px 300px at 50% -10%, rgba(239,68,68,0.30), transparent 60%)",
    icon:   "✕",
    short:  "REFUTES",
    long:   "Likely FALSE",
  },
  "NOT ENOUGH INFO": {
    accent: "#f59e0b",
    bg:     "rgba(245,158,11,0.10)",
    glow:   "radial-gradient(800px 300px at 50% -10%, rgba(245,158,11,0.25), transparent 60%)",
    icon:   "?",
    short:  "NOT ENOUGH INFO",
    long:   "Inconclusive",
  },
};

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class")      e.className = v;
    else if (k === "style") e.setAttribute("style", v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null)     e.setAttribute(k, v);
  }
  for (const kid of kids) {
    if (kid == null) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(kid));
  }
  return e;
}

function nliClass(label) {
  if (label === "SUPPORTS")        return "supports";
  if (label === "REFUTES")         return "refutes";
  return "nei";
}

async function main() {
  const root = document.getElementById("root");
  const { realEyesTextResult } = await chrome.storage.local.get("realEyesTextResult");
  const r = realEyesTextResult;

  if (!r || !r.ok) {
    root.appendChild(el("div", { class: "empty" },
      "No verification result available. Right-click selected text and pick “Send Text to Real Eyes”."));
    return;
  }

  // No-claim short-circuit (gate 1 or gate 2 rejected the text).
  if (r.no_claim) {
    const hero = el("section", {
      class: "hero",
      style: "--accent:#8aa1c8; --accent-bg:transparent; --accent-glow:transparent;"
    });
    hero.appendChild(el("div", { class: "verdict-row" },
      el("div", { class: "verdict-badge" },
        el("span", { class: "icon" }, "—"),
        "NO CLAIM"
      )
    ));
    if (r.claim) {
      hero.appendChild(el("div", { class: "claim" },
        el("span", { class: "label" }, "SELECTED TEXT"),
        r.claim
      ));
    }
    hero.appendChild(el("div", {
      class: "statement",
      style: "font-size:16px;color:var(--text);margin-top:18px;"
    }, "There are no claims."));
    root.appendChild(hero);
    return;
  }

  const verdict = r.final_prediction || "NOT ENOUGH INFO";
  const theme = THEME[verdict] || THEME["NOT ENOUGH INFO"];
  const confPct = (r.confidence * 100).toFixed(1);
  const verif = r.verification || {};

  // Hero
  const hero = el("section", {
    class: "hero",
    style: `--accent:${theme.accent}; --accent-bg:${theme.bg}; --accent-glow:${theme.glow};`
  });

  const row = el("div", { class: "verdict-row" },
    el("div", { class: "verdict-badge" },
      el("span", { class: "icon" }, theme.icon),
      theme.long
    ),
    el("div", { class: "conf" },
      el("div", { class: "num" }, `${confPct}%`),
      el("div", { class: "label" }, "CONFIDENCE")
    ),
    el("div", {
      class: "verified-pill" + (verif.verified ? "" : " unverified"),
    }, verif.verified ? "✓ VERIFIED" : "unverified")
  );
  hero.appendChild(row);

  // Claim quote
  if (r.claim) {
    hero.appendChild(el("div", { class: "claim" },
      el("span", { class: "label" }, "CLAIM"),
      r.claim
    ));
  }

  // Statement from verification
  const statementText = verif.statement
    || r.note
    || `Verdict: ${verdict}`;
  hero.appendChild(el("div", { class: "statement" }, statementText));

  // LLM explanation (renders only for SUPPORTS / REFUTES verdicts)
  if (r.explanation) {
    const exp = el("div", {
      style:
        "margin-top:18px;padding:14px 16px;border-radius:10px;" +
        `border:1px solid ${theme.accent};` +
        `background:${theme.bg};` +
        "color:#ecf2ff;font-size:14px;line-height:1.55;"
    },
      el("div", {
        style: `font-size:11px;letter-spacing:2px;color:${theme.accent};margin-bottom:6px;`,
      }, "WHY"),
      r.explanation
    );
    hero.appendChild(exp);
  }

  // Confidence bar
  const barWrap = el("div", { class: "bar-wrap" },
    el("div", { class: "track" },
      el("div", { class: "fill", style: `width:${confPct}%` })
    ),
    el("div", { class: "legend" },
      el("span", {}, "0%"),
      el("span", {}, `${verif.independent_sources || 0} independent sources · ${ (verif.trusted_agree || []).length } trusted`),
      el("span", {}, "100%")
    )
  );
  hero.appendChild(barWrap);
  root.appendChild(hero);

  // Sources
  const evidence = r.evidence || [];
  root.appendChild(el("h2", { class: "section" }, `Sources (${evidence.length})`));
  if (evidence.length === 0) {
    root.appendChild(el("div", { class: "empty" }, "No source evidence to display."));
    return;
  }

  for (const ev of evidence) {
    const cls = nliClass(ev.nli_label);
    const trusted = (ev.credibility || 1) >= 2.0;
    const probs = ev.probabilities || {};
    const sPct = ((probs["SUPPORTS"] || 0) * 100).toFixed(1);
    const rPct = ((probs["REFUTES"]  || 0) * 100).toFixed(1);
    const nPct = ((probs["NOT ENOUGH INFO"] || 0) * 100).toFixed(1);

    const card = el("div", { class: "src" });
    const head = el("div", { class: "src-head" },
      el("a", {
        class: "src-domain",
        href: ev.url || "#",
        target: "_blank",
        rel: "noopener noreferrer",
      }, ev.domain || "(unknown)"),
      el("span", { class: `nli-chip ${cls}` }, ev.nli_label || "?"),
      trusted ? el("span", { class: "trusted" }, `TRUSTED ×${ev.credibility.toFixed(0)}`) : null,
    );
    card.appendChild(head);

    if (ev.title) {
      card.appendChild(el("div", {
        style: "margin-top:6px;color:#c7d2e8;font-size:13px;"
      }, ev.title));
    }

    card.appendChild(el("div", { class: "src-meta" },
      el("span", { class: "pct s" }, `S ${sPct}%`),
      el("span", { class: "pct r" }, `R ${rPct}%`),
      el("span", { class: "pct n" }, `N ${nPct}%`),
      el("span", {}, `retrieval ${ev.retrieval_score.toFixed(3)}`),
    ));

    if (ev.snippet) {
      card.appendChild(el("div", { class: "snippet" }, `"${ev.snippet}…"`));
    }

    root.appendChild(card);
  }
}

main();
