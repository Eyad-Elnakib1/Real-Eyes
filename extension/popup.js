// popup.js
// Read-only view of the current state, plus a Clear button.
// Useful for debugging the three variables while building.

import { getState, clearState } from "./state.js";

function truncate(value, max = 80) {
  if (value == null) return "null";
  const s = String(value);
  return s.length > max ? s.slice(0, max) + "…" : s;
}

async function render() {
  const state = await getState();
  const { realEyesLastResult } = await chrome.storage.local.get("realEyesLastResult");
  const modelEl = document.getElementById("model");
  if (realEyesLastResult?.ok) {
    const probs = realEyesLastResult.probs || [];
    const pct = probs.map(p => (p * 100).toFixed(1) + "%").join(" / ");
    modelEl.textContent = `${realEyesLastResult.label} (${pct})`;
  } else if (realEyesLastResult?.error) {
    modelEl.textContent = `error: ${realEyesLastResult.error}`;
  } else {
    modelEl.textContent = "—";
  }

  document.getElementById("type").textContent = state.type || "—";
  document.getElementById("image").textContent = truncate(state.image);
  document.getElementById("text").textContent = truncate(state.text);
  document.getElementById("screenshot").textContent =
    state.screenshot ? `[dataURL ${state.screenshot.length} chars]` : "null";

  const preview = document.getElementById("preview");
  const src = state.image || state.screenshot;
  if (src) {
    preview.src = src;
    preview.style.display = "block";
  } else {
    preview.removeAttribute("src");
    preview.style.display = "none";
  }
}

document.getElementById("clear").addEventListener("click", async () => {
  await clearState();
  render();
});

chrome.storage.onChanged.addListener((_changes, area) => {
  if (area === "local") render();
});

render();
