// sender.js
// Sends the current state to the local Python server that wraps
// run_inference() from test_classification.py.

import { getState, clearState } from "./state.js";

const API_BASE = "http://127.0.0.1:5001";

export async function sendImageToModel(imageUrl) {
  if (!imageUrl) return { ok: false, error: "no image" };

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageUrl })
  });

  const json = await res.json().catch(() => ({ ok: false, error: "bad json" }));
  console.log("[Real Eyes] model response:", json);
  return json;
}

export async function sendCurrentToBackend() {
  const state = await getState();
  if (state.type === "image") return sendImageToModel(state.image);
  // text + screenshot are not wired to a model yet.
  console.log("[Real Eyes] no sender for type:", state.type);
  return { ok: false, error: "unsupported type" };
}

export { clearState };
