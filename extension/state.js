// state.js
// Single source of truth for the currently-selected payload.
// Only ONE of { image, text, screenshot } may be non-null at a time.
//
// Two ways to read the values:
//   1) Direct variables  -> import { image, text, screenshot, type } from "./state.js"
//      (live module bindings, always reflect the latest value)
//   2) Async snapshot    -> const s = await getState();
//      (use this from popup/other contexts, or right after worker wake-up)
//
// Storage is still used so values survive service-worker restarts and are
// visible to the popup.

const STORAGE_KEY = "realEyesState";

const EMPTY_STATE = {
  image: null,
  text: null,
  screenshot: null,
  type: null,        // "image" | "text" | "screenshot" | null
  updatedAt: null
};

// ---- Live variables (read these from background.js / sender.js) ----------
export let image = null;
export let text = null;
export let screenshot = null;
export let type = null;
export let updatedAt = null;

function applyToLiveVars(state) {
  image = state.image;
  text = state.text;
  screenshot = state.screenshot;
  type = state.type;
  updatedAt = state.updatedAt;
}

// Rehydrate live variables when the service worker boots.
chrome.storage.local.get(STORAGE_KEY).then((result) => {
  applyToLiveVars(result[STORAGE_KEY] || EMPTY_STATE);
});

// Keep live variables in sync if any other context (popup, etc.) writes.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !changes[STORAGE_KEY]) return;
  applyToLiveVars(changes[STORAGE_KEY].newValue || EMPTY_STATE);
});

// ---- Snapshot accessor ---------------------------------------------------
export async function getState() {
  const result = await chrome.storage.local.get(STORAGE_KEY);
  return result[STORAGE_KEY] || { ...EMPTY_STATE };
}

// ---- Setters (each one nulls the other two) ------------------------------
async function commit(next) {
  applyToLiveVars(next);
  await chrome.storage.local.set({ [STORAGE_KEY]: next });
  return next;
}

export async function clearState() {
  return commit({ ...EMPTY_STATE });
}

export async function setImage(imageUrl) {
  return commit({
    ...EMPTY_STATE,
    image: imageUrl,
    type: "image",
    updatedAt: Date.now()
  });
}

export async function setText(value) {
  return commit({
    ...EMPTY_STATE,
    text: value,
    type: "text",
    updatedAt: Date.now()
  });
}

export async function setScreenshot(dataUrl) {
  return commit({
    ...EMPTY_STATE,
    screenshot: dataUrl,
    type: "screenshot",
    updatedAt: Date.now()
  });
}
