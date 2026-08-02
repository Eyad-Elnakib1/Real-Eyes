// loading.js
// Runs inside the tab opened for the report. The tab itself drives the
// fetch + redirect, so the service worker can suspend without affecting us.

const API_BASE = "http://127.0.0.1:5001";

function showError(msg) {
  const ring = document.getElementById("ring");
  const status = document.getElementById("status");
  if (ring) ring.style.display = "none";
  if (status) {
    status.classList.add("err");
    status.textContent = "Real Eyes error: " + msg;
  }
}

async function pollTask(taskId) {
  const status = document.getElementById("status");
  while (true) {
    const r = await fetch(`${API_BASE}/verify-progress/${taskId}`);
    const d = await r.json().catch(() => ({ ok: false, error: "bad json" }));
    if (!d.ok) throw new Error(d.error || "progress error");
    if (d.progress?.message) status.textContent = d.progress.message;
    if (d.done) {
      if (d.error) throw new Error(d.error);
      return d.result;
    }
    await new Promise(res => setTimeout(res, 500));
  }
}

async function main() {
  const params = new URLSearchParams(location.search);
  const isScreenshot = params.has("ss");
  const isText       = params.has("txt");
  const imageUrl = params.get("img");

  let endpoint, body, mode;
  if (isScreenshot) {
    const { realEyesState } = await chrome.storage.local.get("realEyesState");
    const dataUrl = realEyesState?.screenshot;
    if (!dataUrl) { showError("no screenshot in storage"); return; }
    document.getElementById("status").textContent = "Extracting image and text…";
    endpoint = `${API_BASE}/process-screenshot`;
    body = { dataUrl };
    mode = "ss";
  } else if (isText) {
    const { realEyesState } = await chrome.storage.local.get("realEyesState");
    const txt = realEyesState?.text;
    if (!txt) { showError("no text in storage"); return; }
    document.getElementById("status").textContent = "Preparing claim…";
    endpoint = `${API_BASE}/verify-text-start`;
    body = { text: txt };
    mode = "txt";
  } else {
    if (!imageUrl) { showError("no image URL"); return; }
    endpoint = `${API_BASE}/predict`;
    body = { imageUrl };
    mode = "img";
  }

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    let result = await res.json().catch(() => ({ ok: false, error: "bad json" }));

    // Text mode is async: /verify-text-start returns either an immediate
    // no-claim answer or a task_id we poll for live progress.
    if (mode === "txt" && result?.ok && !result.done && result.task_id) {
      result = await pollTask(result.task_id);
    }

    // Keep popup in sync.
    try {
      await chrome.storage.local.set({ realEyesLastResult: result });
    } catch (e) { /* fine — popup just won't update */ }

    if (!result?.ok) {
      showError(result?.error || "unknown error");
      return;
    }

    // Schedule the auto-close alarm NOW that the report is ready, so a slow
    // inference can never cause us to be killed mid-flight.
    try {
      const myTab = await chrome.tabs.getCurrent();
      if (myTab && typeof myTab.id === "number") {
        await chrome.alarms.create("realEyesCloseTab_" + myTab.id, {
          delayInMinutes: 1
        });
      }
    } catch (e) { /* alarms permission missing? non-fatal */ }

    let reportUrl;
    if (mode === "txt") {
      await chrome.storage.local.set({ realEyesTextResult: result });
      reportUrl = chrome.runtime.getURL("verdict.html");
    } else if (mode === "ss") {
      const regions = Array.isArray(result.regions) ? result.regions : [];
      const textResults = Array.isArray(result.text_results) ? result.text_results : [];
      if (regions.length === 0 && textResults.length === 0) {
        showError("no image region or text detected in screenshot");
        return;
      }
      await chrome.storage.local.set({
        realEyesRegions: regions,
        realEyesTextClaims: textResults,
      });
      reportUrl = chrome.runtime.getURL("report.html") + "?multi=1";
    } else if (result.seg_path) {
      // Image flow + segmentation: show both.
      reportUrl = chrome.runtime.getURL("report.html") +
        "?cls=" + encodeURIComponent(result.out_path) +
        "&seg=" + encodeURIComponent(result.seg_path);
    } else if (result.out_path) {
      reportUrl = `${API_BASE}/report?path=${encodeURIComponent(result.out_path)}` +
                  `&t=${Date.now()}`;
    } else {
      showError("no output path returned");
      return;
    }
    location.replace(reportUrl);
  } catch (err) {
    showError(err?.message || String(err));
  }
}

main();
