// background.js
// Service worker: registers context menus, injects the area-selection
// overlay for screenshots, crops the captured tab to the selected area,
// and routes everything into state.js.

import { setImage, setText, setScreenshot } from "./state.js";

const MENU_IDS = {
  image: "real-eyes-send-image",
  text: "real-eyes-send-text",
  screenshot: "real-eyes-send-screenshot"
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_IDS.image,
      title: "Send Image to Real Eyes",
      contexts: ["image"]
    });

    chrome.contextMenus.create({
      id: MENU_IDS.text,
      title: "Send Text to Real Eyes",
      contexts: ["selection"]
    });

    chrome.contextMenus.create({
      id: MENU_IDS.screenshot,
      title: "Send Screenshot to Real Eyes",
      contexts: ["page"]
    });
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  try {
    switch (info.menuItemId) {
      case MENU_IDS.image: {
        const imageUrl = info.srcUrl || null;
        const state = await setImage(imageUrl);
        console.log("[Real Eyes] image stored:", state);

        if (!imageUrl) break;

        // Open the loading page with the image URL embedded. The page itself
        // calls the model and redirects to the report when done, so the
        // service worker can suspend without breaking the flow.
        const url =
          chrome.runtime.getURL("loading.html") +
          "?img=" + encodeURIComponent(imageUrl);
        const reportTab = await chrome.tabs.create({ url, active: true });
        console.log("[Real Eyes] opened report tab:", reportTab.id);
        // NOTE: no close alarm here. loading.js sets it AFTER the report
        // is shown so a slow inference can't be killed mid-flight.
        break;
      }

      case MENU_IDS.text: {
        const selected = (info.selectionText || "").trim();
        const state = await setText(selected);
        console.log("[Real Eyes] text stored:", state);
        if (!selected) break;
        const url = chrome.runtime.getURL("loading.html") + "?txt=1";
        const tab = await chrome.tabs.create({ url, active: true });
        console.log("[Real Eyes] opened text-verify tab:", tab.id);
        break;
      }

      case MENU_IDS.screenshot: {
        if (!tab?.id) return;
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ["area-select.js"]
        });
        // The content script will message us back with the rect.
        break;
      }
    }
  } catch (err) {
    console.error("[Real Eyes] context menu handler failed:", err);
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "REAL_EYES_CAPTURE_AREA") return;

  const tab = sender.tab;
  if (!tab || tab.windowId === undefined) {
    sendResponse({ ok: false, reason: "no-tab" });
    return;
  }

  captureArea(tab.windowId, message.rect, message.dpr)
    .then((dataUrl) => setScreenshot(dataUrl))
    .then(async (state) => {
      console.log("[Real Eyes] screenshot stored (length):", state.screenshot?.length);
      // Open the loading page in screenshot mode. The page itself POSTs the
      // dataUrl to /process-screenshot so the worker can suspend freely.
      const url = chrome.runtime.getURL("loading.html") + "?ss=1";
      await chrome.tabs.create({ url, active: true });
      sendResponse({ ok: true });
    })
    .catch((err) => {
      console.error("[Real Eyes] screenshot capture failed:", err);
      sendResponse({ ok: false, reason: String(err) });
    });

  return true; // async sendResponse
});

async function captureArea(windowId, rect, dpr) {
  const fullDataUrl = await chrome.tabs.captureVisibleTab(windowId, {
    format: "png"
  });

  const blob = await (await fetch(fullDataUrl)).blob();
  const bitmap = await createImageBitmap(blob);

  // The captured image is in device pixels; the rect is in CSS pixels.
  const sx = Math.round(rect.x * dpr);
  const sy = Math.round(rect.y * dpr);
  const sw = Math.round(rect.w * dpr);
  const sh = Math.round(rect.h * dpr);

  const canvas = new OffscreenCanvas(sw, sh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
  bitmap.close();

  const croppedBlob = await canvas.convertToBlob({ type: "image/png" });
  return await blobToDataUrl(croppedBlob);
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

// ------- Report tab: open + auto-close after 1 minute ----------------------

const CLOSE_ALARM_PREFIX = "realEyesCloseTab_";
const CLOSE_AFTER_MINUTES = 1;

chrome.alarms.onAlarm.addListener((alarm) => {
  if (!alarm.name.startsWith(CLOSE_ALARM_PREFIX)) return;
  const tabId = parseInt(alarm.name.slice(CLOSE_ALARM_PREFIX.length), 10);
  if (Number.isNaN(tabId)) return;
  chrome.tabs.remove(tabId).catch((err) => {
    // Tab already closed by the user — fine.
    console.log("[Real Eyes] could not close tab", tabId, err?.message);
  });
});
