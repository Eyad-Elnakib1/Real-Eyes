// area-select.js
// Injected into the active tab when the user picks "Send Screenshot to Real Eyes".
// Shows a full-page overlay, lets the user drag a rectangle, then asks the
// background worker to capture the visible tab and crop to that rectangle.

(() => {
  if (window.__realEyesAreaSelectActive) return;
  window.__realEyesAreaSelectActive = true;

  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    cursor: crosshair;
    background: rgba(0, 0, 0, 0.25);
    user-select: none;
  `;

  const box = document.createElement("div");
  box.style.cssText = `
    position: fixed;
    border: 2px dashed #00e5ff;
    background: rgba(0, 229, 255, 0.1);
    pointer-events: none;
    display: none;
  `;

  const hint = document.createElement("div");
  hint.textContent = "Drag to select an area · Esc to cancel";
  hint.style.cssText = `
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.75);
    color: #fff;
    font: 12px system-ui, sans-serif;
    padding: 6px 10px;
    border-radius: 4px;
    pointer-events: none;
  `;

  document.documentElement.appendChild(overlay);
  document.documentElement.appendChild(box);
  document.documentElement.appendChild(hint);

  let startX = 0;
  let startY = 0;
  let dragging = false;

  function cleanup() {
    overlay.remove();
    box.remove();
    hint.remove();
    document.removeEventListener("keydown", onKey, true);
    window.__realEyesAreaSelectActive = false;
  }

  function onKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      cleanup();
    }
  }
  document.addEventListener("keydown", onKey, true);

  overlay.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    box.style.left = startX + "px";
    box.style.top = startY + "px";
    box.style.width = "0px";
    box.style.height = "0px";
    box.style.display = "block";
  });

  overlay.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const x = Math.min(e.clientX, startX);
    const y = Math.min(e.clientY, startY);
    const w = Math.abs(e.clientX - startX);
    const h = Math.abs(e.clientY - startY);
    box.style.left = x + "px";
    box.style.top = y + "px";
    box.style.width = w + "px";
    box.style.height = h + "px";
  });

  overlay.addEventListener("mouseup", (e) => {
    if (!dragging) return;
    dragging = false;

    const rect = {
      x: Math.min(e.clientX, startX),
      y: Math.min(e.clientY, startY),
      w: Math.abs(e.clientX - startX),
      h: Math.abs(e.clientY - startY)
    };

    if (rect.w < 4 || rect.h < 4) {
      cleanup();
      return;
    }

    const dpr = window.devicePixelRatio || 1;

    // Hide overlay BEFORE the background captures the tab, otherwise it
    // would appear in the screenshot. Two RAFs == one painted frame.
    overlay.style.display = "none";
    box.style.display = "none";
    hint.style.display = "none";

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        chrome.runtime.sendMessage(
          { type: "REAL_EYES_CAPTURE_AREA", rect, dpr },
          () => cleanup()
        );
      });
    });
  });
})();
