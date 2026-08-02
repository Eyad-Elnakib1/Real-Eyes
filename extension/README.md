# Real Eyes — Chrome Extension (skeleton)

Manifest V3 extension that adds three right-click options:

- **Send Image to Real Eyes** — appears on images (`contexts: ["image"]`).
- **Send Text to Real Eyes** — appears when text is selected (`contexts: ["selection"]`).
- **Send Screenshot to Real Eyes** — appears on the page (`contexts: ["page", ...]`).

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | Extension manifest (MV3). |
| `background.js` | Service worker. Registers context menus and routes clicks. |
| `state.js` | Holds the three variables (`image`, `text`, `screenshot`) in `chrome.storage.local`. Only one is non-null at a time. |
| `sender.js` | Placeholder for the backend call. Wire your API here later. |
| `popup.html` / `popup.js` | Tiny debug popup that shows the current state and a Clear button. |
| `icons/` | Drop `icon16.png`, `icon48.png`, `icon128.png` here. |

## Variable rules

`state.js` enforces the spec: every setter writes a fresh object starting from `EMPTY_STATE`, so setting one field automatically nulls the other two.

```js
// image set => text=null, screenshot=null
// text  set => image=null, screenshot=null
// shot  set => image=null, text=null
```

`type` is `"image" | "text" | "screenshot" | null` for easy switching later.

## Loading the extension

1. Add three PNG icons under `icons/` (any size placeholder works while developing).
2. Open `chrome://extensions`, enable **Developer mode**, click **Load unpacked**, and select the `real-eyes/` folder.
3. Right-click an image / selected text / the page to test each menu.
4. Click the toolbar icon to open the popup and inspect the stored variables. Also check the service-worker console (`chrome://extensions` → Real Eyes → "service worker") for `[Real Eyes] …` logs.

## Connecting your backend later

Edit only `sender.js`. The TODO block already has a `fetch` template using the current state shape. Then call `sendCurrentToBackend()` from `background.js` (e.g. right after each `setImage/setText/setScreenshot`) or from the popup.
