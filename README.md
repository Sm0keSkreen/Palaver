# Palaver

An offline AI chat app. Every model runs entirely on-device in your browser
via [transformers.js](https://huggingface.co/docs/transformers.js). Nothing
you type is ever sent anywhere.

| Model | Size | Notes |
|---|---|---|
| Gemma 270M | ~320MB | Google, fastest replies |
| Gemma 1B | ~860MB | Google, good middle ground |
| Gemma 4 E2B | ~3.4GB | Google, newest architecture, multimodal |

Model weights download once (browser-cached after that) and generation runs
in a Web Worker so the page never freezes.

## Try it

Open **[bootloader.html](https://cdn.jsdelivr.net/gh/Sm0keSkreen/Palaver@main/bootloader.html)**.
It always loads the current version of the app straight from this repo via
jsDelivr's GitHub CDN, no build step or install required.

## How it works

`palaver-app.html` is a single self-contained file (HTML/CSS/JS all inline).
jsDelivr serves `.html` files from GitHub as `text/plain`, so `bootloader.html`
fetches it as text and renders it via an iframe's `srcdoc`. It resolves the
current commit SHA through GitHub's API first, so it always gets the latest
push instead of a possibly-stale CDN edge cache of the `@main` branch alias.

## Rebuilding locally

```
node build.mjs
```

assembles `shell.html` + `style.css` + `chat.js` into `palaver-app.html`.
