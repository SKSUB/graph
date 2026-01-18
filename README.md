# AI Graph

This repository contains an interactive AI concept graph intended to be published with GitHub Pages.

How to publish (quick):

- Ensure this repository is pushed to GitHub (`origin` set to your repo).
- Serve from the `main` branch and the `docs/` folder: GitHub → Settings → Pages → Source: `main` / `docs`.
- After a minute, open: https://SKSUB.github.io/graph/  (or replace `SKSUB`/`graph` with your account/repo)

Files:
- `docs/graph.html` — interactive graph (uses Cytoscape + dagre)
- `docs/index.html` — redirect to `graph.html`

Embedding in Notion / Freeform:
- Host the site (as above) and paste the URL into Notion's Embed block.
- For Freeform, you can embed the URL or export a PNG from the page and drop it into Freeform.
