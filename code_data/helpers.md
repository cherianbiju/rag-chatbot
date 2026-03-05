---
source_file: helpers.js
category: replicad_example
type: annotated_code
use_case: utility module, iframe URL builder, docs website helper, non-CAD
related: genericSweep.md
---

# Helpers (Docs Utility)

## Description
Utility module for the replicad documentation website. Exports two items: BASE_PATH (the GitHub raw URL for replicad example files) and iframePath() (a function that builds a full replicad Studio share URL for embedding any example in an iframe). Not a CAD model — a pure JavaScript utility for docs site use.

## Keywords
helpers, utility, BASE_PATH, iframePath, encodeURIComponent, export, GitHub raw URL, replicad studio, share URL, docs website, iframe embedding, non-model file

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| BASE_PATH | GitHub raw URL | - | Base URL for replicad example JS files |
| filename | string | - | Name of JS example file to embed |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| encodeURIComponent(str) | URL-encodes the full file path for safe embedding in share URL |

## Code
```javascript
export const BASE_PATH =
  "https://raw.githubusercontent.com/sgenoud/replicad/main/packages/replicad-docs/examples/";
export const iframePath = (filename) => {
  return `https://studio.replicad.xyz/share/${encodeURIComponent(BASE_PATH + filename)}`;
};
```
