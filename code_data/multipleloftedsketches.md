---
source_file: multipleloftedsketches.js
category: geometry
type: annotated_code
use_case: revolve experiment — Sketcher half-profile revolved around Z axis to produce a button/cap shape
related: loft-examples.md, loft-pipe.md, occ-bottle.md
---
# Multiple Lofted Sketches — Revolve Experiment

## Description
A minimal experiment file containing several commented-out loft variants and one active revolve. The live code creates a rounded cap or button shape by sketching a half-profile (horizontal lines with a half-ellipse bump) on the XZ plane and revolving it around the default Z axis. Useful as a quick reference for the `revolve()` method and for comparing loft vs. revolve workflows.

## Keywords
revolve, Sketcher, halfEllipse, hLine, XZ-plane, loft, loftWith, sketchCircle, sketchRectangle, endPoint, replicad, revolve-vs-loft, cap-shape, button, simple-example

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| hLine length | 25 | mm | Half-width of the cap base |
| halfEllipse x | 0 | mm | No X translation for half-ellipse |
| halfEllipse y | 40 | mm | Height of the ellipse dome |
| halfEllipse radius | 5 | mm | Minor radius of the half-ellipse |
| revolve axis | Z (default) | — | Rotation axis for the revolve |

## Code
```javascript
const main = ({ Sketcher }) => {
    return new Sketcher("XZ")
      .hLine(25)
      .halfEllipse(0, 40, 5)
      .hLine(-25)
      .close()
      .revolve();
};
```
