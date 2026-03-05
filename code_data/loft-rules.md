---
source_file: loft-rules.js
category: geometry
type: annotated_code
use_case: ruled loft fused with a cylinder then shelled — exploring loft-Boolean-shell workflow and known kernel edge cases
related: loft-examples.md, loft-pipe.md, loft-ruled.md, loft-ruled_v2.md, loft-ruled_v3.md
---
# Loft Rules — Ruled Loft with Cylinder Fuse and Shell

## Description
Creates a ruled loft between two rectangles with a circle at mid-height, fuses it with a cylinder, then applies a shell to open the top face. Primarily a workflow experiment documenting known kernel edge cases: the cylinder radius must be large enough (≥6) relative to the loft or the kernel returns null or errors. Useful as a reference for the loft→fuse→shell pipeline and its limitations.

## Keywords
loft, loftWith, ruled, shell, fuse, sketchCircle, sketchRectangle, inPlane, cylinder, Boolean, kernel-error, edge-case, replicad, surface-modeling, shell-workflow, 3d-printing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| rect1 | 5×10 | mm | Base rectangle profile |
| circle | r=8, origin=10 | mm | Mid-height circular profile |
| rect2 | 5×10, origin=20 | mm | Top rectangle profile |
| loft mode | ruled: true | — | Ruled (straight-line generator) loft |
| cylinder radius | 6 | mm | Fused cylinder (must be ≥6 to avoid kernel error) |
| cylinder height | 20 | mm | Height of fused cylinder |
| shell thickness | -0.5 | mm | Inward shell offset on top face |
| shell face | inPlane("XY",[0,0,20]) | — | Top face removed to open the shell |

## Code
```javascript
const main = ({ sketchCircle, sketchRectangle }) => {
  
  let loft = sketchRectangle(5, 10).loftWith([
    sketchCircle(8, { origin: 10 }),
    sketchRectangle(5, 10, { origin: 20 }),
  ],{ruled:true});

  let cylinder = sketchCircle(6).extrude(20)
  loft = loft.fuse(cylinder)

  loft = loft.shell(-0.5,(f) => f.inPlane("XY",[0,0,20]))

  return loft
};
```
