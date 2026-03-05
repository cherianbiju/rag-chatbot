---
source_file: plunge-v5-rc.js
category: consumer-product
type: annotated_code
use_case: Plunge watering carafe — replicad port of v5, body-only revolve with all filler/spout operations commented out for debugging
related: plunge_example.md, plunge_improved.md, occ-bottle.md
---
# Plunge v5 RC — Body-Only Revolve (Debug State)

## Description
A replicad port of the Plunge carafe started from the v5 CadShaper model. Only the body revolve is active; all filler, spout, shell, and cutter operations from v5 are commented out as the translation was in progress. Useful as a minimal reference for the Plunge side profile revolved on the XZ plane, and as a starting point showing which v5 operations still needed porting to replicad syntax.

## Keywords
Plunge, watering-carafe, revolve, Sketcher, XZ-plane, body-only, work-in-progress, debug, consumer-product, replicad, Robert-Bronwasser, port, commented-out

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| p1 | [20, 0] | mm | Base outer edge |
| p2 | [30, 5] | mm | Base shoulder start |
| p3 | [30, 8] | mm | Shoulder end |
| p4 | [8, 100] | mm | Top radius 8 mm at height 100 mm |
| p5 | [0, 100] | mm | Top centre closes the profile |
| revolve axis | Z (default) | — | XZ sketch revolved around Z axis |

## Code
```javascript
// Model of the Plunge watering carafe designed by Robert Bronwasser
// Body only — filler, spout, shell and cutter operations still being ported from CadShaper v5

function main({Sketcher})
{

let p0=[0,0]; let p1=[20,0]; let p2=[30,5];
let p3=[30,8]; let p4=[8,100]; let p5=[0,100]

let sideview = new Sketcher("XZ")
.lineTo(p1).lineTo(p2).lineTo(p3).lineTo(p4).lineTo(p5).close()

let body = sideview.revolve()

let shapeArray = [
{shape: body, color:"orange"}, 
]

return shapeArray
}
```
