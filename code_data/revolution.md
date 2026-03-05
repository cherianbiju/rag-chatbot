---
source_file: revolution.js
category: geometry
type: annotated_code
use_case: demonstrating the revolution() function to rotate a face through a partial angle around an axis — distinct from revolve()
related: multipleloftedsketches.md, occ-bottle.md, plunge_example.md
---
# Revolution — Partial Angle Face Rotation

## Description
Demonstrates the `revolution()` function, which sweeps a 2D face through a specified angle around a given axis to produce a swept solid — analogous to a partial revolve. Distinct from `revolve()` which always produces a full 360° solid. The example sweeps a rectangular profile face 90° around the Z axis, and displays it alongside a reference sphere at the origin.

## Keywords
revolution, partial-revolve, face, sweep-angle, draw, makeSphere, hLine, vLine, sketchOnPlane, XZ-plane, axis, replicad, 3d-printing, geometry, swept-solid

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| sphere radius | 10 | mm | Reference sphere at origin |
| profile | 50×30 rectangle | mm | Rectangle drawn on XZ plane |
| revolution origin | [0, 0, 0] | mm | Axis passes through origin |
| revolution axis | [0, 0, 1] (Z) | — | Rotation around Z axis |
| revolution angle | 90 | ° | Sweep angle (partial, not full 360°) |

## Code
```javascript
const {draw, makeSphere, revolution} = replicad;

function main()
{
let ball = makeSphere(10)
let profile = draw().hLine(50).vLine(30).hLine(-50).close()
.sketchOnPlane("XZ")
let bodyRevolution = revolution(profile.face(),[0,0,0],[0,0,1],90)

return [ball, bodyRevolution]
}
```

**Note:** `revolution_example.js` is identical to this file.
