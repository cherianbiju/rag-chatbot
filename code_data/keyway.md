---
source_file: keyway.js
category: mechanical
type: annotated_code
use_case: axle cross-section with keyway slot for torque transmission, demonstrating both cut and fuse approaches
related: lever.md, knob8.md
---
# Keyway Axle Cross-Section

## Description
Creates two versions of a keyed axle cross-section: one where the keyway slot is subtracted from the circular axle (standard internal keyway), and one where the slot is fused to the circle (external key). Both are extruded into 3D solids for comparison of cut vs. fuse Boolean operations on 2D profiles.

## Keywords
keyway, axle, key-slot, drawCircle, drawRectangle, cut, fuse, extrude, sketchOnPlane, XZ-plane, torque-transmission, shaft, mechanical, replicad, Boolean-2D, 3d-printing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| axleRadius | 11 | mm | Radius of the axle circle |
| keySlotHeight | 6 | mm | Height (radial depth) of the keyway rectangle |
| keySlotWidth | 2.50 | mm | Half-width of the keyway slot (full width = 5 mm) |
| axle extrude | 25 | mm | Length of the extruded axle |
| axle2 X offset | 3×axleRadius = 33 | mm | Separation between the two demonstration axles |
| sketch plane | XZ | — | Both cross-sections are sketched on the XZ plane |

## Code
```javascript
const { draw, drawCircle, drawRectangle} = replicad;

const main = () => {
let axleRadius = 11
let keySlotHeight = 6
let keySlotWidth  = 2.50  

let axleHole = drawCircle(axleRadius)
let axleHole2 = drawCircle(axleRadius).translate(3*axleRadius,0)
let keySlot  = drawRectangle(2*keySlotWidth,keySlotHeight)
.translate(-axleRadius,0)
let keySlot2  = drawRectangle(2*keySlotWidth,keySlotHeight)
.translate(-axleRadius,0).translate(3*axleRadius,0)
let axleShape = axleHole.cut(keySlot).sketchOnPlane("XZ")
let axleShape2 = axleHole2.fuse(keySlot2).sketchOnPlane("XZ",10)
let axle = axleShape.extrude(25)
let axle2 = axleShape2.extrude(25)
  return [axle,axle2];
};
```
