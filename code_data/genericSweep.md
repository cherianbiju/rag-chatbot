---
source_file: genericSweep.js
category: replicad_example
type: annotated_code
use_case: helix sweep, coil, genericSweep API, makeHelix reference
related: extrude-examples.md, bezier-extrude.md
---

# Generic Sweep (Helix Coil)

## Description
Demonstrates sweeping a 1×1mm rectangle along a helix path using makeHelix and genericSweep. Creates two swept helical shapes rotated 180° from each other (forming a double-helix style pair), plus a third shape using the .sweepSketch convenience method directly on the helix. Reference for wire-based sweep operations.

## Keywords
genericSweep, makeHelix, assembleWire, drawRectangle, sweepSketch, sketchOnPlane, rotate, helix, coil, spring, wire sweep, forceProfileSpineOthogonality, XZ plane, path1.wires

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| wireSize | 5 | mm | Wire cross-section size |
| wireGap | 0.5 | mm | Gap between wire turns |
| innerRadius | 10 | mm | Helix center radius |
| vertTurns | 1 | - | Number of vertical turns |
| pitch | 11 | mm | Helix pitch (2×(wireSize+wireGap)) |
| height | 11 | mm | Total helix height |
| profile offset | 10 | mm | Profile translated 10mm from axis |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| makeHelix(pitch, height, radius) | Creates a helix wire path |
| drawRectangle(w,h) | Creates 1×1 rectangular profile |
| .translate([x,y,z]) | Offsets profile 10mm from helix axis |
| .sketchOnPlane("XZ") | Places profile on XZ plane |
| assembleWire(path1.wires) | Assembles helix wire segments into single wire |
| genericSweep(profileWire, pathWire, options, solid) | Sweeps profile along wire path |
| .rotate(180,[0,0,0],[0,0,1]) | Rotates second coil 180° for double-helix |
| path1.sweepSketch(fn) | Alternative: sweeps sketch directly on helix |

## Code
```javascript
const {Sketcher,genericSweep,makeHelix,drawRectangle,assembleWire} = replicad;
const wireSize=5, wireGap=0.5, innerRadius=10, vertTurns=1;
const main = () => {
  const shapes = [];
  const path1 = makeHelix(2*(wireSize+wireGap), 2*vertTurns*(wireSize+wireGap), innerRadius);
  const outline = drawRectangle(1,1).translate([10,0]).sketchOnPlane('XZ');
  const swp1 = genericSweep(
    outline.wire, assembleWire(path1.wires),
    {forceProfileSpineOthogonality:false}, false);
  shapes.push(swp1);
  const swp2 = genericSweep(
    outline.wire, assembleWire(path1.wires),
    {forceProfileSpineOthogonality:false}, false)
    .rotate(180,[0,0,0],[0,0,1]);
  shapes.push(swp2);
  const swp3 = path1.sweepSketch((plane,origin)=>sketchRectangle(1,1,{plane,origin}));
  shapes.push(swp3);
  return shapes;
};
```
