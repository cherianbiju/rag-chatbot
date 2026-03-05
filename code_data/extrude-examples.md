---
source_file: extrude-examples.js
category: replicad_example
type: annotated_code
use_case: extrusion options reference, twist taper origin comparison
related: extrude.md, bezier-extrude.md
---

# Extrude Examples

## Description
Five towers extruded from the same 20×20mm square base profile showing different combinations of extrusionProfile (linear, s-curve), twistAngle, and origin. Each tower is colored and positioned differently for side-by-side comparison. The most complete reference for replicad extrusion options.

## Keywords
extrude, twistAngle, extrusionProfile, linear, s-curve, endFactor, origin, draw, clone, translate, sketchOnPlane, taper, twist, comparison, reference, tower

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| base size | 20×20 | mm | Square base profile centered at origin |
| extrude height | 100 | mm | Height of all towers |
| twistAngle | 45 | deg | Rotation twist applied over full extrusion |
| endFactor | 0.7 | - | Top is 70% scale of base |
| tower3/4 translate | [30,30] | mm | Offset to show twist origin effect |
| tower5 translate | [60,60] | mm | Offset for plain tower |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw([x,y]) | Starts 2D drawing at point |
| .hLine() .vLine() | Draws square profile |
| .close() | Closes profile |
| .clone() | Duplicates profile for reuse |
| .translate(x,y) | Moves profile before sketching |
| .sketchOnPlane("XY") | Places on XY plane |
| .extrude(h, {extrusionProfile, twistAngle, origin}) | Extrudes with combined options |
| profile:"linear" | Linear taper from base to endFactor×base |
| profile:"s-curve" | Smooth s-curve taper |
| origin:[x,y] | Sets the pivot point for twist |

## Code
```javascript
const {draw} = replicad;
function main() {
  let baseProfile = draw([-10,-10]).hLine(20).vLine(20).hLine(-20).close();
  // linear taper + twist
  let tower = baseProfile.clone().sketchOnPlane("XY")
    .extrude(100,{extrusionProfile:{profile:"linear",endFactor:0.7},twistAngle:45});
  // s-curve taper + twist
  let tower2 = baseProfile.clone().sketchOnPlane("XY")
    .extrude(100,{extrusionProfile:{profile:"s-curve",endFactor:0.7},twistAngle:45});
  // twist with custom origin (twist pivots around [30,30])
  let tower3 = baseProfile.clone().translate([30,30]).sketchOnPlane("XY")
    .extrude(100,{origin:[30,30],twistAngle:45});
  // twist without origin (twist pivots around [0,0])
  let tower4 = baseProfile.clone().translate([30,30]).sketchOnPlane("XY")
    .extrude(100,{twistAngle:45});
  // plain extrude, no twist or taper
  let tower5 = baseProfile.clone().translate(60,60).sketchOnPlane("XY").extrude(100);
  return [tower, {shape:tower2,color:"red"}, tower3, {shape:tower4,color:"grey"}, tower5];
}
```
