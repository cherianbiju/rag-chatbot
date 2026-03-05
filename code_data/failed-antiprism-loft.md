---
source_file: failed-antiprism-loft.js
category: replicad_example
type: annotated_code
use_case: multi-section loft, antiprism tower, loftWith between 3 profiles
related: extrude-examples.md, extrude.md
---

# Failed Antiprism Loft

## Description
Attempts to build an antiprism tower by lofting between three profiles at different heights — a square base, a rotated octagonal mid-section (using polarLine), and a rotated triangular top. All four shapes (3 profiles + tower) are returned for inspection. Named "failed" as the loft result may not match the intended antiprism exactly, but demonstrates multi-section loftWith well.

## Keywords
loftWith, loft, antiprism, polarLine, translate, sketchOnPlane, draw, hLine, vLine, multi-section loft, tower, 3 profiles, height offset, polygon, reference

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| scale | 0.1 | - | All dimensions scaled by 0.1 |
| baseLength | 20 | mm | Square side (200×0.1) |
| topLength | ~14.14 | mm | Side of rotated square (√2 × base/2) |
| height | ~117.1 | mm | Tower height ((1368-196.85)×0.1) |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw() | Starts 2D freeform drawing |
| .hLine() .vLine() | Straight orthogonal segments |
| .polarLine(len, angle) | Line segment defined by length and angle in degrees |
| .close() | Closes profile |
| .translate([dx,dy]) | Centers profiles around origin |
| .sketchOnPlane("XY", z) | Places profile at height z |
| .clone() | Duplicates sketch |
| .loftWith([s1, s2]) | Lofts through multiple cross-section sketches |

## Code
```javascript
const {draw} = replicad;
function main() {
  let scale=1/10, baseLength=200*scale;
  let topLength=Math.sqrt(2*Math.pow(baseLength/2,2));
  let height=(1368-196.85)*scale;
  let baseProfile = draw().hLine(baseLength).vLine(baseLength).hLine(-baseLength).close()
    .translate([-baseLength/2,-baseLength/2]).sketchOnPlane("XY");
  let midProfile = draw().hLine(baseLength/2).polarLine(topLength/2,45).vLine(baseLength/2)
    .polarLine(topLength/2,135).hLine(-baseLength/2).polarLine(topLength/2,225).vLine(-baseLength/2).close()
    .translate([-baseLength/4,-baseLength/2]).sketchOnPlane("XY",height/2);
  let topProfile = draw().polarLine(topLength,45).polarLine(topLength,135).polarLine(topLength,225).close()
    .translate([0,-baseLength/2]).sketchOnPlane("XY",height);
  let tower = baseProfile.clone().loftWith([midProfile.clone(),topProfile.clone()]);
  return [{shape:baseProfile},{shape:midProfile},{shape:topProfile},{shape:tower}];
}
```
