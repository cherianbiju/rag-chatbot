---
source_file: brake_rotor.js
category: structural
type: annotated_code
use_case: automotive brake disc, ventilated rotor, mechanical engineering
related: crankshaft.md, cycle_frame.md, bolts_nuts.md
---

# Brake Rotor

## Description
Detailed ventilated brake disc with hub, friction plates, internal vanes for heat dissipation, lug bolt pattern, and cross-drilled holes. Represents a real automotive brake rotor with all major features including chamfers on the outer edge. Advanced example of parametric mechanical part design.

## Keywords
brake rotor, ventilated disc, hub, vanes, bolt pattern, cross drill, chamfer, draw, drawCircle, makeBaseBox, fuse, cut, rotate, automotive, mechanical, circular pattern, array

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| discOD | 300 | mm | Outer diameter of disc |
| discThickness | 28 | mm | Total disc thickness |
| plateThickness | 9 | mm | Friction plate thickness |
| ventGap | 10 | mm | Gap between friction plates (vane space) |
| hubOD | 170 | mm | Hub outer diameter |
| hubHeight | 65 | mm | Hub height |
| centerBoreDia | 68 | mm | Centre bore diameter |
| boltCount | 5 | - | Number of lug bolt holes |
| boltPCD | 130 | mm | Bolt hole pitch circle diameter |
| boltHoleDia | 12.5 | mm | Lug bolt hole diameter |
| vaneCount | 36 | - | Number of internal cooling vanes |
| drillHoleDia | 5 | mm | Cross-drill hole diameter |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw([x,y]) | Starts 2D drawing at point |
| .lineTo([x,y]) | Draws line to absolute point |
| .close() | Closes sketch path |
| .sketchOnPlane("XY",z) | Places sketch at height z |
| .revolve() | Revolves profile 360° to create solid of revolution |
| drawCircle(r) | Creates circular sketch |
| .cut(other) | Boolean subtract |
| makeBaseBox(l,w,h) | Creates box at origin |
| .translate(x,y,z) | Moves shape |
| .clone() | Duplicates shape |
| .rotate(angle) | Rotates around Z axis |
| .fuse(other) | Boolean union |
| .chamfer(r, edgeFinder) | Chamfers matching edges |
| .ofCurveType("CIRCLE") | Finds circular edges |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, makeBaseBox } = replicad;
  const discOD=300, discThickness=28, plateThickness=9, ventGap=10;
  const hubOD=170, hubHeight=65, hubWallThickness=7, centerBoreDia=68;
  const boltCount=5, boltPCD=130, boltHoleDia=12.5;
  const vaneCount=36, vaneThickness=6;
  const drillHoleDia=5, drillRows=4;
  // Hub profile revolved
  const hubProfile = draw([centerBoreDia/2,0])
    .lineTo([centerBoreDia/2,hubHeight]).lineTo([hubOD/2,hubHeight])
    .lineTo([hubOD/2,discThickness-plateThickness])
    .lineTo([hubOD/2-hubWallThickness,discThickness-plateThickness])
    .lineTo([hubOD/2-hubWallThickness,hubWallThickness])
    .lineTo([centerBoreDia/2,hubWallThickness]).close();
  let hub = hubProfile.sketchOnPlane("XZ").revolve();
  // Plates, vanes, bolt holes, drill holes assembled and cut
  // ...full code in source file
  return [{shape: rotorBody, name:"Ventilated Brake Disc", color:"#71797E"}];
};
```
