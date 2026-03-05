---
source_file: cycle_frame.js
category: structural
type: annotated_code
use_case: bicycle frame, tube assembly, structural design
related: crankshaft.md, brake_rotor.md, shaft_design.md
---

# Cycle Frame

## Description
Detailed bicycle frame with all major structural tubes — head tube, seat tube, top tube, down tube, chain stays, seat stays, bottom bracket shell, and rear dropouts. Tubes are built using a helper function that orients cylinders between two 3D points, enabling accurate geometry for any tube angle. Inner bores are cut for head tube, seat tube, and BB shell.

## Keywords
bicycle frame, tube, makeTube, drawCircle, rotate, translate, fuse, cut, normalize, cross product, dropout, chain stay, seat stay, bottom bracket, head tube, aluminum, structural

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| bbWidth | 68 | mm | Bottom bracket shell width |
| bbRadius | 26 | mm | BB shell outer radius |
| seatTubeLength | 540 | mm | Seat tube length |
| seatTubeRadius | 22 | mm | Seat tube outer radius |
| seatTubeAngle | 81 | deg | Seat tube angle from horizontal |
| headTubeLength | 130 | mm | Head tube length |
| headTubeAngle | 73 | deg | Head tube angle |
| topTubeLength | 560 | mm | Top tube length |
| chainStayLength | 410 | mm | Chain stay length |
| rearAxleWidth | 130 | mm | Rear axle spacing |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawCircle(r) | Creates circular cross-section for tubes |
| .sketchOnPlane() | Places sketch on default plane |
| .extrude(len) | Extrudes circle to cylinder |
| .rotate(angle, origin, axis) | Rotates cylinder to match tube direction |
| .translate([x,y,z]) | Moves tube to start point |
| drawRoundedRectangle(w,h,r) | Creates dropout plate shape |
| .cut(other) | Boolean subtract for bores and slots |
| .fuse(other) | Boolean union to assemble frame |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle, drawRoundedRectangle } = replicad;
  // makeTube helper: creates cylinder between two 3D points
  const makeTube = (p1, p2, r) => {
    const v=sub(p2,p1), len=Math.sqrt(v[0]**2+v[1]**2+v[2]**2), dir=normalize(v);
    let shape = drawCircle(r).sketchOnPlane().extrude(len);
    const rotAxis=cross([0,0,1],dir);
    const rotAngle=Math.acos(dot([0,0,1],dir))*180/Math.PI;
    if(Math.abs(rotAngle)>0.01) shape=shape.rotate(rotAngle,[0,0,0],rotAxis);
    return shape.translate(p1);
  };
  // Key joint points computed from angles and lengths
  // BB shell, head tube, seat tube, top tube, down tube, stays, dropouts
  // All tubes fused, then bores cut
  const solidFrame = bbShell.fuse(seatTube).fuse(headTube).fuse(topTube)
    .fuse(downTube).fuse(dropouts).fuse(stays);
  const finalFrame = solidFrame.cut(bbCutter).cut(headCutter).cut(seatCutter);
  return [{shape: finalFrame, name:"Detailed Cycle Frame", color:"#a8b5c4"}];
};
```
