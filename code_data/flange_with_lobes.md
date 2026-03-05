---
source_file: flange_with_lobes.js
category: structural
type: annotated_code
use_case: lobed flange hub, bearing seat, ribbed body, mounting studs, hex section
related: flange_assembly_telescoping.md, ball_bearings.md, shaft_design.md
---

# Flange With Lobes

## Description
3-lobed flanged hub with mounting studs at 120° intervals, hexagonal section, ribbed cylindrical body, through bore, and countersunk bearing seat. The three lobes are created by fusing circles at 120° angles onto a central circle, then filleting the merged sketch. Ribs are individual ring extrusions fused to the body. A chamfer is applied to the front flange face with a try/catch for geometry safety.

## Keywords
flange, lobed flange, 3 lobes, 120 degree, stud, hex, drawPolysides, ribs, bearing seat, bore, chamfer, drawCircle, fuse, cut, rotate, inPlane, try catch, hub, mechanical component

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| flangeThickness | 12 | mm | Flange plate thickness |
| flangeCenterRadius | 30 | mm | Central circle radius of flange |
| boltCircleRadius | 38 | mm | Distance from center to lobe center |
| lobeRadius | 14 | mm | Radius of each lobe circle |
| flangeFilletRadius | 12 | mm | 2D fillet on merged flange sketch |
| studRadius | 5 | mm | Mounting stud radius |
| studLength | 18 | mm | Mounting stud protrusion |
| hexRadius | 32 | mm | Hexagon circumradius |
| hexThickness | 15 | mm | Hexagonal section thickness |
| bodyOuterRadius | 28 | mm | Rib outer radius |
| bodyInnerRadius | 24 | mm | Body inner radius |
| bodyLength | 40 | mm | Ribbed body length |
| ribCount | 5 | - | Number of ribs |
| ribWidth | 4 | mm | Width of each rib |
| grooveWidth | 3 | mm | Gap between ribs |
| throughBoreRadius | 18 | mm | Through bore radius |
| bearingSeatRadius | 22 | mm | Bearing counter-bore radius |
| bearingSeatDepth | 10 | mm | Bearing seat depth |
| frontChamferSize | 1.5 | mm | Chamfer on top flange face |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawCircle(r) | Creates circular 2D sketch |
| .fuse(other2D) | 2D union to merge lobes onto flange |
| .fillet(r) | 2D fillet on merged flange outline |
| .sketchOnPlane("XY", z) | Places sketch at height z |
| .extrude(h) | Extrudes to 3D |
| .translate(x,y,z) | Positions shapes |
| .rotate(angle) | Rotates around Z axis |
| drawPolysides(r, n) | Creates hexagonal sketch |
| .cut(other) | Boolean subtract for bore and bearing seat |
| .chamfer(r, edgeFinder) | Chamfers front flange face |
| .inPlane("XY", z) | Selects face at height z |
| try/catch | Safely skips chamfer if geometry fails |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides, drawRectangle } = replicad;
  const flangeThickness=12, flangeCenterRadius=30, boltCircleRadius=38;
  const lobeRadius=14, flangeFilletRadius=12, studRadius=5, studLength=18;
  const hexRadius=32, hexThickness=15, bodyOuterRadius=28, bodyInnerRadius=24, bodyLength=40;
  const ribCount=5, ribWidth=4, grooveWidth=3;
  const throughBoreRadius=18, bearingSeatRadius=22, bearingSeatDepth=10, frontChamferSize=1.5;
  // Build 3-lobe flange sketch
  let flangeSketch = drawCircle(flangeCenterRadius);
  for(let i=0;i<3;i++){
    const lobe = drawCircle(lobeRadius).translate(boltCircleRadius,0).rotate(i*120);
    flangeSketch = flangeSketch.fuse(lobe);
  }
  const flange = flangeSketch.fillet(flangeFilletRadius).sketchOnPlane("XY").extrude(flangeThickness);
  // Studs at 120° intervals
  let studsShape=null;
  for(let i=0;i<3;i++){
    const stud = drawCircle(studRadius).sketchOnPlane("XY",flangeThickness).extrude(studLength)
      .translate(boltCircleRadius,0,0).rotate(i*120);
    studsShape = studsShape ? studsShape.fuse(stud) : stud;
  }
  const hexObj = drawPolysides(hexRadius,6).sketchOnPlane("XY",-hexThickness).extrude(hexThickness);
  const bodyStartZ = -(hexThickness+bodyLength);
  let bodyObj = drawCircle(bodyInnerRadius).sketchOnPlane("XY",bodyStartZ).extrude(bodyLength);
  const totalRibSection=ribCount*ribWidth+(ribCount-1)*grooveWidth;
  const startOffset=(bodyLength-totalRibSection)/2;
  for(let i=0;i<ribCount;i++){
    const zPos=bodyStartZ+startOffset+(i*(ribWidth+grooveWidth));
    bodyObj=bodyObj.fuse(drawCircle(bodyOuterRadius).sketchOnPlane("XY",zPos).extrude(ribWidth));
  }
  let hub = flange.fuse(hexObj).fuse(bodyObj).fuse(studsShape);
  hub = hub.cut(drawCircle(throughBoreRadius).sketchOnPlane("XY",bodyStartZ-10).extrude(bodyLength+hexThickness+flangeThickness+20));
  hub = hub.cut(drawCircle(bearingSeatRadius).sketchOnPlane("XY",flangeThickness).extrude(-bearingSeatDepth));
  try { hub=hub.chamfer(frontChamferSize,(e)=>e.inPlane("XY",flangeThickness)); } catch(err){}
  return hub;
};
```
