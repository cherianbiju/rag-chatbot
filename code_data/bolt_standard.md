---
source_file: bolt_standard.js
category: fastener
type: annotated_code
use_case: standard hex head bolt, M10 fastener, mechanical assembly
related: metric_threads.md, bolts_nuts.md
---

# Bolt Standard

## Description
Parametric standard hex-head bolt modeled to M10 dimensions with configurable wrench size, shaft diameter, shaft length, and chamfers. Creates a realistic bolt with hexagonal head, cylindrical shaft, tip chamfer, head chamfer, and neck fillet at the head-shaft junction.

## Keywords
bolt, hex bolt, M10, hexagon, drawPolysides, drawCircle, chamfer, fillet, extrude, fuse, fastener, wrench size, shaft, head, inPlane

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| wrenchSize | 17 | mm | Width across flats (M10 standard) |
| headHeight | 7 | mm | Height of hex head |
| shaftDiameter | 10 | mm | Bolt shaft diameter |
| shaftLength | 40 | mm | Length of shaft |
| tipChamfer | 1 | mm | Chamfer at bolt tip |
| headChamfer | 1 | mm | Chamfer at top of head |
| neckFillet | 0.6 | mm | Fillet at head-shaft junction |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawCircle(r) | Creates circular 2D sketch for shaft |
| drawPolysides(r, n) | Creates n-sided polygon (hexagon) for head |
| .sketchOnPlane("XY", z) | Places sketch at given Z height |
| .extrude(h) | Extrudes sketch to given height |
| .chamfer(r, edgeFinder) | Applies chamfer to matching edges |
| .inPlane("XY", z) | Finds edges in XY plane at height z |
| .fuse(other) | Boolean union of shaft and head |
| .fillet(r, edgeFinder) | Rounds neck edge at head junction |
| .ofCurveType("CIRCLE") | Finds only circular edges |

## Code
```javascript
const main = (replicad) => {
  const { drawPolysides, drawCircle } = replicad;
  const wrenchSize=17, headHeight=7, shaftDiameter=10, shaftLength=40;
  const tipChamfer=1, headChamfer=1, neckFillet=0.6;
  const hexRadius = wrenchSize / Math.sqrt(3);
  const shaftRadius = shaftDiameter / 2;
  let shaft = drawCircle(shaftRadius).sketchOnPlane("XY").extrude(shaftLength);
  shaft = shaft.chamfer(tipChamfer, (e) => e.inPlane("XY", 0));
  let head = drawPolysides(hexRadius, 6).sketchOnPlane("XY", shaftLength).extrude(headHeight);
  head = head.chamfer(headChamfer, (e) => e.inPlane("XY", shaftLength + headHeight));
  let bolt = shaft.fuse(head);
  bolt = bolt.fillet(neckFillet, (e) => e.inPlane("XY", shaftLength) && e.ofCurveType("CIRCLE"));
  return bolt;
};
```
