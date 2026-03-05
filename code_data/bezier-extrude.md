---
source_file: bezier-extrude.js
category: replicad_example
type: annotated_code
use_case: curved path extrusion, bezier curve modeling
related: arc_ellipse.md, cannedSketches.md
---

# Bezier Extrude

## Description
Demonstrates bezier curve creation in replicad using control points to define a complex curved path, then extruding it into a 3D solid. Useful for creating organic curved shapes, handles, or complex path-based geometries.

## Keywords
bezier, bezierCurveTo, Sketcher, extrude, curve, control points, XZ plane, sweep, curved path, organic shape

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| width | 30 | mm | Width of profile |
| thickness | 1 | mm | Thickness of profile |
| p0 | [0,0] | mm | Start point of bezier |
| p1-p4 | various | mm | Bezier control points |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XZ") | Creates 2D sketch on XZ plane |
| .movePointerTo([x,y]) | Moves sketch cursor to point |
| .bezierCurveTo(end, controls) | Draws bezier curve to endpoint through control points |
| .done() | Finalizes open sketch (not closed) |
| .extrude(depth) | Extrudes sketch into 3D solid |

## Code
```javascript
const main = ({ Sketcher, BlueprintSketcher, genericSweep }) => {
  let p0=[0,0], p1=[50,100], p2=[60,-95], p3=[80,30], p4=[100,25];
  let points = [p1,p2,p3,p4];
  let testBezier = new Sketcher("XZ")
    .movePointerTo(p0)
    .bezierCurveTo(p4, points)
    .done();
  testBezier = testBezier.extrude(30);
  return testBezier;
};
```
