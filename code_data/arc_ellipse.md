---
source_file: arc_ellipse.js
category: replicad_example
type: annotated_code
use_case: elliptical arc shapes, 2D profile creation
related: cannedSketches.md, bezier-extrude.md
---

# Arc Ellipse

## Description
Demonstrates the ellipse() drawing command in replicad by creating four different elliptical arc shapes with varying parameters. Shows how the long route and clockwise direction flags affect the resulting arc shape, returned as colored shapes for comparison.

## Keywords
ellipse, draw, arc, sketchOnPlane, elliptical arc, long route, clockwise, color, 2D sketch, XY plane, line, close

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| dx | 10 | mm | X displacement of ellipse endpoint |
| dy | 10 | mm | Y displacement of ellipse endpoint |
| radius_x | 20/50 | mm | X radius of ellipse |
| radius_y | 10 | mm | Y radius of ellipse |
| rotation | 0/135 | deg | Rotation angle of ellipse axis |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw([x,y]) | Starts a 2D drawing at given point |
| .line(dx,dy) | Draws a line by relative offset |
| .ellipse(dx,dy,rx,ry,rot,longRoute,ccw) | Draws elliptical arc with given radii and flags |
| .close() | Closes the sketch path |
| .sketchOnPlane("XY") | Places sketch on XY plane (returns flat face) |

## Code
```javascript
const {draw} = replicad;
function main() {
  let arcEllipse = draw([0,40]).line(10,0).ellipse(10,10,20,10,0,true,true).close().sketchOnPlane("XY");
  let arcEllipse2 = draw([0,20]).line(10,0).ellipse(10,10,20,10,0,false,true).close().sketchOnPlane("XY");
  let arcEllipse3 = draw([0,0]).line(10,0).ellipse(10,10,20,10,0,true,false).close().sketchOnPlane("XY");
  let arcEllipse4 = draw([0,-20]).line(10,0).ellipse(10,10,50,10,135,false,true).close().sketchOnPlane("XY");
  return [
    {shape: arcEllipse, color:"steelblue"},
    {shape: arcEllipse2, color:"red"},
    {shape: arcEllipse3, color:"purple"},
    {shape: arcEllipse4, color:"green"}
  ];
}
```
