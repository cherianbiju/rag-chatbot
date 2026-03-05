---
source_file: projections.js
category: engineering-drawing
type: annotated_code
use_case: generating all six first-angle orthographic projection views from a 3D shape using drawProjection
related: knob11_pretty.md, inlist-test.md
---
# Projections — Six-View First-Angle Orthographic Drawing

## Description
Demonstrates how to generate all six standard first-angle orthographic projection views (front, back, top, bottom, left, right) from a 3D solid using replicad's `drawProjection` function. The helper function `descriptiveGeom` returns the original shape alongside all six visible-line projections as named 2D shapes. The test solid is an L-profile extrusion with a chamfered edge that looks different from every direction.

## Keywords
drawProjection, orthographic-projection, first-angle, engineering-drawing, front-view, top-view, side-view, chamfer, customCorner, extrude, vLine, hLine, containsPoint, inPlane, replicad, 2D-drawing, technical-drawing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| profile | L-shape 15×15 approx | mm | Irregular L-profile for the test solid |
| extrude depth | 10 | mm | Solid extrusion depth |
| chamfer | 5 | mm | Chamfer on one top edge (selected by inPlane + containsPoint) |
| customCorner | 2 | mm | Fillet radius on one profile corner |
| projection views | 6 | — | front, back, top, bottom, left, right |
| projection type | visible lines only | — | Only .visible shapes are extracted |

## Code
```javascript
const { drawProjection, draw } = replicad;

/* First angle projection convention
 * https://en.wikipedia.org/wiki/Multiview_orthographic_projection#First-angle_projection
 */
const descriptiveGeom = (shape) => {
  return [
    { shape, name: "Shape to project" },
    { shape: drawProjection(shape, "front").visible, name: "Front" },
    { shape: drawProjection(shape, "back").visible, name: "Back" },
    { shape: drawProjection(shape, "top").visible, name: "Top" },
    { shape: drawProjection(shape, "bottom").visible, name: "Bottom" },
    { shape: drawProjection(shape, "left").visible, name: "Left" },
    { shape: drawProjection(shape, "right").visible, name: "Right" },
  ];
};

const main = () => {
  const shape = draw()
    .vLine(-10)
    .hLine(-5)
    .vLine(15)
    .customCorner(2)
    .hLine(15)
    .vLine(-5)
    .close()
    .sketchOnPlane()
    .extrude(10)
    .chamfer(5, (e) => e.inPlane("XY", 10).containsPoint([10, 1, 10]));

  return descriptiveGeom(shape);
};
```
