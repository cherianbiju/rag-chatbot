---
source_file: simplehouse.js
category: architecture, modeling
type: annotated_code
use_case: house profile modeling, bezier curves, makeOffset
related: test-offset-2-rc.md, bezier_curves.md
---

# Simple House with Bezier Wing and Offset

## Description
Constructs a simple peaked-roof house shape using a Sketcher profile, then applies a 3D offset to thicken it. A second organic shape is created with a high-degree Bezier curve and fused to the house, demonstrating how straight-line architectural profiles and freeform curves can be combined in one model.

## Keywords
house, peaked roof, Sketcher, vLine, line, extrude, makeOffset, bezier curve, fuse, architecture, organic shape, profile, 3D offset, parametric

## Parameters
| Variable      | Value                          | Unit | Meaning                                      |
|---------------|--------------------------------|------|----------------------------------------------|
| house vLine   | 50                             | mm   | Wall height of the house                     |
| roof lines    | line(10,25) / line(10,-25)     | mm   | Roof ridge offset (dx=10, dz=±25)            |
| house depth   | 40                             | mm   | Extrusion depth of the house                 |
| offset amount | 3                              | mm   | Outward 3D offset applied to house solid     |
| bezier depth  | 30                             | mm   | Extrusion depth of the organic wing          |

## Code
```javascript
// FILE: simplehouse.js
// A peaked-roof house solid fused with an organic bezier-curve wing.
// Demonstrates Sketcher profiles, makeOffset, and bezier curves.

const main = ({ Sketcher, makeOffset }) => {

  // --- House profile in XZ plane (X = width, Z = height) ---
  let houseSketch = new Sketcher("XZ")
    .vLine(50)          // left wall up
    .line(10, 25)       // left roof slope
    .line(10, -25)      // right roof slope
    .vLine(-50)         // right wall down
    .close();

  let house = houseSketch.extrude(40);   // extrude 40mm deep along Y

  // Apply 3D offset to thicken/round the house solid, then move it aside
  house = makeOffset(house, 3).translate([0, 60, 0]);

  // --- Organic wing: high-degree Bezier curve profile ---
  // Control points define an S/wave-like silhouette in XZ
  let p0 = [0, 0];
  let p1 = [50, 100];
  let p2 = [60, -95];
  let p3 = [80, 30];
  let p4 = [100, 25];
  let points = [p1, p2, p3, p4];  // intermediate control points

  let testBezier = new Sketcher("XZ")
    .movePointerTo(p0)
    .bezierCurveTo(p4, points)   // high-degree bezier from p0 to p4 via points
    .vLine(-30)                  // close down
    .hLine(-100)                 // close back
    .close();

  testBezier = testBezier.extrude(30);  // extrude 30mm deep

  // --- Fuse house and bezier wing into one solid ---
  house = house.fuse(testBezier);

  return house;
};
```
