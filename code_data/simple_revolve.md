---
source_file: simple_revolve.js
category: modeling, primitives
type: annotated_code
use_case: revolved solid with cutouts, boolean operations
related: simpleVase.md, simplehouse.md
---

# Simple Revolve with Boolean Cutouts

## Description
Creates a rotationally symmetric base shape by revolving a Sketcher profile containing a half-ellipse, then cuts away a quadratic-bezier extruded solid and a rectangular box to produce an open, architectural cross-section form. Demonstrates combining revolve, extrude, and multi-step boolean cuts.

## Keywords
revolve, Sketcher, halfEllipse, hLine, close, boolean cut, quadratic bezier, sketchRectangle, extrude, translate, multi-body, XZ plane, XY plane, parametric

## Parameters
| Variable       | Value     | Unit | Meaning                                         |
|----------------|-----------|------|-------------------------------------------------|
| hLine (base)   | 25        | mm   | Base radius of the revolved profile             |
| halfEllipse    | 0, 40, 15 | mm   | Ellipse endpoint (dX=0, dY=40) with rx=15       |
| cutter rect    | 40 × 40   | mm   | Rectangle used to cut the revolve               |
| bezier dY      | 20, 30    | mm   | Bezier control point and endpoint Y offsets     |
| translateY     | -12       | mm   | Offset of bezier cutter below center            |

## Code
```javascript
// FILE: simple_revolve.js
// Revolved profile with a half-ellipse, then cut by a bezier solid and a box.

const main = ({ Sketcher, sketchRectangle }) => {

  // --- Revolve profile: drawn in XZ plane ---
  // hLine = base radius, halfEllipse creates a smooth dome top,
  // second hLine closes the top back to axis.
  const sketch = new Sketcher("XZ")
    .hLine(25)                           // base radius = 25mm
    .halfEllipse(0, 40, 15, true)        // dome: height=40, half-width=15
    .hLine(-25)                          // back to axis
    .close();

  // Full 360° revolve around Z axis
  let base = sketch.clone().revolve([0, 0, 1]);

  // --- Bezier cutter: an organic half-profile, mirrored and extruded ---
  const hole = new Sketcher()
    .quadraticBezierCurveTo([0, 20], [20, 30])  // curve to point [20,30] via [0,20]
    .closeWithMirror()                            // mirror to close symmetrically
    .extrude(40)
    .translateY(-12);                             // shift below center

  // --- Box cutter: removes a quadrant of the revolve ---
  let cutter = sketchRectangle(40, 40)
    .extrude(40)
    .translate([20, -20, 0]);

  // --- Boolean cuts ---
  let revolveShape = base.cut(cutter);  // cut the quarter-box out

  // Return the cut shape + the original sketch (shown in light grey for reference)
  return [
    { shape: revolveShape },
    { shape: sketch.extrude(0.1), color: "lightgrey" }  // profile visualization
  ];
};
```
