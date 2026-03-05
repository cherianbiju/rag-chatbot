---
source_file: test_polar_functions.js
category: utilities, geometry
type: annotated_code
use_case: polar coordinate drawing helpers, angled line construction
related: simplehouse.md, draw_utils.md
---

# Polar Coordinate Drawing Helpers

## Description
Defines three utility functions — `Polar`, `PolarX`, and `PolarY` — for computing new 2D points from a current position using polar (distance + angle) notation, making it easier to draw angled lines without manual trigonometry. A demo shape is built by stepping through a sequence of PolarY calls to approximate a segmented arc, then extruding it.

## Keywords
polar coordinates, angle, trigonometry, PolarX, PolarY, utility function, lineTo, draw, extrude, angled line, segment, 2D profile, helper, reusable, sketchOnPlane

## Parameters
| Variable | Value | Unit | Meaning                                          |
|----------|-------|------|--------------------------------------------------|
| radius1  | 20    | mm   | Starting radius / reference distance for points |
| step     | 5     | mm   | Y-distance per PolarY step                      |
| angles   | 100–260 | deg | Sequence of angles used for each PolarY step    |
| extrude  | 10    | mm   | Extrusion depth of the final shape               |

## Code
```javascript
// FILE: test_polar_functions.js
// Utility functions for polar-coordinate point construction,
// plus a demo shape that uses them.

const { draw, makeCylinder, makeBaseBox } = replicad;

// --- Polar: step from currentPoint by a given distance at an angle ---
// angleDegToX is measured from the positive X axis (standard math convention)
function Polar(currentPoint, distance, angleDegToX) {
  const angleRad = angleDegToX * Math.PI / 180;
  return [
    currentPoint[0] + distance * Math.cos(angleRad),
    currentPoint[1] + distance * Math.sin(angleRad),
  ];
}

// --- PolarX: step a fixed X distance; Y is derived from the angle ---
// Useful when you know how far to go horizontally along an angled line.
function PolarX(currentPoint, xdistance, angleDegToX) {
  const angleRad = angleDegToX * Math.PI / 180;
  return [
    currentPoint[0] + xdistance,
    currentPoint[1] + xdistance * Math.tan(angleRad),
  ];
}

// --- PolarY: step a fixed Y distance; X is derived from the angle ---
// Useful when you know how far to go vertically along an angled line.
function PolarY(currentPoint, ydistance, angleDegToX) {
  const angleRad = angleDegToX * Math.PI / 180;
  return [
    currentPoint[0] + ydistance / Math.tan(angleRad),
    currentPoint[1] + ydistance,
  ];
}

// --- Demo: build a segmented arc-like polygon using PolarY steps ---
function main() {
  const radius1 = 20;

  // Start at [radius1, 0] and walk up/down using PolarY steps
  let p1 = [radius1, 0];
  let p2 = PolarY(p1,  5, 100);
  let p3 = PolarY(p2,  5, 120);
  let p4 = PolarY(p3,  5, 135);
  let p5 = [0, radius1];         // manually set quarter-point
  let p6 = PolarY(p5, -5, 200);
  let p7 = PolarY(p6, -5, 220);
  let p8 = PolarY(p7, -5, 240);

  // Draw the closed polygon from these points and extrude
  let shape = draw()
    .lineTo(p1)
    .lineTo(p2)
    .lineTo(p3)
    .lineTo(p4)
    .lineTo(p5)
    .lineTo(p6)
    .lineTo(p7)
    .lineTo(p8)
    .close()
    .sketchOnPlane("XZ")
    .extrude(10);

  return shape;
}
```
