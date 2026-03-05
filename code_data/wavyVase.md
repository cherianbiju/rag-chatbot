---
source_file: wavyVase.js
category: decorative, consumer_product
type: annotated_code
use_case: twisted polygon vase, parametric decorative object
related: simpleVase.md, drawPolysides.md
---

# Wavy Twisted Polygon Vase

## Description
Generates a vase with a polygonal cross-section that twists helically as it extrudes upward, producing a wave or spiral visual effect. Wall thickness is controlled either via a circular inner bore or a shell operation, and optional fillets smooth the top and bottom edges.

## Keywords
vase, polygon, polysides, twist, extrusion profile, s-curve, shell, inner radius, fillet, wavyVase, parametric, drawPolysides, drawCircle, wall thickness, decorative, 3D print

## Parameters
| Variable       | Value | Unit | Meaning                                               |
|----------------|-------|------|-------------------------------------------------------|
| height         | 150   | mm   | Total vase height                                     |
| radius         | 40    | mm   | Outer circumradius of the polygon cross-section       |
| sidesCount     | 12    | —    | Number of polygon sides                               |
| sideRadius     | -2    | mm   | Corner rounding of polygon (negative = concave sides) |
| sideTwist      | 6     | —    | Number of side-widths to twist over full height       |
| endFactor      | 1.5   | —    | S-curve end factor for extrusion profile shaping      |
| topFillet      | 0     | mm   | Fillet radius at top rim (0 = disabled)               |
| bottomFillet   | 5     | mm   | Fillet radius at base edges                           |
| holeMode       | 1     | —    | 1 = circular bore, 2 = shell operation, 0 = solid     |
| wallThickness  | 2     | mm   | Wall thickness for hollowing                          |

## Code
```javascript
// FILE: wavyVase.js
// Twisted polygon vase with configurable sides, twist, and wall mode.

const { drawCircle, drawPolysides, polysideInnerRadius } = replicad;

const defaultParams = {
  height: 150,
  radius: 40,
  sidesCount: 12,
  sideRadius: -2,
  sideTwist: 6,
  endFactor: 1.5,
  topFillet: 0,
  bottomFillet: 5,
  holeMode: 1,       // 1=circle bore, 2=shell, 0=solid
  wallThickness: 2,
};

const main = (
  r,
  { height, radius, sidesCount, sideRadius, sideTwist,
    endFactor, topFillet, bottomFillet, holeMode, wallThickness }
) => {

  // --- Extrusion profile: s-curve warps the cross-section as it rises ---
  const extrusionProfile = endFactor
    ? { profile: "s-curve", endFactor }
    : undefined;

  // --- Total twist angle across full height ---
  const twistAngle = (360 / sidesCount) * sideTwist;

  // --- Extrude polygon with twist ---
  let shape = drawPolysides(radius, sidesCount, -sideRadius)
    .sketchOnPlane()
    .extrude(height, {
      twistAngle,           // helical rotation during extrusion
      extrusionProfile,     // s-curve profile warping
    });

  // --- Fillet base edges ---
  if (bottomFillet) {
    shape = shape.fillet(bottomFillet, (e) => e.inPlane("XY"));
  }

  // --- Hollow out the vase ---
  if (holeMode === 1 || holeMode === 2) {
    const holeHeight = height - wallThickness;

    if (holeMode === 1) {
      // Mode 1: cut a circular cylinder from inside
      const insideRadius =
        polysideInnerRadius(radius, sidesCount, sideRadius) - wallThickness;

      let hole = drawCircle(insideRadius)
        .sketchOnPlane()
        .extrude(holeHeight, { extrusionProfile });

      shape = shape.cut(
        hole
          .fillet(
            Math.max(wallThickness / 3, bottomFillet - wallThickness),
            (e) => e.inPlane("XY")
          )
          .translate([0, 0, wallThickness])   // raise hole off the base
      );
    } else if (holeMode === 2) {
      // Mode 2: shell operation — removes top face and offsets inward
      shape = shape.shell(wallThickness, (f) => f.inPlane("XY", height));
    }
  }

  // --- Optional top rim fillet ---
  if (topFillet) {
    shape = shape.fillet(topFillet, (e) => e.inPlane("XY", height));
  }

  return shape;
};
```
