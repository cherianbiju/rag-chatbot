---
source_file: simpleVase.js
category: decorative, consumer_product
type: annotated_code
use_case: parametric vase modeling, revolve with spline profile
related: wavyVase.md, simpleRevolve.md
---

# Parametric Simple Vase

## Description
Creates a rotationally symmetric vase by revolving a smooth spline profile around the Z axis, with configurable waist positions, radii, and wall thickness. The hollowing is achieved via the shell operation on the top face, and optional fillets are applied at the rim for a finished look.

## Keywords
vase, revolve, spline, smooth spline, shell, wall thickness, parametric, profile, fillet, draw, sketchOnPlane, rotationally symmetric, decorative, 3D print, height

## Parameters
| Variable             | Value | Unit | Meaning                                           |
|----------------------|-------|------|---------------------------------------------------|
| height               | 100   | mm   | Total vase height                                 |
| baseWidth            | 20    | mm   | Radius of the vase base                           |
| wallThickness        | 5     | mm   | Wall thickness after shelling                     |
| lowerCircleRadius    | 1.5   | ×base| Radius multiplier at lower bulge position         |
| lowerCirclePosition  | 0.25  | ×h   | Height fraction of lower bulge                    |
| higherCircleRadius   | 0.75  | ×base| Radius multiplier at upper narrowing position     |
| higherCirclePosition | 0.75  | ×h   | Height fraction of upper narrowing                |
| topRadius            | 0.9   | ×base| Radius multiplier at the top opening              |
| topFillet            | true  | —    | Whether to apply fillet to the rim edge           |
| bottomHeavy          | true  | —    | Increases start tangent factor for a heavier base |

## Code
```javascript
// FILE: simpleVase.js
// Parametric vase using a revolved smooth-spline profile with shelled walls.

const { draw } = replicad;

const defaultParams = {
  height: 100,
  baseWidth: 20,
  wallThickness: 5,
  lowerCircleRadius: 1.5,
  lowerCirclePosition: 0.25,
  higherCircleRadius: 0.75,
  higherCirclePosition: 0.75,
  topRadius: 0.9,
  topFillet: true,
  bottomHeavy: true,
};

const main = (
  r,
  {
    height,
    baseWidth,
    wallThickness,
    lowerCirclePosition,
    lowerCircleRadius,
    higherCircleRadius,
    higherCirclePosition,
    topRadius,
    topFillet,
    bottomHeavy,
  }
) => {

  // --- Spline control points (position along height, radius multiplier) ---
  // Each entry drives a smoothSplineTo call on the 2D profile.
  const splinesConfig = [
    { position: lowerCirclePosition,  radius: lowerCircleRadius },
    {
      position: higherCirclePosition,
      radius: higherCircleRadius,
      startFactor: bottomHeavy ? 3 : 1,  // higher startFactor = pulls curve toward bottom
    },
    { position: 1, radius: topRadius, startFactor: bottomHeavy ? 3 : 1 },
  ];

  // --- Build half-profile in XZ plane (X = radius, Z = height) ---
  const sketchVaseProfile = draw().hLine(baseWidth);

  splinesConfig.forEach(({ position, radius, startFactor, endFactor }) => {
    sketchVaseProfile.smoothSplineTo(
      [baseWidth * radius, height * position],  // target point
      {
        endTangent: [0, 1],   // arrive vertically (tangent pointing up)
        startFactor,
        endFactor,
      }
    );
  });

  // --- Close profile, revolve around Z axis ---
  let vase = sketchVaseProfile
    .lineTo([0, height])       // close to axis at top
    .close()
    .sketchOnPlane("XZ")
    .revolve();                // full 360° revolution around Z

  // --- Shell: remove top face to create hollow interior ---
  if (wallThickness) {
    vase = vase.shell(wallThickness, (f) => f.containsPoint([0, 0, height]));
  }

  // --- Optional rim fillet ---
  if (topFillet) {
    vase = vase.fillet(wallThickness / 3, (e) => e.inPlane("XY", height));
  }

  return vase;
};
```
