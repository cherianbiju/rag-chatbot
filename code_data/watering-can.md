---
source_file: watering-can.js
category: consumer_product, household
type: annotated_code
use_case: watering can modeling, loft, fillet, shell workflow
related: simpleVase.md, loft_examples.md
---

# Watering Can

## Description
Constructs a watering can by revolving a 2D side profile to form the body, lofting between tilted circular cross-sections to form the filler neck, and attaching a cylindrical spout. The final shape is hollowed using a shell operation that opens both the top filler and spout end simultaneously.

## Keywords
watering can, revolve, loft, shell, fillet, makePlane, makeCylinder, drawCircle, spout, filler neck, pivot, boolean fuse, shell opening, household, parametric, profile

## Parameters
| Variable        | Value         | Unit | Meaning                                           |
|-----------------|---------------|------|---------------------------------------------------|
| profile hLine   | 20            | mm   | Base radius of the body at ground level           |
| spout radius    | 5             | mm   | Radius of the cylindrical spout                   |
| spout length    | 70            | mm   | Length of the spout cylinder                      |
| spout angle     | 45            | deg  | Angle of the spout from vertical                  |
| topCircle r     | 12            | mm   | Radius of the top filler opening                  |
| middleCircle r  | 8             | mm   | Radius of loft waist at z=100                     |
| bottomCircle r  | 9             | mm   | Radius of loft base on angled plane at z=80       |
| fillet (spout)  | 10            | mm   | Fillet at spout-to-body junction                  |
| fillet (neck)   | 30            | mm   | Fillet at filler neck base                        |
| shell thickness | 1             | mm   | Wall thickness (shell inward offset)              |

## Code
```javascript
// FILE: watering-can.js
// Watering can: revolved body + lofted filler neck + cylindrical spout, then shelled.

const { makePlane, makeCylinder, draw, drawCircle } = replicad;

const main = () => {

  // --- Body: revolve a 2D profile around Z axis ---
  // Profile drawn in XZ plane: X = radius from axis, Z = height
  const profile = draw()
    .hLine(20)          // base radius = 20mm
    .line(10, 5)        // slight outward flare at base
    .vLine(3)           // short vertical section
    .lineTo([8, 100])   // taper inward as we go up
    .hLine(-8)          // close top to axis
    .close();

  const body = profile.sketchOnPlane("XZ").revolve([0, 0, 1]);

  // --- Filler neck: loft between three tilted circles ---
  // Top opening: tilted plane so the filler mouth angles outward
  const topPlane = makePlane()
    .pivot(-20, "Y")              // tilt 20° around Y
    .translate([-35, 0, 135]);    // position above and to the side of body
  const topCircle = drawCircle(12).sketchOnPlane(topPlane);

  // Waist: flat circle at z=100
  const middleCircle = drawCircle(8).sketchOnPlane("XY", 100);

  // Bottom of neck: angled plane at z=80
  const bottomPlane = makePlane().pivot(20, "Y").translateZ(80);
  const bottomCircle = drawCircle(9).sketchOnPlane(bottomPlane);

  // Loft through all three circles (smooth, not ruled)
  const filler = topCircle.loftWith([middleCircle, bottomCircle], { ruled: false });

  // --- Spout: cylinder tilted 45° upward from z=100 ---
  const spout = makeCylinder(5, 70)
    .translateZ(100)
    .rotate(45, [0, 0, 100], [0, 1, 0]);  // pivot around point at z=100

  // --- Assemble: fuse body + filler neck, add fillet at junction, fuse spout ---
  let wateringCan = body
    .fuse(filler)
    .fillet(30, (e) => e.inPlane("XY", 100))          // smooth neck-to-body blend
    .fuse(spout)
    .fillet(10, (e) => e.inBox([20, 20, 100], [-20, -20, 120]));  // spout junction

  // Compute spout tip position for face selection
  const spoutOpening = [
    Math.cos((45 * Math.PI) / 180) * 70,
    0,
    100 + Math.sin((45 * Math.PI) / 180) * 70,
  ];

  // --- Shell: remove both the filler top face and spout end face ---
  // shell(-1) offsets inward by 1mm, opening faces that match the selector
  wateringCan = wateringCan.shell(-1, (face) =>
    face.either([
      (f) => f.containsPoint(spoutOpening),   // spout opening face
      (f) => f.inPlane(topPlane),             // filler neck top face
    ])
  );

  return { shape: wateringCan, name: "Watering Can" };
};
```
