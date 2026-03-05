---
source_file: plunge_improved.js
category: consumer-product
type: annotated_code
use_case: Plunge watering carafe — cleaner rewrite using makePlane for tilted cross-section positioning and either() for multi-face shell
related: plunge_example.md, plunge-v5-rc.md, occ-bottle.md, loft-pipe.md
---
# Plunge Watering Carafe — Improved Version

## Description
Cleaner rewrite of the Plunge carafe using `makePlane().pivot()` instead of face rotation and `sketchFaceOffset` to position the tilted filler opening. Also demonstrates the `either()` face selector to simultaneously shell two separate openings (spout tip and filler mouth) in a single shell call. The spout is built with a post-rotation translate rather than a starting offset cylinder. Generally the reference implementation for this model.

## Keywords
Plunge, watering-carafe, makePlane, pivot, drawCircle, loftWith, makeCylinder, shell, either, fillet, inPlane, inBox, containsPoint, translateZ, rotate, consumer-product, replicad, 3d-printing, Robert-Bronwasser, face-selector

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| body profile | 6 points on XZ plane | mm | Same profile as plunge_example |
| topPlane tilt | pivot(−20°, Y) | ° | Filler mouth plane tilted 20° from horizontal |
| topCircle radius | 12 | mm | Filler opening circle |
| topCircle position | [−35, 0, 135] | mm | Filler mouth position |
| middleCircle radius | 8, z=100 | mm | Body-top connection circle |
| bottomPlane tilt | pivot(+20°, Y), z=80 | ° | Filler bottom plane |
| bottomCircle radius | 9 | mm | Filler bottom circle |
| loft mode | ruled: false | — | Smooth filler loft |
| spout radius | 5 | mm | Spout cylinder radius |
| spout length | 70 | mm | Spout cylinder length |
| spoutAngle | 45° | ° | Spout inclination |
| fillet (body-filler) | 30 | mm | Junction fillet at z=100 |
| fillet (spout join) | 10 | mm | Spout junction fillet |
| shell thickness | −1 | mm | Shell thickness, two faces opened via either() |

## Code
```javascript
const { makePlane, makeCylinder, draw, drawCircle } = replicad;

const main = () => {
  const profile = draw()
    .hLine(20).line(10, 5).vLine(3).lineTo([8, 100]).hLine(-8).close();

  const body = profile.sketchOnPlane("XZ").revolve([0, 0, 1]);

  const topPlane = makePlane().pivot(-20, "Y").translate([-35, 0, 135]);
  const topCircle = drawCircle(12).sketchOnPlane(topPlane);
  const middleCircle = drawCircle(8).sketchOnPlane("XY", 100);
  const bottomPlane = makePlane().pivot(20, "Y").translateZ(80);
  const bottomCircle = drawCircle(9).sketchOnPlane(bottomPlane);

  const filler = topCircle.loftWith([middleCircle, bottomCircle], { ruled: false });

  const spout = makeCylinder(5, 70)
    .translateZ(100)
    .rotate(45, [0, 0, 100], [0, 1, 0]);

  let wateringCan = body
    .fuse(filler)
    .fillet(30, (e) => e.inPlane("XY", 100))
    .fuse(spout)
    .fillet(10, (e) => e.inBox([20, 20, 100], [-20, -20, 120]));

  const spoutOpening = [
    Math.cos((45 * Math.PI) / 180) * 70,
    0,
    100 + Math.sin((45 * Math.PI) / 180) * 70,
  ];

  wateringCan = wateringCan.shell(-1, (face) =>
    face.either([
      (f) => f.containsPoint(spoutOpening),
      (f) => f.inPlane(topPlane),
    ])
  );

  return { shape: wateringCan, name: "Watering Can" };
};
```
