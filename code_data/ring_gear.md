---
source_file: ring_gear.js
category: differential
type: annotated_code
use_case: receives torque from driveshaft pinion and transfers it to differential carrier
related: bevel_pinion.md, spider_gear.md, diff_carrier.md
---
# Differential Ring Gear

## Description
A large bevel ring gear that meshes with the drive pinion to change torque direction 90 degrees. Bolted to the differential carrier, it forms the final drive ratio. Carburized alloy steel with curved hypoid teeth.

## Keywords
ring gear, bevel gear, hypoid, final drive, differential, carrier, torque, crown wheel, revolve, draw, fuse, cut, cylinder, bolt holes, gear teeth

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| RING_OUTER_R | 120 | mm | outer radius of ring gear |
| RING_INNER_R | 90 | mm | inner bolt flange radius |
| GEAR_THICKNESS | 22 | mm | gear face thickness |
| FLANGE_THICKNESS | 12 | mm | bolt flange thickness |
| BOLT_PCD | 100 | mm | bolt pattern circle diameter |
| NUM_BOLTS | 8 | — | number of mounting bolts |
| BOLT_HOLE_R | 5.5 | mm | bolt hole radius |
| TOOTH_APPROX_H | 8 | mm | approximate tooth height |
| NUM_TEETH | 39 | — | number of ring teeth (3.9:1 ratio) |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const RING_OUTER_R    = 120;
  const RING_INNER_R    = 90;
  const GEAR_THICKNESS  = 22;
  const FLANGE_THICK    = 12;
  const BOLT_PCD        = 100;
  const NUM_BOLTS       = 8;
  const BOLT_HOLE_R     = 5.5;
  const TOOTH_H         = 8;
  const NUM_TEETH       = 39;

  // Ring gear body profile
  const profile = draw([RING_INNER_R, 0])
    .hLine(RING_OUTER_R - RING_INNER_R)
    .vLine(GEAR_THICKNESS)
    .hLine(-(RING_OUTER_R - RING_INNER_R - TOOTH_H))
    .vLine(FLANGE_THICK)
    .hLine(-(TOOTH_H))
    .close();

  let ring = profile.sketchOnPlane("XZ").revolve();

  // Approximate teeth as radial bumps on outer edge
  const TOOTH_W = (2 * Math.PI * (RING_OUTER_R - TOOTH_H / 2)) / NUM_TEETH * 0.45;
  for (let i = 0; i < NUM_TEETH; i++) {
    const angle = (i / NUM_TEETH) * 360;
    const tooth = draw([-TOOTH_W / 2, RING_OUTER_R - TOOTH_H])
      .hLine(TOOTH_W)
      .vLine(TOOTH_H)
      .hLine(-TOOTH_W)
      .close()
      .sketchOnPlane("XY")
      .extrude(GEAR_THICKNESS)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    ring = ring.fuse(tooth);
  }

  // Bolt holes
  for (let i = 0; i < NUM_BOLTS; i++) {
    const angle = (i / NUM_BOLTS) * 360;
    const boltHole = makeCylinder(BOLT_HOLE_R, FLANGE_THICK + GEAR_THICKNESS + 2, [(BOLT_PCD / 2), 0, -1], [0, 0, 1])
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    ring = ring.cut(boltHole);
  }

  return { shape: ring, name: "Ring Gear", color: "steelblue" };
};
```
