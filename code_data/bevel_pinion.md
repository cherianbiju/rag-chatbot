---
source_file: bevel_pinion.js
category: differential
type: annotated_code
use_case: drives the ring gear at 90 degrees to transmit driveshaft torque into the differential
related: ring_gear.md, spider_gear.md, diff_carrier.md
---
# Bevel Pinion Gear

## Description
A hardened steel drive pinion that meshes with the ring gear at 90 degrees. The pinion shaft runs into the differential housing on taper roller bearings. Tooth count gives 3.9:1 final drive ratio with a 39-tooth ring gear.

## Keywords
bevel pinion, drive pinion, hypoid, final drive, differential, pinion shaft, taper roller, gear teeth, revolve, draw, fuse, cut, bevel gear

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PITCH_R | 26 | mm | pitch radius |
| TOOTH_H | 7 | mm | tooth height |
| GEAR_WIDTH | 30 | mm | face width |
| SHAFT_R | 18 | mm | pinion shaft radius |
| SHAFT_LENGTH | 80 | mm | shaft length behind gear |
| NUM_TEETH | 10 | — | number of pinion teeth |
| BEARING_SEAT_R | 22 | mm | bearing seat radius |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const PITCH_R        = 26;
  const TOOTH_H        = 7;
  const GEAR_WIDTH     = 30;
  const SHAFT_R        = 18;
  const SHAFT_LENGTH   = 80;
  const NUM_TEETH      = 10;
  const BEARING_SEAT_R = 22;

  // Gear cone body
  const coneProfile = draw([0, 0])
    .lineTo([PITCH_R, GEAR_WIDTH])
    .hLine(-PITCH_R)
    .close();
  let pinion = coneProfile.sketchOnPlane("XZ").revolve();

  // Teeth approximated as rectangular bumps on outer face
  const TOOTH_W = (2 * Math.PI * PITCH_R) / NUM_TEETH * 0.5;
  for (let i = 0; i < NUM_TEETH; i++) {
    const angle = (i / NUM_TEETH) * 360;
    const tooth = draw([-TOOTH_W / 2, PITCH_R])
      .hLine(TOOTH_W)
      .vLine(TOOTH_H)
      .hLine(-TOOTH_W)
      .close()
      .sketchOnPlane("XY")
      .extrude(GEAR_WIDTH * 0.7)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    pinion = pinion.fuse(tooth);
  }

  // Pinion shaft
  const shaft = makeCylinder(SHAFT_R, SHAFT_LENGTH, [0, 0, -SHAFT_LENGTH], [0, 0, 1]);
  pinion = pinion.fuse(shaft);

  // Bearing seat
  const bearingSeat = makeCylinder(BEARING_SEAT_R, 30, [0, 0, -SHAFT_LENGTH + 10], [0, 0, 1]);
  pinion = pinion.fuse(bearingSeat);

  return { shape: pinion, name: "Bevel Pinion", color: "steelblue" };
};
```
