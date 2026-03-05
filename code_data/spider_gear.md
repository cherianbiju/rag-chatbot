---
source_file: spider_gear.js
category: differential
type: annotated_code
use_case: allows speed difference between two axle outputs in an open differential
related: ring_gear.md, bevel_pinion.md, diff_carrier.md
---
# Spider Gear (Differential)

## Description
A small bevel gear mounted on the spider pin inside the differential carrier. Two spider gears mesh with both side gears, allowing the axles to rotate at different speeds during cornering.

## Keywords
spider gear, differential, bevel gear, side gear, open differential, axle, planet gear, revolve, draw, fuse, bore, pin hole, carrier

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PITCH_R | 22 | mm | pitch radius |
| TOOTH_H | 5 | mm | tooth height |
| GEAR_WIDTH | 20 | mm | face width |
| PIN_BORE_R | 8 | mm | spider pin bore radius |
| NUM_TEETH | 10 | — | number of teeth |
| BACK_BOSS_R | 14 | mm | back boss radius |
| BACK_BOSS_H | 8 | mm | back boss height |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const PITCH_R     = 22;
  const TOOTH_H     = 5;
  const GEAR_WIDTH  = 20;
  const PIN_BORE_R  = 8;
  const NUM_TEETH   = 10;
  const BACK_BOSS_R = 14;
  const BACK_BOSS_H = 8;

  // Gear cone
  const coneProfile = draw([0, 0])
    .lineTo([PITCH_R, GEAR_WIDTH])
    .hLine(-PITCH_R)
    .close();
  let gear = coneProfile.sketchOnPlane("XZ").revolve();

  // Teeth
  const TOOTH_W = (2 * Math.PI * PITCH_R) / NUM_TEETH * 0.5;
  for (let i = 0; i < NUM_TEETH; i++) {
    const angle = (i / NUM_TEETH) * 360;
    const tooth = draw([-TOOTH_W / 2, PITCH_R])
      .hLine(TOOTH_W)
      .vLine(TOOTH_H)
      .hLine(-TOOTH_W)
      .close()
      .sketchOnPlane("XY")
      .extrude(GEAR_WIDTH * 0.6)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    gear = gear.fuse(tooth);
  }

  // Back boss
  const bossProfile = draw([0, 0])
    .hLine(BACK_BOSS_R)
    .vLine(-BACK_BOSS_H)
    .hLine(-BACK_BOSS_R)
    .close();
  const boss = bossProfile.sketchOnPlane("XZ").revolve();
  gear = gear.fuse(boss);

  // Pin bore
  const pinBore = makeCylinder(PIN_BORE_R, GEAR_WIDTH + BACK_BOSS_H + 2, [0, 0, -BACK_BOSS_H - 1], [0, 0, 1]);
  gear = gear.cut(pinBore);

  return { shape: gear, name: "Spider Gear", color: "slategrey" };
};
```
