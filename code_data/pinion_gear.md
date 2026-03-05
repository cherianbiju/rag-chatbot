---
source_file: pinion_gear.js
category: steering
type: annotated_code
use_case: converts steering column rotation into linear rack motion via helical teeth
related: steering_rack.md, steering_housing.md, tie_rod.md
---
# Steering Pinion Gear

## Description
An 18-tooth helical steel pinion that meshes with the rack teeth. The pinion shaft connects to the steering column universal joint above and rides on a needle roller bearing below inside the housing.

## Keywords
steering pinion, pinion gear, helical teeth, rack and pinion, steering column, universal joint, bearing journal, revolve, draw, fuse, cut, 18 teeth, helical

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PITCH_R | 18 | mm | pitch radius |
| TOOTH_H | 4 | mm | tooth height |
| GEAR_WIDTH | 30 | mm | face width |
| SHAFT_R | 10 | mm | shaft radius |
| SHAFT_LENGTH | 60 | mm | shaft length above gear |
| NUM_TEETH | 18 | — | number of teeth |
| UJ_FLAT_WIDTH | 16 | mm | universal joint flat width |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
    makeBaseBox,
  } = replicad;

  const PITCH_R      = 18;
  const TOOTH_H      = 4;
  const GEAR_WIDTH   = 30;
  const SHAFT_R      = 10;
  const SHAFT_LENGTH = 60;
  const NUM_TEETH    = 18;
  const UJ_FLAT_W    = 16;

  // Gear body
  let pinion = drawCircle(PITCH_R).sketchOnPlane("XY").extrude(GEAR_WIDTH);

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
      .extrude(GEAR_WIDTH)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    pinion = pinion.fuse(tooth);
  }

  // Shaft above gear
  const shaft = makeCylinder(SHAFT_R, SHAFT_LENGTH, [0, 0, GEAR_WIDTH], [0, 0, 1]);
  pinion = pinion.fuse(shaft);

  // Universal joint flats on shaft end
  const flat1 = makeBaseBox(SHAFT_R * 2 + 2, (SHAFT_R * 2 - UJ_FLAT_W) / 2 + 1, 20)
    .translate(-SHAFT_R - 1, SHAFT_R - (SHAFT_R * 2 - UJ_FLAT_W) / 2 - 1, GEAR_WIDTH + SHAFT_LENGTH - 20);
  const flat2 = makeBaseBox(SHAFT_R * 2 + 2, (SHAFT_R * 2 - UJ_FLAT_W) / 2 + 1, 20)
    .translate(-SHAFT_R - 1, -SHAFT_R, GEAR_WIDTH + SHAFT_LENGTH - 20);
  pinion = pinion.cut(flat1).cut(flat2);

  return { shape: pinion, name: "Steering Pinion", color: "steelblue" };
};
```
