---
source_file: steering_rack.js
category: steering
type: annotated_code
use_case: converts pinion rotation into linear lateral motion to steer the wheels
related: pinion_gear.md, steering_housing.md, tie_rod.md
---
# Steering Rack Bar

## Description
A machined steel rack bar with teeth on the upper face meshing with the steering pinion. The rack slides laterally inside the housing on bushings. Both ends have threaded tie rod sockets.

## Keywords
steering rack, rack and pinion, rack bar, rack teeth, tie rod socket, lateral motion, steering, linear actuator, extrude, draw, cut, fuse, cylinder, thread

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| RACK_LENGTH | 400 | mm | total rack bar length |
| RACK_R | 14 | mm | rack bar radius |
| TOOTH_ZONE_LENGTH | 120 | mm | length of toothed section |
| TOOTH_H | 4 | mm | tooth height |
| NUM_TEETH | 20 | — | number of rack teeth |
| TIE_ROD_SOCKET_R | 10 | mm | tie rod socket bore radius |
| TIE_ROD_SOCKET_D | 25 | mm | tie rod socket depth |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
    makeBaseBox,
  } = replicad;

  const RACK_LENGTH         = 400;
  const RACK_R              = 14;
  const TOOTH_ZONE_LENGTH   = 120;
  const TOOTH_H             = 4;
  const NUM_TEETH           = 20;
  const TIE_ROD_SOCKET_R    = 10;
  const TIE_ROD_SOCKET_D    = 25;
  const TOOTH_ZONE_START    = (RACK_LENGTH - TOOTH_ZONE_LENGTH) / 2;

  // Main rack bar body
  let rack = drawCircle(RACK_R).sketchOnPlane("YZ").extrude(RACK_LENGTH);

  // Flat on top for teeth
  const flat = makeBaseBox(RACK_LENGTH + 2, RACK_R + 2, RACK_R)
    .translate(-1, -RACK_R - 1, RACK_R * 0.3);
  rack = rack.cut(flat);

  // Rack teeth
  const TOOTH_PITCH = TOOTH_ZONE_LENGTH / NUM_TEETH;
  const TOOTH_W = TOOTH_PITCH * 0.55;
  for (let i = 0; i < NUM_TEETH; i++) {
    const xPos = TOOTH_ZONE_START + i * TOOTH_PITCH;
    const tooth = makeBaseBox(TOOTH_W, RACK_R * 2, TOOTH_H)
      .translate(xPos, -RACK_R, RACK_R * 0.3 + RACK_R);
    rack = rack.fuse(tooth);
  }

  // Tie rod sockets both ends
  const socket1 = makeCylinder(TIE_ROD_SOCKET_R, TIE_ROD_SOCKET_D, [0, 0, 0], [1, 0, 0]);
  const socket2 = makeCylinder(TIE_ROD_SOCKET_R, TIE_ROD_SOCKET_D, [RACK_LENGTH - TIE_ROD_SOCKET_D, 0, 0], [1, 0, 0]);
  rack = rack.cut(socket1).cut(socket2);

  return { shape: rack, name: "Steering Rack", color: "steelblue" };
};
```
