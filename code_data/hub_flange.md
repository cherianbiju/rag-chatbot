---
source_file: hub_flange.js
category: wheel_hub
type: annotated_code
use_case: mounts wheel to vehicle axle, houses bearing race, and carries ABS tone ring
related: bearing_race.md, abs_tone_ring.md, wheel_stud.md
---
# Wheel Hub Flange

## Description
A forged steel hub flange with integrated bearing race bore, 5×114.3mm wheel stud pattern, ABS tone ring land, and axle spline bore. The flanged face carries the wheel while the hub barrel fits into the knuckle bearing.

## Keywords
wheel hub, hub flange, wheel stud, 5x114.3, bearing race, ABS tone ring, axle bore, knuckle, forged steel, revolve, draw, fuse, cut, cylinder, bolt pattern

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| FLANGE_R | 70 | mm | wheel mounting flange radius |
| FLANGE_THICK | 20 | mm | flange thickness |
| BARREL_R | 38 | mm | hub barrel outer radius |
| BARREL_LENGTH | 55 | mm | hub barrel length |
| BEARING_BORE_R | 30 | mm | bearing inner race bore radius |
| AXLE_BORE_R | 18 | mm | axle spline bore radius |
| STUD_PCD | 57.15 | mm | stud pattern radius (114.3/2) |
| STUD_R | 7 | mm | wheel stud radius |
| NUM_STUDS | 5 | — | number of wheel studs |
| TONE_RING_R | 65 | mm | ABS tone ring land radius |
| TONE_RING_W | 10 | mm | tone ring land width |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const FLANGE_R      = 70;
  const FLANGE_THICK  = 20;
  const BARREL_R      = 38;
  const BARREL_LENGTH = 55;
  const BEARING_BORE_R = 30;
  const AXLE_BORE_R   = 18;
  const STUD_PCD      = 57.15;
  const STUD_R        = 7;
  const NUM_STUDS     = 5;
  const TONE_RING_R   = 65;
  const TONE_RING_W   = 10;

  // Flange disc
  const flangeProfile = draw([AXLE_BORE_R, 0])
    .hLine(FLANGE_R - AXLE_BORE_R)
    .vLine(FLANGE_THICK)
    .hLine(-(FLANGE_R - AXLE_BORE_R))
    .close();
  let hub = flangeProfile.sketchOnPlane("XZ").revolve();

  // Hub barrel behind flange
  const barrelProfile = draw([BEARING_BORE_R, 0])
    .hLine(BARREL_R - BEARING_BORE_R)
    .vLine(-BARREL_LENGTH)
    .hLine(-(BARREL_R - BEARING_BORE_R))
    .close();
  const barrel = barrelProfile.sketchOnPlane("XZ").revolve();
  hub = hub.fuse(barrel);

  // Axle bore through full hub
  const axleBore = makeCylinder(AXLE_BORE_R, FLANGE_THICK + BARREL_LENGTH + 2, [0, 0, -BARREL_LENGTH - 1], [0, 0, 1]);
  hub = hub.cut(axleBore);

  // ABS tone ring land — raised lip on flange face
  const toneProfile = draw([TONE_RING_R - 3, FLANGE_THICK])
    .hLine(3)
    .vLine(8)
    .hLine(-3)
    .close();
  const toneLand = toneProfile.sketchOnPlane("XZ").revolve();
  hub = hub.fuse(toneLand);

  // Wheel studs — 5x114.3mm
  for (let i = 0; i < NUM_STUDS; i++) {
    const angle = (i / NUM_STUDS) * 360;
    const stud = makeCylinder(STUD_R, FLANGE_THICK + 30, [STUD_PCD, 0, -2], [0, 0, 1])
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    hub = hub.fuse(stud);
  }

  // Stud press-fit holes through flange (studs press in from back)
  for (let i = 0; i < NUM_STUDS; i++) {
    const angle = (i / NUM_STUDS) * 360;
    const hole = makeCylinder(STUD_R - 1, FLANGE_THICK + 2, [STUD_PCD, 0, -1], [0, 0, 1])
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    hub = hub.cut(hole);
  }

  return { shape: hub, name: "Wheel Hub Flange", color: "steelblue" };
};
```
