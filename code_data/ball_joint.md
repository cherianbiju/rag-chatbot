---
source_file: ball_joint.js
category: suspension
type: annotated_code
use_case: allows multi-axis rotation between control arm and steering knuckle
related: control_arm.md, bushing.md
---
# Ball Joint

## Description
A ball-and-socket joint connecting the control arm to the steering knuckle. The tapered stud fits into the knuckle taper while the housing presses into the arm socket, allowing steering and suspension travel simultaneously.

## Keywords
ball joint, ball stud, socket, taper, suspension, steering knuckle, control arm, revolve, cylinder, sphere, fuse, cut, tapered stud

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| HOUSING_R | 16 | mm | housing outer radius |
| HOUSING_H | 30 | mm | housing height |
| BALL_R | 10 | mm | ball radius |
| STUD_R | 7 | mm | stud shank radius |
| STUD_LENGTH | 28 | mm | stud length above housing |
| TAPER_R_TOP | 5 | mm | stud top (small) radius |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const HOUSING_R  = 16;
  const HOUSING_H  = 30;
  const BALL_R     = 10;
  const STUD_R     = 7;
  const STUD_LENGTH = 28;
  const TAPER_R_TOP = 5;

  // Housing cylinder
  const housingProfile = draw([0, 0])
    .hLine(HOUSING_R)
    .vLine(HOUSING_H)
    .hLine(-HOUSING_R)
    .close();
  let housing = housingProfile.sketchOnPlane("XZ").revolve();

  // Socket cavity inside housing
  const socketProfile = draw([0, HOUSING_H * 0.2])
    .hLine(BALL_R + 1)
    .vLine(HOUSING_H * 0.7)
    .hLine(-(BALL_R + 1))
    .close();
  const socket = socketProfile.sketchOnPlane("XZ").revolve();
  housing = housing.cut(socket);

  // Tapered stud
  const studProfile = draw([TAPER_R_TOP, HOUSING_H])
    .lineTo([STUD_R, HOUSING_H + STUD_LENGTH * 0.4])
    .vLine(STUD_LENGTH * 0.6)
    .hLine(-STUD_R)
    .lineTo([0, HOUSING_H])
    .close();
  const stud = studProfile.sketchOnPlane("XZ").revolve();
  housing = housing.fuse(stud);

  return { shape: housing, name: "Ball Joint", color: "slategrey" };
};
```
