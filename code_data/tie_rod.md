---
source_file: tie_rod.js
category: steering
type: annotated_code
use_case: connects steering rack end to steering knuckle, transmitting lateral steering force
related: steering_rack.md, pinion_gear.md, steering_housing.md
---
# Tie Rod End

## Description
A steel tie rod with a threaded inner end that screws into the rack socket and a ball joint outer end that connects to the steering knuckle. The adjustable length allows toe alignment setting.

## Keywords
tie rod, tie rod end, ball joint, steering knuckle, rack end, toe adjustment, threaded rod, cylinder, revolve, draw, fuse, ball stud, steering

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| ROD_LENGTH | 180 | mm | total tie rod length |
| ROD_R | 8 | mm | rod shank radius |
| INNER_SOCKET_R | 14 | mm | inner threaded socket radius |
| INNER_SOCKET_L | 30 | mm | inner socket length |
| OUTER_BALL_R | 16 | mm | outer ball housing radius |
| OUTER_BALL_H | 28 | mm | outer ball housing height |
| LOCK_NUT_R | 14 | mm | lock nut hex radius |
| LOCK_NUT_H | 10 | mm | lock nut height |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const ROD_LENGTH      = 180;
  const ROD_R           = 8;
  const INNER_SOCKET_R  = 14;
  const INNER_SOCKET_L  = 30;
  const OUTER_BALL_R    = 16;
  const OUTER_BALL_H    = 28;
  const LOCK_NUT_R      = 14;
  const LOCK_NUT_H      = 10;

  // Inner socket
  const innerProfile = draw([0, 0])
    .hLine(INNER_SOCKET_R)
    .vLine(INNER_SOCKET_L)
    .hLine(-INNER_SOCKET_R)
    .close();
  let rod = innerProfile.sketchOnPlane("XZ").revolve();

  // Rod shank
  const shank = makeCylinder(ROD_R, ROD_LENGTH, [0, 0, INNER_SOCKET_L], [0, 0, 1]);
  rod = rod.fuse(shank);

  // Lock nut hex approximation (6-sided)
  const nutProfile = draw([0, 0])
    .hLine(LOCK_NUT_R)
    .vLine(LOCK_NUT_H)
    .hLine(-LOCK_NUT_R)
    .close();
  const lockNut = nutProfile.sketchOnPlane("XZ").revolve()
    .translateZ(INNER_SOCKET_L + 20);
  rod = rod.fuse(lockNut);

  // Outer ball housing
  const ballProfile = draw([0, 0])
    .hLine(OUTER_BALL_R)
    .vLine(OUTER_BALL_H)
    .hLine(-OUTER_BALL_R)
    .close();
  const ballHousing = ballProfile.sketchOnPlane("XZ").revolve()
    .translateZ(INNER_SOCKET_L + ROD_LENGTH);
  rod = rod.fuse(ballHousing);

  // Ball stud bore
  const studBore = makeCylinder(7, OUTER_BALL_H + 2, [0, 0, INNER_SOCKET_L + ROD_LENGTH - 1], [0, 0, 1]);
  rod = rod.cut(studBore);

  return { shape: rod, name: "Tie Rod End", color: "dimgrey" };
};
```
