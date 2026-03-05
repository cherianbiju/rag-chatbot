---
source_file: valve_stem.js
category: engine
type: annotated_code
use_case: opens and closes intake or exhaust port when pushed by lifter and rocker
related: camshaft.md, hydraulic_lifter.md, valve_spring.md
---
# Engine Valve Stem

## Description
A stainless steel poppet valve with a tulip-shaped head, long stem, keeper groove, and spring retainer groove. The head seals against the valve seat at 45 degrees. Intake valves are larger diameter than exhaust.

## Keywords
valve stem, poppet valve, valve head, keeper groove, spring retainer, valve seat, 45 degree seat, intake valve, exhaust valve, revolve, draw, fuse, cut, stainless steel

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| HEAD_R | 22 | mm | valve head radius |
| STEM_R | 4 | mm | stem radius |
| STEM_LENGTH | 100 | mm | total stem length |
| SEAT_ANGLE_H | 6 | mm | height of 45deg seat face |
| KEEPER_GROOVE_Z | 10 | mm | keeper groove from stem tip |
| KEEPER_GROOVE_D | 1 | mm | keeper groove depth |
| KEEPER_GROOVE_W | 3 | mm | keeper groove width |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const HEAD_R         = 22;
  const STEM_R         = 4;
  const STEM_LENGTH    = 100;
  const SEAT_ANGLE_H   = 6;
  const KEEPER_Z       = STEM_LENGTH - 10;
  const KEEPER_GROOVE_D = 1;
  const KEEPER_GROOVE_W = 3;

  // Valve profile — head + seat taper + stem (revolved)
  const profile = draw([0, 0])
    .hLine(HEAD_R)
    .lineTo([STEM_R + 2, -SEAT_ANGLE_H])
    .lineTo([STEM_R, -SEAT_ANGLE_H - 3])
    .vLine(-STEM_LENGTH + SEAT_ANGLE_H + 3)
    .hLine(-STEM_R)
    .close();

  let valve = profile.sketchOnPlane("XZ").revolve();

  // Keeper groove near stem tip
  const keeperProfile = draw([STEM_R - KEEPER_GROOVE_D, KEEPER_Z])
    .hLine(KEEPER_GROOVE_D)
    .vLine(KEEPER_GROOVE_W)
    .hLine(-KEEPER_GROOVE_D)
    .close();
  const keeper = keeperProfile.sketchOnPlane("XZ").revolve();
  valve = valve.cut(keeper);

  return { shape: valve, name: "Valve Stem", color: "silver" };
};
```
