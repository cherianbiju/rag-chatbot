---
source_file: bushing.js
category: suspension
type: annotated_code
use_case: isolates vibration and allows controlled rotation at suspension pivot points
related: control_arm.md, ball_joint.md
---
# Suspension Bushing

## Description
A polyurethane cylindrical bushing press-fit into control arm tubes. The inner steel sleeve bonds to the subframe bolt while the outer sleeve bonds to the arm, with the polyurethane allowing controlled deflection.

## Keywords
bushing, polyurethane, press fit, vibration isolation, pivot, suspension, inner sleeve, outer sleeve, revolve, cylinder, bore, annular, rubber mount

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| OUTER_R | 18 | mm | outer radius (press fit into arm) |
| INNER_R | 6 | mm | inner bore radius for bolt |
| LENGTH | 40 | mm | bushing length |
| FLANGE_R | 22 | mm | end flange radius |
| FLANGE_THICK | 4 | mm | end flange thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
  } = replicad;

  const OUTER_R      = 18;
  const INNER_R      = 6;
  const LENGTH       = 40;
  const FLANGE_R     = 22;
  const FLANGE_THICK = 4;

  // Bushing body profile (revolved)
  const profile = draw([INNER_R, 0])
    .hLine(OUTER_R - INNER_R)
    .vLine(LENGTH)
    .hLine(-(OUTER_R - INNER_R))
    .close();

  let bushing = profile.sketchOnPlane("XZ").revolve();

  // End flange bottom
  const flange1Profile = draw([INNER_R, 0])
    .hLine(FLANGE_R - INNER_R)
    .vLine(-FLANGE_THICK)
    .hLine(-(FLANGE_R - INNER_R))
    .close();
  const flange1 = flange1Profile.sketchOnPlane("XZ").revolve();
  bushing = bushing.fuse(flange1);

  // End flange top
  const flange2Profile = draw([INNER_R, LENGTH])
    .hLine(FLANGE_R - INNER_R)
    .vLine(FLANGE_THICK)
    .hLine(-(FLANGE_R - INNER_R))
    .close();
  const flange2 = flange2Profile.sketchOnPlane("XZ").revolve();
  bushing = bushing.fuse(flange2);

  return { shape: bushing, name: "Suspension Bushing", color: "dimgrey" };
};
```
