---
source_file: bearing_race.js
category: wheel_hub
type: annotated_code
use_case: provides hardened rolling surface for tapered roller bearing in wheel hub assembly
related: hub_flange.md, abs_tone_ring.md, wheel_stud.md
---
# Wheel Bearing Race Ring

## Description
A hardened steel tapered roller bearing outer race press-fit into the steering knuckle. The tapered inner surface guides the rollers that carry axial and radial wheel loads. Heat-treated for wear resistance.

## Keywords
bearing race, outer race, tapered roller bearing, wheel bearing, press fit, knuckle, hardened steel, revolve, draw, taper, axial load, radial load, hub bearing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| OUTER_R | 40 | mm | outer press-fit radius |
| INNER_R_SMALL | 30 | mm | inner bore radius small end |
| INNER_R_LARGE | 35 | mm | inner bore radius large end |
| RACE_LENGTH | 25 | mm | axial length of race |
| FLANGE_R | 45 | mm | retention flange radius |
| FLANGE_THICK | 4 | mm | retention flange thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
  } = replicad;

  const OUTER_R       = 40;
  const INNER_R_SMALL = 30;
  const INNER_R_LARGE = 35;
  const RACE_LENGTH   = 25;
  const FLANGE_R      = 45;
  const FLANGE_THICK  = 4;

  // Outer race body with tapered bore
  const profile = draw([INNER_R_SMALL, 0])
    .lineTo([INNER_R_LARGE, RACE_LENGTH])
    .hLine(OUTER_R - INNER_R_LARGE)
    .vLine(-RACE_LENGTH)
    .close();

  let race = profile.sketchOnPlane("XZ").revolve();

  // Retention flange at large end
  const flangeProfile = draw([OUTER_R, RACE_LENGTH])
    .hLine(FLANGE_R - OUTER_R)
    .vLine(FLANGE_THICK)
    .hLine(-(FLANGE_R - OUTER_R))
    .close();
  const flange = flangeProfile.sketchOnPlane("XZ").revolve();
  race = race.fuse(flange);

  return { shape: race, name: "Bearing Race", color: "dimgrey" };
};
```
