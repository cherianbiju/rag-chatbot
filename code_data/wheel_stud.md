---
source_file: wheel_stud.js
category: wheel_hub
type: annotated_code
use_case: secures wheel to hub flange via lug nuts, transmitting braking and acceleration torque
related: hub_flange.md, bearing_race.md, abs_tone_ring.md
---
# Wheel Stud

## Description
A hardened steel press-fit wheel stud with knurled press shoulder, plain shank, and threaded end for lug nut. The knurled section grips the hub flange bore to resist rotation under torque loads.

## Keywords
wheel stud, lug bolt, press fit, knurl, threaded stud, lug nut, wheel fastener, hub, revolve, draw, cylinder, fuse, cut, hardened steel

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| THREAD_R | 7 | mm | threaded end radius (M14) |
| SHANK_R | 7 | mm | plain shank radius |
| KNURL_R | 8 | mm | knurl press section radius |
| THREAD_LENGTH | 28 | mm | exposed thread length |
| SHANK_LENGTH | 15 | mm | plain shank length |
| KNURL_LENGTH | 12 | mm | knurl section length |
| HEAD_R | 11 | mm | press head radius |
| HEAD_THICK | 6 | mm | press head thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
  } = replicad;

  const THREAD_R      = 7;
  const KNURL_R       = 8;
  const THREAD_LENGTH = 28;
  const SHANK_LENGTH  = 15;
  const KNURL_LENGTH  = 12;
  const HEAD_R        = 11;
  const HEAD_THICK    = 6;

  // Full stepped profile revolved
  const profile = draw([0, 0])
    .hLine(HEAD_R)
    .vLine(HEAD_THICK)
    .hLine(-(HEAD_R - KNURL_R))
    .vLine(KNURL_LENGTH)
    .hLine(-(KNURL_R - THREAD_R))
    .vLine(SHANK_LENGTH)
    .vLine(THREAD_LENGTH)
    .hLine(-THREAD_R)
    .close();

  const stud = profile.sketchOnPlane("XZ").revolve();

  return { shape: stud, name: "Wheel Stud", color: "steelblue" };
};
```
