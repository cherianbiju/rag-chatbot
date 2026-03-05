---
source_file: turbo_shaft.js
category: turbocharger
type: annotated_code
use_case: connects turbine and compressor wheels, transmitting rotational energy at high speed
related: turbine_housing.md, compressor_wheel.md
---
# Turbocharger Shaft

## Description
A precision-ground steel shaft connecting turbine wheel to compressor wheel. Runs on floating sleeve bearings fed by engine oil. Includes oil feed groove, thrust collar, and threaded compressor nut end.

## Keywords
turbo shaft, turbine shaft, compressor shaft, floating bearing, oil feed, thrust collar, high speed, revolve, draw, fuse, cut, cylinder, stepped shaft

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| SHAFT_TOTAL_L | 120 | mm | total shaft length |
| MAIN_R | 6 | mm | main journal radius |
| THRUST_COLLAR_R | 12 | mm | thrust collar radius |
| THRUST_COLLAR_W | 8 | mm | thrust collar width |
| TURBINE_END_R | 9 | mm | turbine end stub radius |
| COMPRESSOR_END_R | 5 | mm | compressor threaded end radius |
| OIL_GROOVE_D | 1 | mm | oil groove depth |
| OIL_GROOVE_W | 3 | mm | oil groove width |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const SHAFT_TOTAL_L    = 120;
  const MAIN_R           = 6;
  const THRUST_COLLAR_R  = 12;
  const THRUST_COLLAR_W  = 8;
  const TURBINE_END_R    = 9;
  const COMPRESSOR_END_R = 5;
  const OIL_GROOVE_D     = 1;
  const OIL_GROOVE_W     = 3;

  // Main shaft stepped profile
  const profile = draw([0, 0])
    .hLine(TURBINE_END_R)         .vLine(15)
    .hLine(-(TURBINE_END_R - MAIN_R)).vLine(SHAFT_TOTAL_L * 0.3)
    .hLine(THRUST_COLLAR_R - MAIN_R).vLine(THRUST_COLLAR_W)
    .hLine(-(THRUST_COLLAR_R - MAIN_R)).vLine(SHAFT_TOTAL_L * 0.4)
    .hLine(-(MAIN_R - COMPRESSOR_END_R)).vLine(SHAFT_TOTAL_L * 0.2)
    .hLine(-COMPRESSOR_END_R)
    .close();

  let shaft = profile.sketchOnPlane("XZ").revolve();

  // Oil feed groove on main journal
  const grooveProfile = draw([MAIN_R - OIL_GROOVE_D, SHAFT_TOTAL_L * 0.45])
    .hLine(OIL_GROOVE_D)
    .vLine(OIL_GROOVE_W)
    .hLine(-OIL_GROOVE_D)
    .close();
  const groove = grooveProfile.sketchOnPlane("XZ").revolve();
  shaft = shaft.cut(groove);

  return { shape: shaft, name: "Turbocharger Shaft", color: "steelblue" };
};
```
