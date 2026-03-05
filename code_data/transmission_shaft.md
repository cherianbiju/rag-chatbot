---
source_file: transmission_shaft.js
category: transmission
type: annotated_code
use_case: carries gears and transmits torque through manual gearbox
related: helical_gear.md, synchro_hub.md
---
# Transmission Shaft

## Description
A hardened steel stepped shaft for a manual transmission. Includes bearing seats, gear seats, snap ring grooves, and splined end for synchro hub. The stepped diameters locate gears and bearings axially.

## Keywords
transmission shaft, stepped shaft, bearing seat, gear seat, spline, snap ring groove, keyway, revolve, draw, cylinder, extrude, hardened steel, gearbox

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| TOTAL_LENGTH | 280 | mm | total shaft length |
| MAIN_R | 18 | mm | main shaft radius |
| BEARING_SEAT_R | 20 | mm | bearing seat radius |
| GEAR_SEAT_R | 22 | mm | gear seat radius |
| SPLINE_R | 16 | mm | splined end radius |
| SNAP_GROOVE_DEPTH | 2 | mm | snap ring groove depth |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const TOTAL_LENGTH      = 280;
  const MAIN_R            = 18;
  const BEARING_SEAT_R    = 20;
  const GEAR_SEAT_R       = 22;
  const SPLINE_R          = 16;
  const SNAP_GROOVE_DEPTH = 2;
  const SNAP_GROOVE_WIDTH = 3;

  // Stepped shaft profile revolved around Z
  const profile = draw([0, 0])
    .hLine(BEARING_SEAT_R)        .vLine(25)     // front bearing seat
    .hLine(-(BEARING_SEAT_R - GEAR_SEAT_R))      .vLine(0)
    .hLine(GEAR_SEAT_R - MAIN_R - 2).vLine(0)
    .vLine(10)                                     // step to gear seat
    .hLine(GEAR_SEAT_R - MAIN_R)  .vLine(50)    // first gear seat
    .hLine(-(GEAR_SEAT_R - MAIN_R)).vLine(5)    // step down
    .vLine(100)                                    // main shaft
    .hLine(GEAR_SEAT_R - MAIN_R)  .vLine(50)    // second gear seat
    .hLine(-(GEAR_SEAT_R - MAIN_R)).vLine(5)
    .hLine(-(BEARING_SEAT_R - MAIN_R)).vLine(25) // rear bearing seat
    .hLine(-BEARING_SEAT_R)
    .close();

  let shaft = profile.sketchOnPlane("XZ").revolve();

  // Snap ring grooves
  const groove1 = draw([MAIN_R - SNAP_GROOVE_DEPTH, 22])
    .hLine(SNAP_GROOVE_DEPTH)
    .vLine(SNAP_GROOVE_WIDTH)
    .hLine(-SNAP_GROOVE_DEPTH)
    .close();
  const snap1 = groove1.sketchOnPlane("XZ").revolve();
  shaft = shaft.cut(snap1);

  const groove2 = draw([MAIN_R - SNAP_GROOVE_DEPTH, TOTAL_LENGTH - 22 - SNAP_GROOVE_WIDTH])
    .hLine(SNAP_GROOVE_DEPTH)
    .vLine(SNAP_GROOVE_WIDTH)
    .hLine(-SNAP_GROOVE_DEPTH)
    .close();
  const snap2 = groove2.sketchOnPlane("XZ").revolve();
  shaft = shaft.cut(snap2);

  return { shape: shaft, name: "Transmission Shaft", color: "steelblue" };
};
```
