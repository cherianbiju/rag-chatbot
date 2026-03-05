---
source_file: diff_carrier.js
category: differential
type: annotated_code
use_case: houses spider gears and mounts ring gear, rotating as a unit inside differential housing
related: ring_gear.md, spider_gear.md, bevel_pinion.md
---
# Differential Carrier

## Description
A cast steel carrier (cage) that holds the spider pin and spider gears. The ring gear bolts to its flange. Carrier rotates on bearings inside the axle housing and transmits torque to both axle side gears.

## Keywords
differential carrier, diff cage, ring gear flange, spider pin, bearing journal, axle housing, revolve, draw, fuse, cut, cylinder, bolt flange, open differential

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| CARRIER_OUTER_R | 95 | mm | outer radius of carrier body |
| CARRIER_INNER_R | 60 | mm | inner cavity radius |
| CARRIER_HEIGHT | 80 | mm | total height |
| FLANGE_R | 105 | mm | ring gear bolt flange radius |
| FLANGE_THICK | 14 | mm | flange thickness |
| JOURNAL_R | 28 | mm | bearing journal radius |
| JOURNAL_LENGTH | 30 | mm | bearing journal length |
| AXLE_BORE_R | 22 | mm | axle shaft bore radius |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const CARRIER_OUTER_R = 95;
  const CARRIER_INNER_R = 60;
  const CARRIER_HEIGHT  = 80;
  const FLANGE_R        = 105;
  const FLANGE_THICK    = 14;
  const JOURNAL_R       = 28;
  const JOURNAL_LENGTH  = 30;
  const AXLE_BORE_R     = 22;

  // Main carrier body
  const bodyProfile = draw([AXLE_BORE_R, 0])
    .hLine(CARRIER_OUTER_R - AXLE_BORE_R)
    .vLine(CARRIER_HEIGHT)
    .hLine(-(CARRIER_OUTER_R - AXLE_BORE_R))
    .close();
  let carrier = bodyProfile.sketchOnPlane("XZ").revolve();

  // Hollow interior cavity
  const cavityProfile = draw([AXLE_BORE_R + 8, 10])
    .hLine(CARRIER_INNER_R - AXLE_BORE_R - 8)
    .vLine(CARRIER_HEIGHT - 20)
    .hLine(-(CARRIER_INNER_R - AXLE_BORE_R - 8))
    .close();
  const cavity = cavityProfile.sketchOnPlane("XZ").revolve();
  carrier = carrier.cut(cavity);

  // Ring gear bolt flange
  const flangeProfile = draw([CARRIER_OUTER_R, CARRIER_HEIGHT - FLANGE_THICK])
    .hLine(FLANGE_R - CARRIER_OUTER_R)
    .vLine(FLANGE_THICK)
    .hLine(-(FLANGE_R - CARRIER_OUTER_R))
    .close();
  const flange = flangeProfile.sketchOnPlane("XZ").revolve();
  carrier = carrier.fuse(flange);

  // Bearing journals both sides
  const journal1 = makeCylinder(JOURNAL_R, JOURNAL_LENGTH, [0, 0, -JOURNAL_LENGTH], [0, 0, 1]);
  const journal2 = makeCylinder(JOURNAL_R, JOURNAL_LENGTH, [0, 0, CARRIER_HEIGHT], [0, 0, 1]);
  carrier = carrier.fuse(journal1).fuse(journal2);

  // Axle bore through both journals
  const axleBore = makeCylinder(AXLE_BORE_R, CARRIER_HEIGHT + JOURNAL_LENGTH * 2 + 4, [0, 0, -JOURNAL_LENGTH - 2], [0, 0, 1]);
  carrier = carrier.cut(axleBore);

  return { shape: carrier, name: "Differential Carrier", color: "dimgrey" };
};
```
