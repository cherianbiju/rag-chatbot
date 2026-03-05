---
source_file: hydraulic_lifter.js
category: engine
type: annotated_code
use_case: transfers cam lobe motion to pushrods or valves, self-adjusting via oil pressure
related: camshaft.md, valve_stem.md, valve_spring.md
---
# Hydraulic Lifter / Tappet

## Description
A cylindrical hydraulic valve lifter that rides in the block bore on the cam lobe. Oil pressure inside auto-adjusts valve lash to zero. The flat-bottom face rides directly on the cam lobe surface.

## Keywords
hydraulic lifter, tappet, cam follower, valve lash, oil pressure, lifter bore, flat bottom, cylinder, revolve, draw, cut, fuse, engine, block bore

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| LIFTER_R | 12.5 | mm | lifter outer radius |
| LIFTER_HEIGHT | 50 | mm | total lifter height |
| OIL_FEED_R | 2.5 | mm | oil feed hole radius |
| OIL_GROOVE_DEPTH | 1.5 | mm | oil groove depth |
| OIL_GROOVE_WIDTH | 4 | mm | oil groove width |
| INNER_PLUNGER_R | 9 | mm | inner plunger bore radius |
| PLUNGER_DEPTH | 30 | mm | plunger bore depth |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const LIFTER_R       = 12.5;
  const LIFTER_HEIGHT  = 50;
  const OIL_FEED_R     = 2.5;
  const OIL_GROOVE_D   = 1.5;
  const OIL_GROOVE_W   = 4;
  const INNER_R        = 9;
  const PLUNGER_DEPTH  = 30;

  // Outer body
  const bodyProfile = draw([0, 0])
    .hLine(LIFTER_R)
    .vLine(LIFTER_HEIGHT)
    .hLine(-LIFTER_R)
    .close();
  let lifter = bodyProfile.sketchOnPlane("XZ").revolve();

  // Oil groove around circumference
  const grooveProfile = draw([LIFTER_R - OIL_GROOVE_D, LIFTER_HEIGHT * 0.4])
    .hLine(OIL_GROOVE_D)
    .vLine(OIL_GROOVE_W)
    .hLine(-OIL_GROOVE_D)
    .close();
  const groove = grooveProfile.sketchOnPlane("XZ").revolve();
  lifter = lifter.cut(groove);

  // Oil feed hole
  const oilHole = makeCylinder(OIL_FEED_R, LIFTER_R + 2, [-(LIFTER_R + 1), 0, LIFTER_HEIGHT * 0.42], [1, 0, 0]);
  lifter = lifter.cut(oilHole);

  // Inner plunger bore
  const plungerBore = makeCylinder(INNER_R, PLUNGER_DEPTH, [0, 0, LIFTER_HEIGHT - PLUNGER_DEPTH], [0, 0, 1]);
  lifter = lifter.cut(plungerBore);

  return { shape: lifter, name: "Hydraulic Lifter", color: "steelblue" };
};
```
