---
source_file: compressor_wheel.js
category: turbocharger
type: annotated_code
use_case: compresses intake air by spinning at high speed driven by turbine shaft
related: turbine_housing.md, turbo_shaft.md
---
# Turbocharger Compressor Wheel

## Description
A billet aluminum compressor wheel with 6 full blades and 6 splitter blades. The inducer (inlet) end is smaller and the exducer (outlet) is larger. Mounted on the turbo shaft via central bore and nut.

## Keywords
compressor wheel, turbocharger, inducer, exducer, full blade, splitter blade, billet aluminum, impeller, revolve, draw, fuse, cut, cylinder, turbine shaft, boost

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| INDUCER_R | 22.5 | mm | inducer (inlet) radius |
| EXDUCER_R | 40 | mm | exducer (outlet) radius |
| WHEEL_HEIGHT | 45 | mm | axial height of wheel |
| HUB_R | 12 | mm | hub bore radius |
| NUM_FULL_BLADES | 6 | — | number of full blades |
| NUM_SPLITTERS | 6 | — | number of splitter blades |
| BLADE_THICK | 2 | mm | blade thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const INDUCER_R      = 22.5;
  const EXDUCER_R      = 40;
  const WHEEL_HEIGHT   = 45;
  const HUB_R          = 12;
  const NUM_FULL_BLADES = 6;
  const NUM_SPLITTERS   = 6;
  const BLADE_THICK     = 2;

  // Hub cone body
  const hubProfile = draw([0, 0])
    .lineTo([INDUCER_R, WHEEL_HEIGHT])
    .hLine(-INDUCER_R)
    .close();
  let wheel = hubProfile.sketchOnPlane("XZ").revolve();

  // Exducer disc base
  const exducerProfile = draw([0, 0])
    .hLine(EXDUCER_R)
    .vLine(8)
    .hLine(-EXDUCER_R)
    .close();
  const exducer = exducerProfile.sketchOnPlane("XZ").revolve();
  wheel = wheel.fuse(exducer);

  // Full blades — 6 evenly spaced
  for (let i = 0; i < NUM_FULL_BLADES; i++) {
    const angle = (i / NUM_FULL_BLADES) * 360;
    const bladeProfile = draw([HUB_R + 2, 0])
      .lineTo([EXDUCER_R - 5, 8])
      .vLine(BLADE_THICK)
      .lineTo([HUB_R + 2, BLADE_THICK])
      .close();
    const blade = bladeProfile.sketchOnPlane("XZ")
      .revolve([0,0,0],[0,0,1], 360 / NUM_FULL_BLADES * 0.4)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    wheel = wheel.fuse(blade);
  }

  // Splitter blades — offset between full blades, start halfway up
  for (let i = 0; i < NUM_SPLITTERS; i++) {
    const angle = (i / NUM_SPLITTERS) * 360 + (360 / NUM_FULL_BLADES / 2);
    const splitProfile = draw([INDUCER_R * 0.6, WHEEL_HEIGHT * 0.4])
      .lineTo([EXDUCER_R - 6, 8])
      .vLine(BLADE_THICK)
      .lineTo([INDUCER_R * 0.6, WHEEL_HEIGHT * 0.4 + BLADE_THICK])
      .close();
    const splitter = splitProfile.sketchOnPlane("XZ")
      .revolve([0,0,0],[0,0,1], 360 / NUM_SPLITTERS * 0.35)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    wheel = wheel.fuse(splitter);
  }

  // Central shaft bore
  const bore = makeCylinder(HUB_R, WHEEL_HEIGHT + 10, [0, 0, -1], [0, 0, 1]);
  wheel = wheel.cut(bore);

  return { shape: wheel, name: "Compressor Wheel", color: "silver" };
};
```
