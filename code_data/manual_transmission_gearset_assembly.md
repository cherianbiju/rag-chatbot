---
source_file: manual_transmission_gearset_assembly.md
category: assembly
type: annotated_code
use_case: manual gearbox speed and torque multiplication, 5-speed layshaft transmission
related: spur_gear.md, shaft_design.md, differential_assembly.md
---

# 5-Speed Manual Transmission Gearset Assembly

## Description
A layshaft (countershaft) 5-speed manual transmission gearset consisting of an input shaft gear, five countershaft gears of varying sizes, and five output shaft gears. Gear pairs mesh to provide five forward speed ratios. Each gear is a simplified spur gear with hub and tooth profile represented as a polyside cylinder. The input shaft drives the countershaft, which meshes with output gears.

## Keywords
manual transmission, gearset, layshaft, countershaft, spur gear, gear ratio, input shaft, output shaft, gear mesh, 5-speed, synchromesh, gear hub, tooth profile, speed reduction, torque multiplication, gearbox

## Parameters
| Variable           | Value | Unit | Meaning                          |
|--------------------|-------|------|----------------------------------|
| shaftRadius        | 15    | mm   | Main shaft radius                |
| shaftLength        | 320   | mm   | Shaft total length               |
| gear1Radius        | 55    | mm   | 1st gear (largest) radius        |
| gear2Radius        | 48    | mm   | 2nd gear radius                  |
| gear3Radius        | 40    | mm   | 3rd gear radius                  |
| gear4Radius        | 33    | mm   | 4th gear radius                  |
| gear5Radius        | 27    | mm   | 5th gear (smallest) radius       |
| gearWidth          | 24    | mm   | Gear face width                  |
| hubRadius          | 20    | mm   | Gear hub radius                  |
| toothDepth         | 5     | mm   | Gear tooth depth (addendum)      |
| gearSpacing        | 40    | mm   | Centre-to-centre gear spacing    |
| counterShaftOffset | 100   | mm   | Countershaft Y-axis offset       |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides } = replicad;

  const shaftRadius        = 15;
  const shaftLength        = 320;
  const gear1Radius        = 55;
  const gear2Radius        = 48;
  const gear3Radius        = 40;
  const gear4Radius        = 33;
  const gear5Radius        = 27;
  const gearWidth          = 24;
  const hubRadius          = 20;
  const toothDepth         = 5;
  const gearSpacing        = 40;
  const counterShaftOffset = 100;

  const makeGear = (radius, zOffset) => {
    const teeth = drawPolysides(radius + toothDepth, 20)
      .sketchOnPlane("XY", zOffset)
      .extrude(gearWidth);
    const body = drawCircle(radius)
      .sketchOnPlane("XY", zOffset)
      .extrude(gearWidth);
    const hub = drawCircle(hubRadius)
      .sketchOnPlane("XY", zOffset)
      .extrude(gearWidth);
    const bore = drawCircle(shaftRadius)
      .sketchOnPlane("XY", zOffset)
      .extrude(gearWidth);
    return teeth.intersect(body).fuse(hub).cut(bore);
  };

  // ── INPUT / OUTPUT SHAFT ──────────────────────────────────
  const mainShaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength);

  // ── OUTPUT SHAFT GEARS (5 gears) ─────────────────────────
  const outGear1 = makeGear(gear1Radius, gearSpacing * 0);
  const outGear2 = makeGear(gear2Radius, gearSpacing * 1);
  const outGear3 = makeGear(gear3Radius, gearSpacing * 2);
  const outGear4 = makeGear(gear4Radius, gearSpacing * 3);
  const outGear5 = makeGear(gear5Radius, gearSpacing * 4);

  // ── COUNTERSHAFT ──────────────────────────────────────────
  const counterShaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength)
    .translateY(counterShaftOffset);

  // Countershaft gears mesh with output — mirrored ratios
  const ctrGear1 = makeGear(gear5Radius, gearSpacing * 0).translateY(counterShaftOffset);
  const ctrGear2 = makeGear(gear4Radius, gearSpacing * 1).translateY(counterShaftOffset);
  const ctrGear3 = makeGear(gear3Radius, gearSpacing * 2).translateY(counterShaftOffset);
  const ctrGear4 = makeGear(gear2Radius, gearSpacing * 3).translateY(counterShaftOffset);
  const ctrGear5 = makeGear(gear1Radius, gearSpacing * 4).translateY(counterShaftOffset);

  return [
    { shape: mainShaft,   name: "Main Shaft",        color: "#708090" },
    { shape: outGear1,    name: "Output Gear 1st",   color: "#CD853F" },
    { shape: outGear2,    name: "Output Gear 2nd",   color: "#CD853F" },
    { shape: outGear3,    name: "Output Gear 3rd",   color: "#CD853F" },
    { shape: outGear4,    name: "Output Gear 4th",   color: "#CD853F" },
    { shape: outGear5,    name: "Output Gear 5th",   color: "#CD853F" },
    { shape: counterShaft,name: "Counter Shaft",     color: "#607080" },
    { shape: ctrGear1,    name: "Counter Gear 1st",  color: "#8B6914" },
    { shape: ctrGear2,    name: "Counter Gear 2nd",  color: "#8B6914" },
    { shape: ctrGear3,    name: "Counter Gear 3rd",  color: "#8B6914" },
    { shape: ctrGear4,    name: "Counter Gear 4th",  color: "#8B6914" },
    { shape: ctrGear5,    name: "Counter Gear 5th",  color: "#8B6914" },
  ];
};
```
