---
source_file: differential_bevel_gear_assembly.md
category: assembly
type: annotated_code
use_case: torque distribution between two output shafts allowing speed differentiation during cornering
related: manual_transmission_gearset_assembly.md, axle_shaft.md, ring_gear.md
---

# Open Differential Bevel Gear Assembly

## Description
An open differential consisting of a ring gear, differential case, two side bevel gears (sun gears) on the output axles, and two spider bevel pinion gears on the cross-pin. The ring gear receives torque from the driveshaft pinion, the case rotates and drives the spider pinions, which in turn drive the side gears. During straight driving all gears turn together; during cornering the spider pinions rotate allowing the outer wheel to turn faster.

## Keywords
differential, bevel gear, ring gear, spider gear, side gear, sun gear, pinion, differential case, open differential, axle, torque split, cornering, cross pin, drive axle, gear mesh, automotive drivetrain

## Parameters
| Variable         | Value | Unit | Meaning                         |
|------------------|-------|------|---------------------------------|
| caseRadius       | 90    | mm   | Differential case outer radius  |
| caseHeight       | 80    | mm   | Differential case height        |
| ringGearRadius   | 85    | mm   | Ring gear outer radius          |
| ringGearWidth    | 22    | mm   | Ring gear face width            |
| sideGearRadius   | 38    | mm   | Side bevel gear radius          |
| sideGearHeight   | 25    | mm   | Side bevel gear height          |
| spiderGearRadius | 28    | mm   | Spider pinion gear radius       |
| spiderGearHeight | 22    | mm   | Spider pinion height            |
| axleRadius       | 18    | mm   | Output axle shaft radius        |
| axleLength       | 120   | mm   | Output axle stub length         |
| crossPinRadius   | 10    | mm   | Cross pin radius                |
| toothDepth       | 4     | mm   | Bevel gear tooth depth          |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawPolysides } = replicad;

  const caseRadius       = 90;
  const caseHeight       = 80;
  const ringGearRadius   = 85;
  const ringGearWidth    = 22;
  const sideGearRadius   = 38;
  const sideGearHeight   = 25;
  const spiderGearRadius = 28;
  const spiderGearHeight = 22;
  const axleRadius       = 18;
  const axleLength       = 120;
  const crossPinRadius   = 10;
  const toothDepth       = 4;

  // ── DIFFERENTIAL CASE ─────────────────────────────────────
  const caseOuter = drawCircle(caseRadius)
    .sketchOnPlane("XY", 0)
    .extrude(caseHeight);

  const caseInner = drawCircle(caseRadius - 12)
    .sketchOnPlane("XY", 10)
    .extrude(caseHeight - 20);

  const axleBoreLeft = drawCircle(axleRadius + 2)
    .sketchOnPlane("XY", 0)
    .extrude(caseHeight);

  const diffCase = caseOuter.cut(caseInner).cut(axleBoreLeft);

  // ── RING GEAR ─────────────────────────────────────────────
  const ringTeeth = drawPolysides(ringGearRadius + toothDepth, 36)
    .sketchOnPlane("XY", caseHeight)
    .extrude(ringGearWidth);

  const ringBody = drawCircle(ringGearRadius)
    .sketchOnPlane("XY", caseHeight)
    .extrude(ringGearWidth);

  const ringBore = drawCircle(caseRadius - 5)
    .sketchOnPlane("XY", caseHeight)
    .extrude(ringGearWidth);

  const ringGear = ringTeeth.intersect(ringBody).cut(ringBore);

  // ── SIDE BEVEL GEAR (left axle) ───────────────────────────
  const sideGearTeeth = drawPolysides(sideGearRadius + toothDepth, 16)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);

  const sideGearBody = drawCircle(sideGearRadius)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);

  const sideGearBore = drawCircle(axleRadius)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);

  const sideGearLeft = sideGearTeeth.intersect(sideGearBody).cut(sideGearBore)
    .translateZ(caseHeight * 0.15);

  const sideGearRight = sideGearLeft.clone()
    .translateZ(caseHeight * 0.55);

  // ── SPIDER PINION GEARS ───────────────────────────────────
  const spiderTeeth = drawPolysides(spiderGearRadius + toothDepth, 12)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderGearHeight);

  const spiderBody = drawCircle(spiderGearRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderGearHeight);

  const spiderBore = drawCircle(crossPinRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderGearHeight);

  const spider1 = spiderTeeth.intersect(spiderBody).cut(spiderBore)
    .translateY(-spiderGearHeight / 2)
    .translateZ(caseHeight / 2);

  const spider2 = spider1.clone().rotate(90, [0, 0, caseHeight / 2], [0, 0, 1]);

  // ── OUTPUT AXLES ──────────────────────────────────────────
  const axleLeft = drawCircle(axleRadius)
    .sketchOnPlane("XY", -axleLength)
    .extrude(axleLength);

  const axleRight = drawCircle(axleRadius)
    .sketchOnPlane("XY", caseHeight)
    .extrude(axleLength);

  return [
    { shape: diffCase,    name: "Differential Case", color: "#696969" },
    { shape: ringGear,    name: "Ring Gear",          color: "#B8860B" },
    { shape: sideGearLeft,  name: "Side Gear Left",  color: "#CD853F" },
    { shape: sideGearRight, name: "Side Gear Right", color: "#CD853F" },
    { shape: spider1,     name: "Spider Pinion 1",   color: "#8B4513" },
    { shape: spider2,     name: "Spider Pinion 2",   color: "#8B4513" },
    { shape: axleLeft,    name: "Axle Left",         color: "#778899" },
    { shape: axleRight,   name: "Axle Right",        color: "#778899" },
  ];
};
```
