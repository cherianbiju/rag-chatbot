---
source_file: rack_pinion_steering_assembly.md
category: assembly
type: annotated_code
use_case: converts steering wheel rotary motion to linear rack movement for wheel steering
related: suspension_control_arm_assembly.md, tie_rod.md, steering_column.md
---

# Rack-and-Pinion Steering Assembly

## Description
A rack-and-pinion steering assembly consisting of a toothed rack bar that slides linearly inside a steering housing tube, driven by a pinion gear connected to the steering column. Tie rod ends at each rack end connect to the steering knuckles. Rotation of the pinion translates to lateral rack movement, turning both front wheels simultaneously via the tie rods.

## Keywords
rack and pinion, steering rack, pinion gear, tie rod, steering housing, linear actuator, steering knuckle, front wheel steering, rack teeth, pinion teeth, steering ratio, lateral force, automotive steering, column input, rack travel

## Parameters
| Variable         | Value | Unit | Meaning                          |
|------------------|-------|------|---------------------------------|
| rackLength       | 500   | mm   | Total rack bar length           |
| rackWidth        | 20    | mm   | Rack bar width                  |
| rackHeight       | 20    | mm   | Rack bar height                 |
| toothHeight      | 5     | mm   | Rack tooth height               |
| toothCount       | 24   | -    | Number of rack teeth            |
| pinionRadius     | 28    | mm   | Pinion gear pitch radius        |
| pinionHeight     | 22    | mm   | Pinion gear face width          |
| pinionTeeth      | 12    | -    | Number of pinion teeth          |
| housingRadius    | 22    | mm   | Steering housing tube radius    |
| housingLength    | 380   | mm   | Steering housing tube length    |
| tieRodRadius     | 8     | mm   | Tie rod radius                  |
| tieRodLength     | 120   | mm   | Tie rod length                  |
| tieRodEndRadius  | 14    | mm   | Tie rod end ball housing radius |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle, drawPolysides } = replicad;

  const rackLength      = 500;
  const rackWidth       = 20;
  const rackHeight      = 20;
  const toothHeight     = 5;
  const toothCount      = 24;
  const pinionRadius    = 28;
  const pinionHeight    = 22;
  const pinionTeeth     = 12;
  const housingRadius   = 22;
  const housingLength   = 380;
  const tieRodRadius    = 8;
  const tieRodLength    = 120;
  const tieRodEndRadius = 14;

  // ── RACK BAR ──────────────────────────────────────────────
  const rackBody = drawRectangle(rackLength, rackHeight)
    .sketchOnPlane("XZ", 0)
    .extrude(rackWidth)
    .translateZ(-rackHeight / 2)
    .translateY(-rackWidth / 2);

  // Teeth on top surface (simplified as polyside extrusions)
  const toothSpacing = rackLength / toothCount;
  let rack = rackBody;
  for (let i = 0; i < toothCount; i++) {
    const tooth = drawRectangle(toothSpacing * 0.6, toothHeight)
      .sketchOnPlane("XZ", rackWidth)
      .extrude(rackWidth * 0.8)
      .translateX(-rackLength / 2 + toothSpacing * i + toothSpacing * 0.2)
      .translateZ(-toothHeight)
      .translateY(-rackWidth * 0.9);
    rack = rack.fuse(tooth);
  }

  // ── PINION GEAR ───────────────────────────────────────────
  const pinionTeethShape = drawPolysides(pinionRadius + 4, pinionTeeth)
    .sketchOnPlane("XY", 0)
    .extrude(pinionHeight);

  const pinionBody = drawCircle(pinionRadius)
    .sketchOnPlane("XY", 0)
    .extrude(pinionHeight);

  const pinionShaftBore = drawCircle(10)
    .sketchOnPlane("XY", 0)
    .extrude(pinionHeight);

  const pinion = pinionTeethShape.intersect(pinionBody).cut(pinionShaftBore)
    .translateY(-pinionRadius - rackHeight / 2)
    .translateZ(-pinionHeight / 2);

  // ── STEERING HOUSING TUBE ─────────────────────────────────
  const housingOuter = drawCircle(housingRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(housingLength)
    .translateX(-housingLength / 2);

  const housingBore = drawCircle(housingRadius - 4)
    .sketchOnPlane("XZ", 0)
    .extrude(housingLength)
    .translateX(-housingLength / 2);

  const housing = housingOuter.cut(housingBore);

  // ── TIE RODS ──────────────────────────────────────────────
  const tieRodLeft = drawCircle(tieRodRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(tieRodLength)
    .translateX(-rackLength / 2 - tieRodLength);

  const tieRodRight = drawCircle(tieRodRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(tieRodLength)
    .translateX(rackLength / 2);

  // Tie rod ends (ball housing)
  const tieEndLeft = drawCircle(tieRodEndRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(tieRodEndRadius * 2)
    .translateX(-rackLength / 2 - tieRodLength - tieRodEndRadius)
    .translateZ(-tieRodEndRadius);

  const tieEndRight = drawCircle(tieRodEndRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(tieRodEndRadius * 2)
    .translateX(rackLength / 2 + tieRodLength)
    .translateZ(-tieRodEndRadius);

  return [
    { shape: rack,         name: "Steering Rack",   color: "#A9A9A9" },
    { shape: pinion,       name: "Pinion Gear",     color: "#B8860B" },
    { shape: housing,      name: "Rack Housing",    color: "#2F4F4F" },
    { shape: tieRodLeft,   name: "Tie Rod Left",    color: "#808080" },
    { shape: tieRodRight,  name: "Tie Rod Right",   color: "#808080" },
    { shape: tieEndLeft,   name: "Tie End Left",    color: "#696969" },
    { shape: tieEndRight,  name: "Tie End Right",   color: "#696969" },
  ];
};
```
