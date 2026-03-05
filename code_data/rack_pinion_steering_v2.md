---
source_file: rack_pinion_steering_v2.md
category: assembly
type: annotated_code
use_case: power-assisted rack and pinion steering with 18T helical pinion, bellows, tie rod ball joints and steering stops
related: suspension_control_arm_v2.md
---

# Rack-and-Pinion Steering Assembly — Helical 18T / Power Assist

## Description
A power-assisted rack-and-pinion steering system with a ground steel rack bar (travel ±40 mm), 18-tooth helical pinion (carburized), power-assist housing bore, two tie-rod ends with ball joints, rubber bellows boots, and integrated steering stops. Universal coupling input flange at pinion top. Two M8 bracket mounts on the housing. Lubrication channels and steering stop bolt bosses modelled.

## Keywords
rack and pinion, helical pinion 18T, steering rack, power assist, tie rod ball joint, bellows boot, steering stop, rack travel 40mm, tooth backlash 0.08mm, carburized pinion, rack ground, M8 bracket mount, universal coupling, lubrication channel, rack housing, ball joint socket, steering ratio, power steering, front axle steering

## Parameters
| Variable            | Value  | Unit | Meaning                                |
|---------------------|--------|------|----------------------------------------|
| rackLength          | 520.0  | mm   | Total rack bar length                  |
| rackTravel          | 40.0   | mm   | Rack travel each side from centre      |
| rackWidth           | 22.0   | mm   | Rack bar width                         |
| rackHeight          | 22.0   | mm   | Rack bar height                        |
| toothHeight         | 4.5    | mm   | Rack tooth height                      |
| toothCount          | 26     | -    | Number of rack teeth                   |
| pinionTeeth         | 18     | -    | Pinion tooth count                     |
| pinionModule        | 2.5    | mm   | Pinion module                          |
| pinionFaceWidth     | 28.0   | mm   | Pinion face width                      |
| pinionShaftRadius   | 12.0   | mm   | Pinion input shaft radius              |
| pinionShaftLength   | 60.0   | mm   | Pinion shaft above housing             |
| couplingRadius      | 22.0   | mm   | Universal coupling flange radius       |
| housingRadius       | 28.0   | mm   | Rack housing tube radius               |
| housingLength       | 400.0  | mm   | Housing tube length                    |
| pasBoreRadius       | 18.0   | mm   | Power-assist cylinder bore radius      |
| tieRodRadius        | 9.0    | mm   | Tie rod body radius                    |
| tieRodLength        | 130.0  | mm   | Tie rod length                         |
| ballJointRadius     | 16.0   | mm   | Tie rod end ball joint housing radius  |
| bellowsRadius       | 32.0   | mm   | Bellows boot max radius                |
| bellowsLength       | 80.0   | mm   | Bellows boot length                    |
| mountBossRadius     | 12.0   | mm   | M8 bracket mount boss radius           |
| mountBoltRadius     | 4.0    | mm   | M8 bolt hole radius                    |
| stopBossRadius      | 10.0   | mm   | Steering stop boss radius              |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle, drawPolysides } = replicad;

  const rackLength        = 520.0;
  const rackWidth         = 22.0;
  const rackHeight        = 22.0;
  const toothHeight       = 4.5;
  const toothCount        = 26;
  const pinionTeeth       = 18;
  const pinionModule      = 2.5;
  const pinionFaceWidth   = 28.0;
  const pinionShaftRadius = 12.0;
  const pinionShaftLength = 60.0;
  const couplingRadius    = 22.0;
  const housingRadius     = 28.0;
  const housingLength     = 400.0;
  const pasBoreRadius     = 18.0;
  const tieRodRadius      = 9.0;
  const tieRodLength      = 130.0;
  const ballJointRadius   = 16.0;
  const bellowsRadius     = 32.0;
  const bellowsLength     = 80.0;
  const mountBossRadius   = 12.0;
  const mountBoltRadius   = 4.0;
  const stopBossRadius    = 10.0;
  const toothSpacing      = rackLength / toothCount;

  // ── RACK BAR ──────────────────────────────────────────────
  const rackBody = drawRectangle(rackLength, rackHeight)
    .sketchOnPlane("XZ", 0)
    .extrude(rackWidth)
    .translateX(-rackLength / 2)
    .translateZ(-rackHeight / 2)
    .translateY(-rackWidth / 2);

  // Rack teeth on top face
  let rack = rackBody;
  for (let i = 0; i < toothCount; i++) {
    const tx = -rackLength / 2 + toothSpacing * i + toothSpacing * 0.2;
    const tooth = drawRectangle(toothSpacing * 0.55, toothHeight)
      .sketchOnPlane("XZ", rackWidth)
      .extrude(rackWidth * 0.85)
      .translateX(tx)
      .translateZ(-toothHeight)
      .translateY(-rackWidth * 0.925);
    rack = rack.fuse(tooth);
  }

  // Oil lubrication groove along rack
  const lubGroove = drawRectangle(rackLength * 0.85, 2.5)
    .sketchOnPlane("XZ", rackWidth / 2)
    .extrude(1.5)
    .translateX(-rackLength * 0.425)
    .translateZ(-rackHeight * 0.4)
    .translateY(-rackWidth / 2 - 1.5);
  rack = rack.fuse(lubGroove);

  // Steering stop bosses at travel limits
  const stopLeft = drawCircle(stopBossRadius)
    .sketchOnPlane("XZ", rackWidth)
    .extrude(stopBossRadius)
    .translateX(-rackLength / 2 + 30)
    .translateZ(-rackHeight / 4);
  const stopRight = stopLeft.clone().translateX(rackLength - 60);
  rack = rack.fuse(stopLeft).fuse(stopRight);

  // ── HELICAL PINION GEAR ───────────────────────────────────
  const pinionPitchRadius = (pinionTeeth * pinionModule) / 2;
  const pinionTeethShape = drawPolysides(pinionPitchRadius + pinionModule * 1.2, pinionTeeth)
    .sketchOnPlane("XY", 0)
    .extrude(pinionFaceWidth);
  const pinionBody = drawCircle(pinionPitchRadius)
    .sketchOnPlane("XY", 0)
    .extrude(pinionFaceWidth);
  const pinionShaft = drawCircle(pinionShaftRadius)
    .sketchOnPlane("XY", pinionFaceWidth)
    .extrude(pinionShaftLength);
  const pinionBore = drawCircle(pinionShaftRadius - 4)
    .sketchOnPlane("XY", 0)
    .extrude(pinionFaceWidth + pinionShaftLength);

  // Universal coupling flange at shaft top
  const coupling = drawCircle(couplingRadius)
    .sketchOnPlane("XY", pinionFaceWidth + pinionShaftLength)
    .extrude(12)
    .cut(drawCircle(pinionShaftRadius)
      .sketchOnPlane("XY", pinionFaceWidth + pinionShaftLength)
      .extrude(12));

  const pinion = pinionTeethShape.intersect(pinionBody)
    .fuse(pinionShaft)
    .fuse(coupling)
    .cut(pinionBore)
    .translateY(-(pinionPitchRadius + rackHeight / 2))
    .translateZ(-pinionFaceWidth / 2);

  // ── RACK HOUSING TUBE ─────────────────────────────────────
  const housingOuter = drawCircle(housingRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(housingLength)
    .translateX(-housingLength / 2);
  const housingBore = drawCircle(housingRadius - 5)
    .sketchOnPlane("XZ", 0)
    .extrude(housingLength)
    .translateX(-housingLength / 2);
  // Power-assist cylinder bore on housing side
  const pasBore = drawCircle(pasBoreRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(housingLength * 0.4)
    .translateX(-housingLength / 2 + housingLength * 0.3)
    .translateY(housingRadius);

  // M8 bracket mount bosses (×2)
  const mountBoss1 = drawCircle(mountBossRadius)
    .sketchOnPlane("XZ", housingRadius)
    .extrude(16)
    .translateX(-housingLength * 0.3)
    .cut(drawCircle(mountBoltRadius)
      .sketchOnPlane("XZ", housingRadius)
      .extrude(16)
      .translateX(-housingLength * 0.3));
  const mountBoss2 = mountBoss1.clone().translateX(housingLength * 0.6);

  const housing = housingOuter.cut(housingBore).cut(pasBore)
    .fuse(mountBoss1).fuse(mountBoss2);

  // ── BELLOWS BOOTS ─────────────────────────────────────────
  const makeBellows = (tx) => {
    const outerProfile = drawCircle(bellowsRadius)
      .sketchOnPlane("XZ", 0)
      .extrude(bellowsLength)
      .translateX(tx);
    const innerProfile = drawCircle(housingRadius - 3)
      .sketchOnPlane("XZ", 0)
      .extrude(bellowsLength)
      .translateX(tx);
    return outerProfile.cut(innerProfile);
  };
  const bellowsLeft  = makeBellows(-housingLength / 2 - bellowsLength);
  const bellowsRight = makeBellows(housingLength / 2);

  // ── TIE RODS + BALL JOINTS ────────────────────────────────
  const makeTieRod = (tx) => {
    const rod = drawCircle(tieRodRadius)
      .sketchOnPlane("XZ", 0)
      .extrude(tieRodLength)
      .translateX(tx);
    const bjHousing = drawCircle(ballJointRadius)
      .sketchOnPlane("XZ", 0)
      .extrude(ballJointRadius * 2)
      .translateX(tx + (tx < 0 ? -tieRodLength - ballJointRadius * 2 : tieRodLength))
      .translateZ(-ballJointRadius);
    const bjBore = drawCircle(ballJointRadius - 4)
      .sketchOnPlane("XZ", 0)
      .extrude(ballJointRadius * 2)
      .translateX(tx + (tx < 0 ? -tieRodLength - ballJointRadius * 2 : tieRodLength))
      .translateZ(-ballJointRadius);
    return rod.fuse(bjHousing.cut(bjBore));
  };
  const tieRodLeft  = makeTieRod(-rackLength / 2 - tieRodLength);
  const tieRodRight = makeTieRod(rackLength / 2);

  return [
    { shape: rack,         name: "Ground Steel Rack",      color: "#A9A9A9" },
    { shape: pinion,       name: "Helical Pinion 18T",     color: "#B8860B" },
    { shape: housing,      name: "Power-Assist Housing",   color: "#3A5470" },
    { shape: bellowsLeft,  name: "Bellows Boot Left",      color: "#2E2E2E" },
    { shape: bellowsRight, name: "Bellows Boot Right",     color: "#2E2E2E" },
    { shape: tieRodLeft,   name: "Tie Rod Left",           color: "#808080" },
    { shape: tieRodRight,  name: "Tie Rod Right",          color: "#808080" },
  ];
};
```
