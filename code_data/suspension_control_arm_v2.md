---
source_file: suspension_control_arm_v2.md
category: assembly
type: annotated_code
use_case: vehicle suspension linkage, stamped high-strength steel arm with polyurethane bushings and taper ball joint
related: wheel_hub_bearing_v2.md, brake_rotor_caliper_v2.md, rack_pinion_steering_v2.md
---

# Suspension Control Arm with Bushings — HSLA Steel / Polyurethane

## Description
A stamped high-strength steel lower control arm (320 mm centre-to-centre) with two subframe mount bushing bores (Ø20 mm interference press-fit) and a tapered ball joint socket at the knuckle end. Polyurethane bushings are pressed into flanged steel sleeves. The arm profile includes reinforcement ribs and corrosion-resistant coating geometry. Two M10 subframe bolt holes and a ball joint taper bore for steering knuckle attachment.

## Keywords
control arm, HSLA steel, polyurethane bushing, ball joint, taper bore, subframe mount, press-fit bushing, 320mm arm, bushing bore 20mm, lower arm, wishbone, interference fit, suspension geometry, knuckle interface, stamped steel, corrosion coating, M10 subframe bolt, ball joint socket, pivot bushing

## Parameters
| Variable           | Value  | Unit | Meaning                               |
|--------------------|--------|------|---------------------------------------|
| armLength          | 320.0  | mm   | Arm length pivot-to-ball-joint        |
| armSpread          | 130.0  | mm   | Chassis end spread between bushings   |
| armThickness       | 5.0    | mm   | Stamped steel thickness               |
| armHeight          | 40.0   | mm   | Arm cross-section height              |
| bushing1PosX       | 0.0    | mm   | Front bushing X position              |
| bushing2PosX       | 50.0   | mm   | Rear bushing X position               |
| bushing2PosY       | 130.0  | mm   | Rear bushing Y spread                 |
| bushingOuterRadius | 30.0   | mm   | Bushing outer steel sleeve radius     |
| bushingBoreRadius  | 10.0   | mm   | Bushing inner bore radius (Ø20mm)     |
| bushingLength      | 55.0   | mm   | Bushing total length                  |
| polyUrethaneThick  | 8.0    | mm   | Polyurethane layer thickness          |
| ballJointRadius    | 28.0   | mm   | Ball joint housing radius             |
| ballJointHeight    | 38.0   | mm   | Ball joint housing height             |
| taperTopRadius     | 14.0   | mm   | Ball joint taper bore top radius      |
| taperBotRadius     | 9.0    | mm   | Ball joint taper bore bottom radius   |
| subframeBoltRadius | 5.0    | mm   | M10 subframe bolt hole radius         |
| ribThickness       | 4.0    | mm   | Reinforcement rib thickness           |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle } = replicad;

  const armLength          = 320.0;
  const armSpread          = 130.0;
  const armThickness       = 5.0;
  const armHeight          = 40.0;
  const bushingOuterRadius = 30.0;
  const bushingBoreRadius  = 10.0;
  const bushingLength      = 55.0;
  const polyUrethaneThick  = 8.0;
  const ballJointRadius    = 28.0;
  const ballJointHeight    = 38.0;
  const taperTopRadius     = 14.0;
  const taperBotRadius     = 9.0;
  const subframeBoltRadius = 5.0;
  const ribThickness       = 4.0;

  // ── CONTROL ARM BODY (triangular stamped plate) ───────────
  const armProfile = draw([0, 0])
    .lineTo([armLength, 0])
    .lineTo([armLength * 0.12, armSpread])
    .close();

  const armBody = armProfile
    .sketchOnPlane("XY", 0)
    .extrude(armHeight);

  // Lightening pocket
  const pocket = draw([armLength * 0.22, armSpread * 0.12])
    .lineTo([armLength * 0.78, armSpread * 0.04])
    .lineTo([armLength * 0.65, armSpread * 0.7])
    .close();

  const pocketCut = pocket
    .sketchOnPlane("XY", armThickness)
    .extrude(armHeight - armThickness * 2);

  // Reinforcement rib along arm spine
  const rib = drawRectangle(armLength * 0.7, ribThickness)
    .sketchOnPlane("XY", armHeight)
    .extrude(ribThickness)
    .translateX(armLength * 0.1)
    .translateY(-ribThickness / 2);

  const arm = armBody.cut(pocketCut).fuse(rib);

  // ── BUSHING 1 — FRONT SUBFRAME MOUNT ─────────────────────
  const makeBushing = (tx, ty) => {
    const outerSleeve = drawCircle(bushingOuterRadius)
      .sketchOnPlane("YZ", 0)
      .extrude(bushingLength)
      .translateZ(armHeight / 2 - bushingLength / 2)
      .translateX(tx)
      .translateY(ty);

    const polyRing = drawCircle(bushingOuterRadius - armThickness)
      .sketchOnPlane("YZ", 0)
      .extrude(bushingLength)
      .translateZ(armHeight / 2 - bushingLength / 2)
      .translateX(tx)
      .translateY(ty)
      .cut(drawCircle(bushingBoreRadius + polyUrethaneThick)
        .sketchOnPlane("YZ", 0)
        .extrude(bushingLength)
        .translateZ(armHeight / 2 - bushingLength / 2)
        .translateX(tx)
        .translateY(ty));

    const innerSleeve = drawCircle(bushingBoreRadius + 3)
      .sketchOnPlane("YZ", 0)
      .extrude(bushingLength)
      .translateZ(armHeight / 2 - bushingLength / 2)
      .translateX(tx)
      .translateY(ty)
      .cut(drawCircle(bushingBoreRadius)
        .sketchOnPlane("YZ", 0)
        .extrude(bushingLength)
        .translateZ(armHeight / 2 - bushingLength / 2)
        .translateX(tx)
        .translateY(ty));

    return { outerSleeve, polyRing, innerSleeve };
  };

  const b1 = makeBushing(0, 0);
  const b2 = makeBushing(armLength * 0.12, armSpread);

  // ── BALL JOINT HOUSING ────────────────────────────────────
  const bjHousing = drawCircle(ballJointRadius)
    .sketchOnPlane("XY", armHeight)
    .extrude(ballJointHeight)
    .translateX(armLength);

  // Tapered bore for knuckle taper
  const taperBore = draw([taperBotRadius, 0])
    .lineTo([taperTopRadius, ballJointHeight])
    .lineTo([-taperTopRadius, ballJointHeight])
    .lineTo([-taperBotRadius, 0])
    .close()
    .sketchOnPlane("XZ", 0)
    .revolve([0, 0, 1])
    .translateX(armLength)
    .translateZ(armHeight);

  const ballJoint = bjHousing.cut(taperBore);

  // Subframe bolt holes in bushing area
  const boltHole1 = drawCircle(subframeBoltRadius)
    .sketchOnPlane("YZ", 0)
    .extrude(bushingLength)
    .translateZ(armHeight / 2 - bushingLength / 2)
    .translateX(0);
  const boltHole2 = boltHole1.clone().translateX(armLength * 0.12).translateY(armSpread);

  return [
    { shape: arm,                name: "HSLA Steel Control Arm",   color: "#708090" },
    { shape: b1.outerSleeve,     name: "Bushing 1 Steel Sleeve",   color: "#505050" },
    { shape: b1.polyRing,        name: "Bushing 1 Polyurethane",   color: "#E8C840" },
    { shape: b1.innerSleeve,     name: "Bushing 1 Inner Sleeve",   color: "#606060" },
    { shape: b2.outerSleeve,     name: "Bushing 2 Steel Sleeve",   color: "#505050" },
    { shape: b2.polyRing,        name: "Bushing 2 Polyurethane",   color: "#E8C840" },
    { shape: b2.innerSleeve,     name: "Bushing 2 Inner Sleeve",   color: "#606060" },
    { shape: ballJoint,          name: "Ball Joint Housing",       color: "#8B7355" },
  ];
};
```
