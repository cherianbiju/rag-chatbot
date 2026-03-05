---
source_file: control_arm.js
category: suspension
type: annotated_code
use_case: locates wheel hub relative to chassis, controls wheel motion in suspension travel
related: bushing.md, ball_joint.md
---
# Suspension Control Arm

## Description
A stamped steel lower control arm with two subframe bushing bores at the chassis end and a ball joint taper socket at the knuckle end. The A-shaped profile provides lateral and longitudinal stiffness.

## Keywords
control arm, suspension, A-arm, wishbone, bushing bore, ball joint, subframe, knuckle, extrude, sketcher, fuse, cut, cylinder, sweep, stamped steel, suspension geometry

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| ARM_LENGTH | 320 | mm | pivot-to-ball-joint center length |
| ARM_WIDTH | 80 | mm | max width at chassis end |
| THICKNESS | 8 | mm | material thickness |
| BUSHING_BORE_R | 12 | mm | inner radius of bushing bore |
| BUSHING_TUBE_R | 18 | mm | outer radius of bushing tube |
| BUSHING_LENGTH | 40 | mm | bushing tube length |
| BALL_JOINT_R | 16 | mm | ball joint socket outer radius |
| BALL_JOINT_H | 30 | mm | ball joint socket height |

## Code
```javascript
const main = (replicad) => {
  const {
    Sketcher,
    makeCylinder,
  } = replicad;

  const ARM_LENGTH      = 320;
  const ARM_WIDTH       = 80;
  const THICKNESS       = 8;
  const BUSHING_BORE_R  = 12;
  const BUSHING_TUBE_R  = 18;
  const BUSHING_LENGTH  = 40;
  const BALL_JOINT_R    = 16;
  const BALL_JOINT_H    = 30;

  // A-arm profile in XY plane
  const armProfile = new Sketcher("XY")
    .movePointerTo([0, 0])
    .lineTo([ARM_LENGTH, 0])
    .lineTo([ARM_LENGTH * 0.15, ARM_WIDTH / 2])
    .lineTo([0, ARM_WIDTH * 0.3])
    .close();

  let arm = armProfile.extrude(THICKNESS);

  // Bushing tube 1 at chassis pivot front
  const tube1Outer = makeCylinder(BUSHING_TUBE_R, BUSHING_LENGTH, [0, ARM_WIDTH * 0.15, -BUSHING_LENGTH / 2], [0, 0, 1]);
  const tube1Inner = makeCylinder(BUSHING_BORE_R, BUSHING_LENGTH + 2, [0, ARM_WIDTH * 0.15, -BUSHING_LENGTH / 2 - 1], [0, 0, 1]);
  arm = arm.fuse(tube1Outer).cut(tube1Inner);

  // Bushing tube 2 at chassis pivot rear
  const tube2Outer = makeCylinder(BUSHING_TUBE_R, BUSHING_LENGTH, [0, 0, -BUSHING_LENGTH / 2], [0, 0, 1]);
  const tube2Inner = makeCylinder(BUSHING_BORE_R, BUSHING_LENGTH + 2, [0, 0, -BUSHING_LENGTH / 2 - 1], [0, 0, 1]);
  arm = arm.fuse(tube2Outer).cut(tube2Inner);

  // Ball joint socket at knuckle end
  const bjOuter = makeCylinder(BALL_JOINT_R, BALL_JOINT_H, [ARM_LENGTH, 0, -BALL_JOINT_H / 2], [0, 0, 1]);
  const bjInner = makeCylinder(BALL_JOINT_R - 5, BALL_JOINT_H + 2, [ARM_LENGTH, 0, -BALL_JOINT_H / 2 - 1], [0, 0, 1]);
  arm = arm.fuse(bjOuter).cut(bjInner);

  return { shape: arm, name: "Control Arm", color: "steelblue" };
};
```
