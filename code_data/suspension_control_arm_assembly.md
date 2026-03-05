---
source_file: suspension_control_arm_assembly.md
category: assembly
type: annotated_code
use_case: vehicle suspension linkage, connects wheel hub to chassis while allowing controlled motion
related: wheel_hub_assembly.md, ball_joint.md, chassis_mount.md
---

# Suspension Control Arm with Bushings Assembly

## Description
A lower wishbone-style suspension control arm featuring a triangular arm body with two chassis mounting bushings and a ball joint socket at the wheel end. The rubber bushings allow rotational compliance at the chassis pivot points, while the ball joint permits multi-axis wheel movement. Used in independent front and rear suspension systems.

## Keywords
control arm, wishbone, suspension arm, bushing, ball joint, chassis mount, pivot, rubber bushing, suspension geometry, camber control, lower arm, independent suspension, wheel alignment, compliance, knuckle

## Parameters
| Variable         | Value | Unit | Meaning                            |
|------------------|-------|------|------------------------------------|
| armLength        | 280   | mm   | Control arm length (pivot to ball) |
| armWidth         | 120   | mm   | Arm spread at chassis end          |
| armThickness     | 12    | mm   | Arm body plate thickness           |
| armHeight        | 35    | mm   | Arm body cross-section height      |
| bushingRadius    | 22    | mm   | Bushing outer radius               |
| bushingLength    | 50    | mm   | Bushing length                     |
| bushingBoreRadius| 10    | mm   | Bushing inner bore radius          |
| ballJointRadius  | 25    | mm   | Ball joint housing radius          |
| ballJointHeight  | 35    | mm   | Ball joint housing height          |
| ballRadius       | 14    | mm   | Ball sphere radius                 |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle } = replicad;

  const armLength         = 280;
  const armWidth          = 120;
  const armThickness      = 12;
  const armHeight         = 35;
  const bushingRadius     = 22;
  const bushingLength     = 50;
  const bushingBoreRadius = 10;
  const ballJointRadius   = 25;
  const ballJointHeight   = 35;
  const ballRadius        = 14;

  // ── CONTROL ARM BODY (triangular plate) ───────────────────
  const armProfile = draw([0, 0])
    .lineTo([armLength, 0])
    .lineTo([armLength * 0.15, armWidth])
    .close();

  const armBody = armProfile
    .sketchOnPlane("XY", 0)
    .extrude(armHeight);

  // Lightening pocket
  const pocket = draw([armLength * 0.25, armWidth * 0.15])
    .lineTo([armLength * 0.75, armWidth * 0.05])
    .lineTo([armLength * 0.6, armWidth * 0.65])
    .close();

  const pocketCut = pocket
    .sketchOnPlane("XY", armThickness)
    .extrude(armHeight - armThickness * 2);

  const arm = armBody.cut(pocketCut);

  // ── BUSHING 1 (front chassis mount) ──────────────────────
  const bushingOuter1 = drawCircle(bushingRadius)
    .sketchOnPlane("YZ", 0)
    .extrude(bushingLength)
    .translateZ(-bushingLength / 2);

  const bushingRubber1 = drawCircle(bushingRadius - 4)
    .sketchOnPlane("YZ", 0)
    .extrude(bushingLength)
    .translateZ(-bushingLength / 2);

  const bushingBore1 = drawCircle(bushingBoreRadius)
    .sketchOnPlane("YZ", 0)
    .extrude(bushingLength)
    .translateZ(-bushingLength / 2);

  const bushing1 = bushingOuter1
    .cut(bushingBore1)
    .translateX(0)
    .translateY(0)
    .translateZ(armHeight / 2);

  // ── BUSHING 2 (rear chassis mount) ───────────────────────
  const bushing2 = bushing1.clone()
    .translateX(armLength * 0.15)
    .translateY(armWidth);

  // ── BALL JOINT HOUSING ────────────────────────────────────
  const ballHousing = drawCircle(ballJointRadius)
    .sketchOnPlane("XY", armHeight)
    .extrude(ballJointHeight)
    .translateX(armLength)
    .translateY(0);

  const ballSphere = drawCircle(ballRadius)
    .sketchOnPlane("XZ", 0)
    .revolve()
    .translateX(armLength)
    .translateY(ballJointRadius)
    .translateZ(armHeight + ballJointHeight * 0.5);

  const ballHousingBore = drawCircle(ballRadius + 1)
    .sketchOnPlane("XY", armHeight)
    .extrude(ballJointHeight)
    .translateX(armLength)
    .translateY(0);

  const ballJoint = ballHousing.cut(ballHousingBore).fuse(ballSphere);

  return [
    { shape: arm,       name: "Control Arm",   color: "#778899" },
    { shape: bushing1,  name: "Front Bushing", color: "#2E2E2E" },
    { shape: bushing2,  name: "Rear Bushing",  color: "#2E2E2E" },
    { shape: ballJoint, name: "Ball Joint",    color: "#B8860B" },
  ];
};
```
