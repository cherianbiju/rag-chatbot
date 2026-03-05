---
source_file: brake_rotor_caliper_assembly.md
category: assembly
type: annotated_code
use_case: vehicle braking system, friction-based deceleration via disc and hydraulic caliper
related: wheel_hub_assembly.md, brake_pad.md, suspension_assembly.md
---

# Brake Rotor and Caliper Assembly

## Description
A disc brake assembly comprising a vented rotor disc with bolt holes and a floating caliper housing with brake pad channels. The rotor attaches to the wheel hub and spins with the wheel, while the hydraulic caliper clamps brake pads against the rotor faces to generate braking force. Vented rotors dissipate heat through internal cooling vanes.

## Keywords
brake rotor, disc brake, caliper, vented disc, brake pad, hydraulic caliper, rotor hat, bolt circle, cooling vanes, friction surface, floating caliper, braking torque, heat dissipation, automotive braking, wheel brake

## Parameters
| Variable          | Value | Unit | Meaning                            |
|-------------------|-------|------|------------------------------------|
| rotorOuterRadius  | 140   | mm   | Rotor outer disc radius            |
| rotorInnerRadius  | 70    | mm   | Rotor inner (hat) radius           |
| rotorThickness    | 28    | mm   | Total rotor disc thickness (vented)|
| hatHeight         | 40    | mm   | Hat (hub mount) height             |
| hatRadius         | 45    | mm   | Hat outer radius                   |
| boltCircleRadius  | 57    | mm   | Bolt hole PCD radius               |
| boltHoleRadius    | 7     | mm   | Individual bolt hole radius        |
| boltCount         | 5     | mm   | Number of wheel bolts              |
| caliperWidth      | 80    | mm   | Caliper housing width              |
| caliperHeight     | 60    | mm   | Caliper housing height             |
| caliperDepth      | 40    | mm   | Caliper housing depth              |
| padChannelWidth   | 30    | mm   | Brake pad channel width            |
| padChannelDepth   | 15    | mm   | Brake pad channel depth            |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle } = replicad;

  const rotorOuterRadius = 140;
  const rotorInnerRadius = 70;
  const rotorThickness   = 28;
  const hatHeight        = 40;
  const hatRadius        = 45;
  const boltCircleRadius = 57;
  const boltHoleRadius   = 7;
  const caliperWidth     = 80;
  const caliperHeight    = 60;
  const caliperDepth     = 40;
  const padChannelWidth  = 30;
  const padChannelDepth  = 15;

  // ── ROTOR DISC ────────────────────────────────────────────
  const rotorDisc = drawCircle(rotorOuterRadius)
    .sketchOnPlane("XY", 0)
    .extrude(rotorThickness);

  const rotorCentreBore = drawCircle(rotorInnerRadius)
    .sketchOnPlane("XY", 0)
    .extrude(rotorThickness);

  // Venting slot cut (simplified as inner ring removal)
  const ventingRing = drawCircle(rotorInnerRadius + 20)
    .sketchOnPlane("XY", rotorThickness * 0.25)
    .extrude(rotorThickness * 0.5)
    .cut(
      drawCircle(rotorInnerRadius + 5)
        .sketchOnPlane("XY", rotorThickness * 0.25)
        .extrude(rotorThickness * 0.5)
    );

  // Bolt holes on PCD
  const boltAngles = [0, 72, 144, 216, 288];
  const boltHoles = boltAngles.map(angle => {
    const bx = boltCircleRadius * Math.cos(angle * Math.PI / 180);
    const by = boltCircleRadius * Math.sin(angle * Math.PI / 180);
    return drawCircle(boltHoleRadius)
      .sketchOnPlane("XY", 0)
      .extrude(rotorThickness)
      .translateX(bx)
      .translateY(by);
  });

  let rotor = rotorDisc.cut(rotorCentreBore).cut(ventingRing);
  boltHoles.forEach(hole => { rotor = rotor.cut(hole); });

  // ── ROTOR HAT ─────────────────────────────────────────────
  const hat = drawCircle(hatRadius)
    .sketchOnPlane("XY", -hatHeight)
    .extrude(hatHeight);

  const hatBore = drawCircle(30)
    .sketchOnPlane("XY", -hatHeight)
    .extrude(hatHeight);

  const rotorWithHat = rotor.fuse(hat.cut(hatBore));

  // ── CALIPER HOUSING ───────────────────────────────────────
  const caliperBody = drawRectangle(caliperWidth, caliperHeight)
    .sketchOnPlane("XZ", 0)
    .extrude(caliperDepth);

  const padChannel = drawRectangle(padChannelWidth, rotorThickness + 6)
    .sketchOnPlane("XZ", 0)
    .extrude(padChannelDepth);

  const boreLeft = drawCircle(14)
    .sketchOnPlane("XZ", caliperHeight * 0.5)
    .extrude(caliperDepth * 0.4);

  const boreRight = drawCircle(14)
    .sketchOnPlane("XZ", caliperHeight * 0.5)
    .extrude(caliperDepth * 0.4)
    .translateY(caliperDepth * 0.6);

  const caliper = caliperBody
    .cut(padChannel.translateX(-padChannelWidth / 2).translateZ(-1))
    .cut(boreLeft)
    .cut(boreRight)
    .translateX(rotorOuterRadius - caliperWidth / 2)
    .translateY(-caliperDepth / 2)
    .translateZ(rotorThickness / 2 - caliperHeight / 2);

  return [
    { shape: rotorWithHat, name: "Brake Rotor",    color: "#808080" },
    { shape: caliper,      name: "Brake Caliper",  color: "#2F4F4F" },
  ];
};
```
