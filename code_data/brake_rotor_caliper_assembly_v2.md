---
source_file: brake_rotor_caliper_assembly_v2.md
category: assembly
type: annotated_code
use_case: disc brake system, cast iron vented rotor with 4-piston aluminum caliper for vehicle braking
related: wheel_hub_bearing_v2.md, suspension_control_arm_v2.md
---

# Brake Rotor and Caliper Assembly — Vented Cast Iron / 4-Piston Aluminum

## Description
A high-performance disc brake assembly featuring a vented cast iron rotor (Ø320×30 mm) with internal cooling vanes, machined friction faces to ±0.2 mm thickness tolerance, 5×114.3 mm wheel bolt pattern, and a rotor hat. The 4-piston aluminum caliper spans the rotor with two caliper mounting bosses (M10), a bleed port, and brake pad channels. Rotor runout spec ≤0.05 mm.

## Keywords
vented rotor, cast iron rotor, 4-piston caliper, brake disc, cooling vanes, bleed port, caliper mount boss, M10 mount, 5x114.3 bolt pattern, rotor hat, brake pad shim, aluminum caliper, anodized caliper, disc brake, rotor runout, braking force, heat dissipation, piston bore, pad channel

## Parameters
| Variable           | Value  | Unit | Meaning                                |
|--------------------|--------|------|----------------------------------------|
| rotorOuterRadius   | 160.0  | mm   | Rotor outer radius (Ø320/2)            |
| rotorThickness     | 30.0   | mm   | Rotor disc total thickness             |
| rotorInnerRadius   | 78.0   | mm   | Inner friction surface boundary radius |
| ventInnerRadius    | 80.0   | mm   | Cooling vane inner radius              |
| ventOuterRadius    | 148.0  | mm   | Cooling vane outer radius              |
| ventCount          | 36     | -    | Number of cooling vanes                |
| ventWidth          | 4.0    | mm   | Cooling vane width                     |
| hatRadius          | 50.0   | mm   | Rotor hat outer radius                 |
| hatHeight          | 42.0   | mm   | Rotor hat height                       |
| hatBoreRadius      | 33.0   | mm   | Hat centre bore radius                 |
| boltPCDRadius      | 57.15  | mm   | Wheel bolt PCD radius (114.3/2)        |
| boltHoleRadius     | 7.5    | mm   | Wheel bolt hole radius                 |
| caliperLength      | 160.0  | mm   | Caliper housing length                 |
| caliperHeight      | 70.0   | mm   | Caliper housing height                 |
| caliperDepth       | 50.0   | mm   | Caliper housing depth (straddles rotor)|
| pistonBoreRadius   | 20.0   | mm   | Individual piston bore radius          |
| pistonBoreDepth    | 30.0   | mm   | Piston bore depth                      |
| padChannelWidth    | 36.0   | mm   | Brake pad channel width                |
| bleedPortRadius    | 4.0    | mm   | Bleed nipple port radius               |
| mountBossRadius    | 14.0   | mm   | M10 caliper mount boss radius          |
| mountBossHeight    | 18.0   | mm   | Caliper mount boss height              |
| mountBoltRadius    | 5.0    | mm   | M10 bolt hole radius                   |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle, draw } = replicad;

  const rotorOuterRadius  = 160.0;
  const rotorThickness    = 30.0;
  const rotorInnerRadius  = 78.0;
  const ventInnerRadius   = 80.0;
  const ventOuterRadius   = 148.0;
  const ventCount         = 36;
  const ventWidth         = 4.0;
  const hatRadius         = 50.0;
  const hatHeight         = 42.0;
  const hatBoreRadius     = 33.0;
  const boltPCDRadius     = 57.15;
  const boltHoleRadius    = 7.5;
  const caliperLength     = 160.0;
  const caliperHeight     = 70.0;
  const caliperDepth      = 50.0;
  const pistonBoreRadius  = 20.0;
  const pistonBoreDepth   = 30.0;
  const padChannelWidth   = 36.0;
  const bleedPortRadius   = 4.0;
  const mountBossRadius   = 14.0;
  const mountBossHeight   = 18.0;
  const mountBoltRadius   = 5.0;

  // ── ROTOR DISC (two friction faces + vented core) ─────────
  const faceTop = drawCircle(rotorOuterRadius)
    .sketchOnPlane("XY", rotorThickness * 0.6)
    .extrude(rotorThickness * 0.4)
    .cut(drawCircle(rotorInnerRadius)
      .sketchOnPlane("XY", rotorThickness * 0.6)
      .extrude(rotorThickness * 0.4));

  const faceBottom = drawCircle(rotorOuterRadius)
    .sketchOnPlane("XY", 0)
    .extrude(rotorThickness * 0.4)
    .cut(drawCircle(rotorInnerRadius)
      .sketchOnPlane("XY", 0)
      .extrude(rotorThickness * 0.4));

  // Venting web ring between faces
  const ventRing = drawCircle(rotorOuterRadius)
    .sketchOnPlane("XY", rotorThickness * 0.4)
    .extrude(rotorThickness * 0.2)
    .cut(drawCircle(rotorInnerRadius)
      .sketchOnPlane("XY", rotorThickness * 0.4)
      .extrude(rotorThickness * 0.2));

  // Cooling vane cuts
  const ventSlotAngle = 360 / ventCount;
  let rotor = faceBottom.fuse(faceTop).fuse(ventRing);
  for (let i = 0; i < ventCount; i++) {
    const angle = i * ventSlotAngle;
    const rad = angle * Math.PI / 180;
    const vx = ((ventInnerRadius + ventOuterRadius) / 2) * Math.cos(rad);
    const vy = ((ventInnerRadius + ventOuterRadius) / 2) * Math.sin(rad);
    const ventSlot = drawRectangle(ventWidth, ventOuterRadius - ventInnerRadius)
      .sketchOnPlane("XY", rotorThickness * 0.4)
      .extrude(rotorThickness * 0.2)
      .rotate(angle, [0, 0, rotorThickness * 0.5], [0, 0, 1])
      .translateX(vx * 0.0)
      .translateY(vy * 0.0);
    rotor = rotor.cut(ventSlot);
  }

  // Wheel bolt holes (5×114.3)
  const boltAngles = [0, 72, 144, 216, 288];
  boltAngles.forEach(angle => {
    const rad = angle * Math.PI / 180;
    const bx = boltPCDRadius * Math.cos(rad);
    const by = boltPCDRadius * Math.sin(rad);
    const boltHole = drawCircle(boltHoleRadius)
      .sketchOnPlane("XY", 0)
      .extrude(rotorThickness)
      .translateX(bx)
      .translateY(by);
    rotor = rotor.cut(boltHole);
  });

  // ── ROTOR HAT ─────────────────────────────────────────────
  const hatOuter = drawCircle(hatRadius)
    .sketchOnPlane("XY", -hatHeight)
    .extrude(hatHeight);
  const hatBore = drawCircle(hatBoreRadius)
    .sketchOnPlane("XY", -hatHeight)
    .extrude(hatHeight);
  const hat = hatOuter.cut(hatBore);
  const rotorAssembly = rotor.fuse(hat);

  // ── 4-PISTON CALIPER HOUSING ──────────────────────────────
  const caliperBody = drawRectangle(caliperLength, caliperHeight)
    .sketchOnPlane("XZ", -caliperDepth / 2)
    .extrude(caliperDepth)
    .translateX(-caliperLength / 2)
    .translateZ(-caliperHeight / 2);

  // Rotor slot through caliper centre
  const rotorSlot = drawRectangle(caliperLength * 0.7, rotorThickness + 6)
    .sketchOnPlane("XZ", -caliperDepth / 2)
    .extrude(caliperDepth)
    .translateX(-caliperLength * 0.35)
    .translateZ(-(rotorThickness + 6) / 2);

  // 4 piston bores (2 per side)
  const piston1 = drawCircle(pistonBoreRadius)
    .sketchOnPlane("XZ", -(caliperDepth / 2))
    .extrude(pistonBoreDepth)
    .translateX(-caliperLength * 0.2)
    .translateZ(caliperHeight * 0.1);
  const piston2 = piston1.clone().translateX(caliperLength * 0.4);
  const piston3 = drawCircle(pistonBoreRadius)
    .sketchOnPlane("XZ", caliperDepth / 2 - pistonBoreDepth)
    .extrude(pistonBoreDepth)
    .translateX(-caliperLength * 0.2)
    .translateZ(caliperHeight * 0.1);
  const piston4 = piston3.clone().translateX(caliperLength * 0.4);

  // Bleed port
  const bleedPort = drawCircle(bleedPortRadius)
    .sketchOnPlane("XY", caliperHeight * 0.3)
    .extrude(caliperDepth)
    .translateX(caliperLength * 0.3)
    .translateZ(caliperHeight * 0.25);

  // Caliper mount bosses (2×M10)
  const boss1 = drawCircle(mountBossRadius)
    .sketchOnPlane("XZ", caliperDepth / 2)
    .extrude(mountBossHeight)
    .translateX(-caliperLength * 0.38)
    .translateZ(-caliperHeight * 0.35);
  const boss1Bore = drawCircle(mountBoltRadius)
    .sketchOnPlane("XZ", caliperDepth / 2)
    .extrude(mountBossHeight)
    .translateX(-caliperLength * 0.38)
    .translateZ(-caliperHeight * 0.35);
  const boss2 = boss1.clone().translateX(caliperLength * 0.76);
  const boss2Bore = boss1Bore.clone().translateX(caliperLength * 0.76);

  const caliper = caliperBody
    .cut(rotorSlot)
    .cut(piston1).cut(piston2).cut(piston3).cut(piston4)
    .cut(bleedPort)
    .fuse(boss1.cut(boss1Bore))
    .fuse(boss2.cut(boss2Bore))
    .translateX(rotorOuterRadius - caliperLength / 2)
    .translateY(-caliperDepth / 2)
    .translateZ(rotorThickness / 2 - caliperHeight / 2);

  return [
    { shape: rotorAssembly, name: "Vented Cast Iron Rotor", color: "#808080" },
    { shape: caliper,       name: "4-Piston Aluminum Caliper", color: "#2B4A6F" },
  ];
};
```
