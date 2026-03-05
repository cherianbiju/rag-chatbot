---
source_file: wheel_hub_bearing_brake_assembly.md
category: assembly
type: annotated_code
use_case: connects wheel to suspension, transmits drive torque and braking forces through wheel bearing
related: brake_rotor_caliper_assembly.md, suspension_control_arm_assembly.md, differential_bevel_gear_assembly.md
---

# Wheel Hub, Bearing and Brake-Mounting Assembly

## Description
A front wheel hub assembly consisting of a flanged hub body with wheel bolt studs, a double-row wheel bearing (inner and outer races with rolling element representation), an ABS tone ring, and a brake disc mounting hat. The hub flange provides bolt holes for wheel and rotor attachment, while the bearing allows the hub to rotate on the suspension knuckle with minimal friction under combined radial and axial loads.

## Keywords
wheel hub, wheel bearing, hub flange, bearing race, inner race, outer race, ABS ring, tone ring, wheel stud, brake hat, knuckle, hub bore, bearing preload, wheel bolt, lug stud, drive flange, wheel assembly, suspension knuckle, axial load, radial load

## Parameters
| Variable           | Value | Unit | Meaning                           |
|--------------------|-------|------|-----------------------------------|
| hubFlangeRadius    | 75    | mm   | Hub flange outer radius           |
| hubFlangeThickness | 18    | mm   | Hub flange thickness              |
| hubBoreRadius      | 32    | mm   | Hub centre bore radius            |
| hubBodyRadius      | 38    | mm   | Hub cylindrical body radius       |
| hubBodyLength      | 70    | mm   | Hub body length                   |
| studRadius         | 7     | mm   | Wheel stud radius                 |
| studLength         | 40    | mm   | Wheel stud protruding length      |
| studPCD            | 57    | mm   | Stud bolt circle radius           |
| boltCount          | 5     | -    | Number of wheel studs             |
| bearingOuterRadius | 45    | mm   | Bearing outer race radius         |
| bearingInnerRadius | 32    | mm   | Bearing inner race radius         |
| bearingWidth       | 35    | mm   | Bearing total width               |
| absRingRadius      | 55    | mm   | ABS tone ring outer radius        |
| absRingWidth       | 10    | mm   | ABS tone ring width               |
| absToothCount      | 48   | -    | Number of ABS tone ring teeth     |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides } = replicad;

  const hubFlangeRadius    = 75;
  const hubFlangeThickness = 18;
  const hubBoreRadius      = 32;
  const hubBodyRadius      = 38;
  const hubBodyLength      = 70;
  const studRadius         = 7;
  const studLength         = 40;
  const studPCD            = 57;
  const boltCount          = 5;
  const bearingOuterRadius = 45;
  const bearingInnerRadius = 32;
  const bearingWidth       = 35;
  const absRingRadius      = 55;
  const absRingWidth       = 10;
  const absToothCount      = 48;

  // ── HUB FLANGE ────────────────────────────────────────────
  const flangeDisc = drawCircle(hubFlangeRadius)
    .sketchOnPlane("XY", 0)
    .extrude(hubFlangeThickness);

  const flangeBore = drawCircle(hubBoreRadius)
    .sketchOnPlane("XY", 0)
    .extrude(hubFlangeThickness);

  // Wheel studs
  const studAngles = Array.from({ length: boltCount }, (_, i) => (360 / boltCount) * i);
  const studs = studAngles.map(angle => {
    const sx = studPCD * Math.cos(angle * Math.PI / 180);
    const sy = studPCD * Math.sin(angle * Math.PI / 180);
    return drawCircle(studRadius)
      .sketchOnPlane("XY", -studLength)
      .extrude(studLength + hubFlangeThickness)
      .translateX(sx)
      .translateY(sy);
  });

  let hubFlange = flangeDisc.cut(flangeBore);
  studs.forEach(stud => { hubFlange = hubFlange.fuse(stud); });

  // ── HUB BODY ──────────────────────────────────────────────
  const hubBodyOuter = drawCircle(hubBodyRadius)
    .sketchOnPlane("XY", hubFlangeThickness)
    .extrude(hubBodyLength);

  const hubBodyBore = drawCircle(hubBoreRadius)
    .sketchOnPlane("XY", hubFlangeThickness)
    .extrude(hubBodyLength);

  const hubBody = hubBodyOuter.cut(hubBodyBore);
  const hub = hubFlange.fuse(hubBody);

  // ── WHEEL BEARING ─────────────────────────────────────────
  const bearingZOffset = hubFlangeThickness + (hubBodyLength - bearingWidth) / 2;

  const outerRace = drawCircle(bearingOuterRadius)
    .sketchOnPlane("XY", bearingZOffset)
    .extrude(bearingWidth)
    .cut(
      drawCircle(bearingOuterRadius - 6)
        .sketchOnPlane("XY", bearingZOffset)
        .extrude(bearingWidth)
    );

  const innerRace = drawCircle(bearingInnerRadius + 6)
    .sketchOnPlane("XY", bearingZOffset)
    .extrude(bearingWidth)
    .cut(
      drawCircle(bearingInnerRadius)
        .sketchOnPlane("XY", bearingZOffset)
        .extrude(bearingWidth)
    );

  // Rolling elements (simplified as ring of spheres represented by a torus-like ring)
  const rollerRing = drawCircle((bearingOuterRadius + bearingInnerRadius) / 2 - bearingInnerRadius)
    .sketchOnPlane("XZ", 0)
    .revolve()
    .translateZ(bearingZOffset + bearingWidth / 2)
    .scale(1);

  // ── ABS TONE RING ─────────────────────────────────────────
  const absOuter = drawPolysides(absRingRadius + 3, absToothCount)
    .sketchOnPlane("XY", hubFlangeThickness + hubBodyLength)
    .extrude(absRingWidth);

  const absBody = drawCircle(absRingRadius)
    .sketchOnPlane("XY", hubFlangeThickness + hubBodyLength)
    .extrude(absRingWidth);

  const absInner = drawCircle(hubBodyRadius - 2)
    .sketchOnPlane("XY", hubFlangeThickness + hubBodyLength)
    .extrude(absRingWidth);

  const absRing = absOuter.intersect(absBody).cut(absInner);

  return [
    { shape: hub,       name: "Wheel Hub",    color: "#808080" },
    { shape: outerRace, name: "Outer Race",   color: "#B0B0B0" },
    { shape: innerRace, name: "Inner Race",   color: "#909090" },
    { shape: absRing,   name: "ABS Tone Ring",color: "#4A4A4A" },
  ];
};
```
