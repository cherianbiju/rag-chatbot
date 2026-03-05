---
source_file: wheel_hub_bearing_v2.md
category: assembly
type: annotated_code
use_case: wheel carrier with integrated tapered roller bearing race, ABS tone ring and 5x114.3 wheel studs
related: brake_rotor_caliper_v2.md, suspension_control_arm_v2.md
---

# Wheel Hub, Bearing and Brake-Mounting Assembly — Forged Hub / Tapered Roller Bearing

## Description
A forged hub flange (20 mm thick) with an integrated tapered roller bearing race, 5×114.3 mm wheel studs, an ABS tone ring (48 teeth), and a brake disc hat mounting surface. The hub bore houses a tapered roller inner race, and an outer race seats in the knuckle bore. A rubber dust seal groove and ABS sensor boss are included. Bearing axial runout spec 0.05–0.15 mm.

## Keywords
wheel hub, tapered roller bearing, hub flange, ABS tone ring, wheel stud, 5x114.3, bearing inner race, bearing outer race, tapered roller, dust seal, ABS sensor boss, hub bore, brake hat mount, forged hub, hub thickness 20mm, bearing preload, axial runout, knuckle bore, wheel bearing, drive flange

## Parameters
| Variable             | Value  | Unit | Meaning                                  |
|----------------------|--------|------|------------------------------------------|
| hubFlangeRadius      | 77.0   | mm   | Hub flange outer radius                  |
| hubFlangeThick       | 20.0   | mm   | Hub flange thickness (spec 20mm)         |
| hubBoreRadius        | 34.0   | mm   | Hub centre bore radius                   |
| hubBodyRadius        | 40.0   | mm   | Hub cylindrical body radius              |
| hubBodyLength        | 72.0   | mm   | Hub body length                          |
| studPCDRadius        | 57.15  | mm   | Wheel stud PCD radius (114.3/2)          |
| studRadius           | 7.5    | mm   | Wheel stud body radius                   |
| studLength           | 45.0   | mm   | Wheel stud exposed length                |
| studKnurledLength    | 15.0   | mm   | Press-fit knurled section length         |
| boltCount            | 5      | -    | Number of wheel studs                    |
| bearingOuterRaceR    | 48.0   | mm   | Tapered bearing outer race radius        |
| bearingInnerRaceR    | 34.0   | mm   | Bearing inner race (hub bore) radius     |
| bearingWidth         | 38.0   | mm   | Bearing total axial width                |
| taperAngle           | 15.0   | mm   | Taper cone depth (simplified)            |
| rollerRadius         | 5.0    | mm   | Tapered roller element radius            |
| rollerLength         | 18.0   | mm   | Roller length                            |
| rollerCount          | 14     | -    | Rollers per row                          |
| absRingOuterRadius   | 58.0   | mm   | ABS tone ring outer radius               |
| absRingInnerRadius   | 50.0   | mm   | ABS tone ring inner radius               |
| absRingWidth         | 8.0    | mm   | ABS tone ring axial width                |
| absToothCount        | 48     | -    | ABS tone ring tooth count                |
| dustSealRadius       | 38.0   | mm   | Dust seal groove radius                  |
| dustSealDepth        | 4.0    | mm   | Dust seal groove depth                   |
| absSensorBossRadius  | 11.0   | mm   | ABS sensor mounting boss radius          |
| absSensorBossHeight  | 20.0   | mm   | ABS sensor boss height                   |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides, draw } = replicad;

  const hubFlangeRadius     = 77.0;
  const hubFlangeThick      = 20.0;
  const hubBoreRadius       = 34.0;
  const hubBodyRadius       = 40.0;
  const hubBodyLength       = 72.0;
  const studPCDRadius       = 57.15;
  const studRadius          = 7.5;
  const studLength          = 45.0;
  const studKnurledLength   = 15.0;
  const boltCount           = 5;
  const bearingOuterRaceR   = 48.0;
  const bearingInnerRaceR   = 34.0;
  const bearingWidth        = 38.0;
  const taperAngle          = 15.0;
  const rollerRadius        = 5.0;
  const rollerLength        = 18.0;
  const rollerCount         = 14;
  const absRingOuterRadius  = 58.0;
  const absRingInnerRadius  = 50.0;
  const absRingWidth        = 8.0;
  const absToothCount       = 48;
  const dustSealRadius      = 38.0;
  const dustSealDepth       = 4.0;
  const absSensorBossRadius = 11.0;
  const absSensorBossHeight = 20.0;

  // ── HUB FLANGE ────────────────────────────────────────────
  const flangeDisc = drawCircle(hubFlangeRadius)
    .sketchOnPlane("XY", 0)
    .extrude(hubFlangeThick);

  const flangeBore = drawCircle(hubBoreRadius)
    .sketchOnPlane("XY", 0)
    .extrude(hubFlangeThick);

  // 5× wheel studs (press-fit with knurled shank)
  const studAngles = Array.from({ length: boltCount }, (_, i) => (360 / boltCount) * i);
  let hubFlange = flangeDisc.cut(flangeBore);

  studAngles.forEach(angle => {
    const rad = angle * Math.PI / 180;
    const sx = studPCDRadius * Math.cos(rad);
    const sy = studPCDRadius * Math.sin(rad);
    // Knurled press-fit section (larger radius in flange)
    const knurledSection = drawCircle(studRadius + 1)
      .sketchOnPlane("XY", 0)
      .extrude(hubFlangeThick)
      .translateX(sx).translateY(sy);
    // Exposed stud above flange
    const studBody = drawCircle(studRadius)
      .sketchOnPlane("XY", -studLength)
      .extrude(studLength)
      .translateX(sx).translateY(sy);
    hubFlange = hubFlange.fuse(knurledSection).fuse(studBody);
  });

  // ── HUB BODY ──────────────────────────────────────────────
  const hubBodyOuter = drawCircle(hubBodyRadius)
    .sketchOnPlane("XY", hubFlangeThick)
    .extrude(hubBodyLength);
  const hubBodyBore = drawCircle(hubBoreRadius)
    .sketchOnPlane("XY", hubFlangeThick)
    .extrude(hubBodyLength);

  // Dust seal groove at inboard end
  const dustSealGroove = drawCircle(dustSealRadius + dustSealDepth)
    .sketchOnPlane("XY", hubFlangeThick + hubBodyLength - 8)
    .extrude(6)
    .cut(drawCircle(dustSealRadius)
      .sketchOnPlane("XY", hubFlangeThick + hubBodyLength - 8)
      .extrude(6));

  const hubBody = hubBodyOuter.cut(hubBodyBore).fuse(dustSealGroove);
  const hub = hubFlange.fuse(hubBody);

  // ── TAPERED ROLLER BEARING — INNER RACE ───────────────────
  const bearingZ = hubFlangeThick + (hubBodyLength - bearingWidth) / 2;
  // Tapered bore profile (cone inner race)
  const innerRaceProfile = draw([bearingInnerRaceR, 0])
    .lineTo([bearingInnerRaceR + taperAngle, 0])
    .lineTo([bearingInnerRaceR + taperAngle + 4, bearingWidth])
    .lineTo([bearingInnerRaceR, bearingWidth])
    .close();
  const innerRace = innerRaceProfile
    .sketchOnPlane("XZ", 0)
    .revolve()
    .translateZ(bearingZ);

  // ── TAPERED ROLLER BEARING — OUTER RACE ───────────────────
  const outerRaceProfile = draw([bearingOuterRaceR - 4, 0])
    .lineTo([bearingOuterRaceR, 0])
    .lineTo([bearingOuterRaceR, bearingWidth])
    .lineTo([bearingOuterRaceR - 4 - taperAngle, bearingWidth])
    .close();
  const outerRace = outerRaceProfile
    .sketchOnPlane("XZ", 0)
    .revolve()
    .translateZ(bearingZ);

  // ── TAPERED ROLLERS ───────────────────────────────────────
  const rollerPCDRadius = (bearingInnerRaceR + taperAngle + bearingOuterRaceR - 4) / 2;
  const rollerAngleStep = 360 / rollerCount;
  let rollerAssembly = null;
  for (let i = 0; i < rollerCount; i++) {
    const angle = i * rollerAngleStep;
    const rad = angle * Math.PI / 180;
    const rx = rollerPCDRadius * Math.cos(rad);
    const ry = rollerPCDRadius * Math.sin(rad);
    const roller = drawCircle(rollerRadius)
      .sketchOnPlane("XY", bearingZ + (bearingWidth - rollerLength) / 2)
      .extrude(rollerLength)
      .translateX(rx).translateY(ry);
    rollerAssembly = rollerAssembly ? rollerAssembly.fuse(roller) : roller;
  }

  // ── ABS TONE RING ─────────────────────────────────────────
  const absTeethShape = drawPolysides(absRingOuterRadius + 2, absToothCount)
    .sketchOnPlane("XY", hubFlangeThick + hubBodyLength)
    .extrude(absRingWidth);
  const absRingBody = drawCircle(absRingOuterRadius)
    .sketchOnPlane("XY", hubFlangeThick + hubBodyLength)
    .extrude(absRingWidth);
  const absRingBore = drawCircle(absRingInnerRadius)
    .sketchOnPlane("XY", hubFlangeThick + hubBodyLength)
    .extrude(absRingWidth);
  const absRing = absTeethShape.intersect(absRingBody).cut(absRingBore);

  // ── ABS SENSOR BOSS ───────────────────────────────────────
  const absBoss = drawCircle(absSensorBossRadius)
    .sketchOnPlane("XZ", hubBodyRadius + absSensorBossHeight)
    .extrude(absSensorBossHeight)
    .translateZ(hubFlangeThick + hubBodyLength * 0.5)
    .cut(drawCircle(absSensorBossRadius - 3.5)
      .sketchOnPlane("XZ", hubBodyRadius + absSensorBossHeight)
      .extrude(absSensorBossHeight - 4)
      .translateZ(hubFlangeThick + hubBodyLength * 0.5));

  return [
    { shape: hub,           name: "Forged Wheel Hub",          color: "#909090" },
    { shape: innerRace,     name: "Tapered Inner Race",        color: "#C0C0C0" },
    { shape: outerRace,     name: "Tapered Outer Race",        color: "#B0B0B0" },
    { shape: rollerAssembly,name: "Tapered Rollers",           color: "#D4AF37" },
    { shape: absRing,       name: "ABS Tone Ring (48T)",       color: "#404040" },
    { shape: absBoss,       name: "ABS Sensor Boss",           color: "#707070" },
  ];
};
```
