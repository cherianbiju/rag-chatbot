---
source_file: turbocharger_assembly.md
category: assembly
type: annotated_code
use_case: exhaust-driven forced induction, compresses intake air to increase engine power density
related: intake_manifold_throttle_body_assembly.md, camshaft_tappet_valve_assembly.md, shaft_design.md
---

# Turbocharger Turbine and Compressor Wheel Assembly

## Description
A turbocharger assembly consisting of a turbine wheel driven by exhaust gases, a compressor wheel that pressurises intake air, a centre housing with bearing shaft connecting both wheels, and simplified turbine and compressor housings (volutes). Exhaust energy spins the turbine which drives the compressor via a common shaft, forcing compressed air into the engine intake.

## Keywords
turbocharger, turbine wheel, compressor wheel, turbo shaft, volute, compressor housing, turbine housing, centre housing, boost pressure, exhaust energy, forced induction, impeller, axial flow, radial compressor, wastegate, turbo lag, charge air

## Parameters
| Variable            | Value | Unit | Meaning                            |
|---------------------|-------|------|------------------------------------|
| turbineWheelRadius  | 45    | mm   | Turbine wheel outer radius         |
| turbineWheelHeight  | 35    | mm   | Turbine wheel height               |
| turbineHubRadius    | 15    | mm   | Turbine wheel hub radius           |
| turbineBladeCount   | 11    | -    | Number of turbine blades           |
| compWheelRadius     | 38    | mm   | Compressor wheel outer radius      |
| compWheelHeight     | 30    | mm   | Compressor wheel height            |
| compHubRadius       | 13    | mm   | Compressor hub radius              |
| compBladeCount      | 9     | -    | Number of compressor blades        |
| shaftRadius         | 8     | mm   | Turbo shaft radius                 |
| shaftLength         | 120   | mm   | Turbo shaft length                 |
| centreHousingRadius | 40    | mm   | Centre bearing housing radius      |
| centreHousingLength | 60    | mm   | Centre housing length              |
| turbineVoluteRadius | 65    | mm   | Turbine volute outer radius        |
| compVoluteRadius    | 58    | mm   | Compressor volute outer radius     |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides } = replicad;

  const turbineWheelRadius  = 45;
  const turbineWheelHeight  = 35;
  const turbineHubRadius    = 15;
  const turbineBladeCount   = 11;
  const compWheelRadius     = 38;
  const compWheelHeight     = 30;
  const compHubRadius       = 13;
  const compBladeCount      = 9;
  const shaftRadius         = 8;
  const shaftLength         = 120;
  const centreHousingRadius = 40;
  const centreHousingLength = 60;
  const turbineVoluteRadius = 65;
  const compVoluteRadius    = 58;

  // ── TURBO SHAFT ───────────────────────────────────────────
  const shaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength);

  // ── TURBINE WHEEL ─────────────────────────────────────────
  const turbineBase = drawCircle(turbineWheelRadius)
    .sketchOnPlane("XY", 0)
    .extrude(turbineWheelHeight);

  const turbineInducer = drawCircle(turbineWheelRadius * 0.6)
    .sketchOnPlane("XY", turbineWheelHeight * 0.3)
    .extrude(turbineWheelHeight * 0.7);

  const turbineBladeProfile = drawPolysides(turbineWheelRadius - 2, turbineBladeCount)
    .sketchOnPlane("XY", 0)
    .extrude(turbineWheelHeight);

  const turbineHub = drawCircle(turbineHubRadius)
    .sketchOnPlane("XY", 0)
    .extrude(turbineWheelHeight);

  const turbineBore = drawCircle(shaftRadius + 1)
    .sketchOnPlane("XY", 0)
    .extrude(turbineWheelHeight);

  const turbineWheel = turbineBladeProfile
    .intersect(turbineBase)
    .fuse(turbineInducer)
    .fuse(turbineHub)
    .cut(turbineBore)
    .translateZ(shaftLength * 0.7);

  // ── COMPRESSOR WHEEL ─────────────────────────────────────
  const compBase = drawCircle(compWheelRadius)
    .sketchOnPlane("XY", 0)
    .extrude(compWheelHeight);

  const compBladeProfile = drawPolysides(compWheelRadius - 2, compBladeCount)
    .sketchOnPlane("XY", 0)
    .extrude(compWheelHeight);

  const compHub = drawCircle(compHubRadius)
    .sketchOnPlane("XY", 0)
    .extrude(compWheelHeight);

  const compBore = drawCircle(shaftRadius + 1)
    .sketchOnPlane("XY", 0)
    .extrude(compWheelHeight);

  const compWheel = compBladeProfile
    .intersect(compBase)
    .fuse(compHub)
    .cut(compBore)
    .translateZ(0);

  // ── CENTRE HOUSING ────────────────────────────────────────
  const centreOuter = drawCircle(centreHousingRadius)
    .sketchOnPlane("XY", compWheelHeight)
    .extrude(centreHousingLength);

  const centreBore = drawCircle(shaftRadius + 4)
    .sketchOnPlane("XY", compWheelHeight)
    .extrude(centreHousingLength);

  const centreHousing = centreOuter.cut(centreBore);

  // ── TURBINE VOLUTE HOUSING ────────────────────────────────
  const turbineVoluteOuter = drawCircle(turbineVoluteRadius)
    .sketchOnPlane("XY", shaftLength * 0.65)
    .extrude(turbineWheelHeight + 15);

  const turbineVoluteBore = drawCircle(turbineWheelRadius + 5)
    .sketchOnPlane("XY", shaftLength * 0.65)
    .extrude(turbineWheelHeight + 15);

  const turbineHousing = turbineVoluteOuter.cut(turbineVoluteBore);

  // ── COMPRESSOR VOLUTE HOUSING ─────────────────────────────
  const compVoluteOuter = drawCircle(compVoluteRadius)
    .sketchOnPlane("XY", -15)
    .extrude(compWheelHeight + 15);

  const compVoluteBore = drawCircle(compWheelRadius + 5)
    .sketchOnPlane("XY", -15)
    .extrude(compWheelHeight + 15);

  const compHousing = compVoluteOuter.cut(compVoluteBore);

  return [
    { shape: shaft,          name: "Turbo Shaft",         color: "#A9A9A9" },
    { shape: turbineWheel,   name: "Turbine Wheel",       color: "#B8860B" },
    { shape: compWheel,      name: "Compressor Wheel",    color: "#C0C0C0" },
    { shape: centreHousing,  name: "Centre Housing",      color: "#696969" },
    { shape: turbineHousing, name: "Turbine Housing",     color: "#2F4F4F" },
    { shape: compHousing,    name: "Compressor Housing",  color: "#3C5A6E" },
  ];
};
```
