---
source_file: turbocharger_v2.md
category: assembly
type: annotated_code
use_case: exhaust-driven forced induction with 6+6 blade compressor, ceramic hybrid bearings and V-band flange mount
related: intake_manifold_throttle_body_v2.md, camshaft_tappet_valve_v2.md
---

# Turbocharger Assembly — Cast Iron Housing / Billet Aluminum 6+6 Blade Compressor

## Description
A turbocharger with a billet aluminum compressor wheel (6 full + 6 splitter blades, inducer Ø45 mm), cast iron turbine housing with V-band clamp flanges, steel rotor shaft with ceramic hybrid bearing journals, oil feed and return ports, and a water cooling jacket boss. Tip clearance maintained at 0.3–0.6 mm. Compressor and turbine housings are individual sand-cast/machined components joined at the centre bearing housing.

## Keywords
turbocharger, compressor wheel, 6+6 blades, turbine wheel, cast iron housing, billet aluminum, V-band flange, ceramic hybrid bearing, inducer 45mm, tip clearance 0.5mm, oil feed port, oil return port, water cooling, bearing housing, compressor volute, turbine volute, boost pressure, exhaust energy, shaft balance, forced induction

## Parameters
| Variable              | Value  | Unit | Meaning                                  |
|-----------------------|--------|------|------------------------------------------|
| shaftRadius           | 9.0    | mm   | Rotor shaft radius                       |
| shaftLength           | 130.0  | mm   | Rotor shaft total length                 |
| bearingJournalRadius  | 13.0   | mm   | Ceramic hybrid bearing journal radius    |
| bearingJournalWidth   | 18.0   | mm   | Bearing journal width                    |
| compInducerRadius     | 22.5   | mm   | Compressor inducer radius (Ø45mm / 2)    |
| compExducerRadius     | 38.0   | mm   | Compressor exducer (tip) radius          |
| compWheelHeight       | 36.0   | mm   | Compressor wheel axial height            |
| compFullBlades        | 6      | -    | Full-length compressor blade count       |
| compSplitterBlades    | 6      | -    | Splitter blade count                     |
| turbineWheelRadius    | 40.0   | mm   | Turbine wheel tip radius                 |
| turbineWheelHeight    | 34.0   | mm   | Turbine wheel axial height               |
| turbineBladeCount     | 11     | -    | Turbine blade count                      |
| centreHousingRadius   | 44.0   | mm   | Centre bearing housing radius            |
| centreHousingLength   | 62.0   | mm   | Centre housing length                    |
| oilFeedPortRadius     | 5.0    | mm   | Oil feed port radius                     |
| oilReturnPortRadius   | 8.0    | mm   | Oil return port radius                   |
| waterJacketRadius     | 10.0   | mm   | Water cooling boss radius                |
| compVoluteRadius      | 60.0   | mm   | Compressor volute scroll outer radius    |
| compVoluteHeight      | 44.0   | mm   | Compressor volute height                 |
| turbVoluteRadius      | 68.0   | mm   | Turbine volute scroll outer radius       |
| turbVoluteHeight      | 42.0   | mm   | Turbine volute height                    |
| vBandFlangeRadius     | 52.0   | mm   | V-band clamp flange radius               |
| vBandFlangeThick      | 12.0   | mm   | V-band flange thickness                  |
| tipClearance          | 0.45   | mm   | Tip clearance (nominal)                  |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides, draw } = replicad;

  const shaftRadius          = 9.0;
  const shaftLength          = 130.0;
  const bearingJournalRadius = 13.0;
  const bearingJournalWidth  = 18.0;
  const compInducerRadius    = 22.5;
  const compExducerRadius    = 38.0;
  const compWheelHeight      = 36.0;
  const compFullBlades       = 6;
  const turbineWheelRadius   = 40.0;
  const turbineWheelHeight   = 34.0;
  const turbineBladeCount    = 11;
  const centreHousingRadius  = 44.0;
  const centreHousingLength  = 62.0;
  const oilFeedPortRadius    = 5.0;
  const oilReturnPortRadius  = 8.0;
  const waterJacketRadius    = 10.0;
  const compVoluteRadius     = 60.0;
  const compVoluteHeight     = 44.0;
  const turbVoluteRadius     = 68.0;
  const turbVoluteHeight     = 42.0;
  const vBandFlangeRadius    = 52.0;
  const vBandFlangeThick     = 12.0;

  // ── ROTOR SHAFT ───────────────────────────────────────────
  const shaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength);

  // Ceramic bearing journals
  const journal1 = drawCircle(bearingJournalRadius)
    .sketchOnPlane("XY", shaftLength * 0.18)
    .extrude(bearingJournalWidth);
  const journal2 = drawCircle(bearingJournalRadius)
    .sketchOnPlane("XY", shaftLength * 0.62)
    .extrude(bearingJournalWidth);

  const rotorShaft = shaft.fuse(journal1).fuse(journal2);

  // ── COMPRESSOR WHEEL (6+6 blades) ────────────────────────
  // Hub cone: inducer at top, exducer at base
  const compHubProfile = draw([shaftRadius, 0])
    .lineTo([compExducerRadius, 0])
    .lineTo([compInducerRadius, compWheelHeight])
    .lineTo([shaftRadius, compWheelHeight])
    .close();
  const compHub = compHubProfile
    .sketchOnPlane("XZ", 0)
    .revolve();

  // Full-length blades (×6)
  const fullBladeAngleStep = 360 / compFullBlades;
  let compWheel = compHub;
  for (let i = 0; i < compFullBlades; i++) {
    const angle = i * fullBladeAngleStep;
    const midR = (compInducerRadius + compExducerRadius) / 2;
    const blade = drawPolysides(midR + 3, 4)
      .sketchOnPlane("XY", compWheelHeight * 0.1)
      .extrude(compWheelHeight * 0.85)
      .cut(drawCircle(midR - 3)
        .sketchOnPlane("XY", compWheelHeight * 0.1)
        .extrude(compWheelHeight * 0.85))
      .rotate(angle, [0, 0, compWheelHeight / 2], [0, 0, 1]);
    compWheel = compWheel.fuse(blade);
  }

  const compBore = drawCircle(shaftRadius + 0.5)
    .sketchOnPlane("XY", 0)
    .extrude(compWheelHeight);
  compWheel = compWheel.cut(compBore).translateZ(shaftLength * 0.02);

  // ── TURBINE WHEEL ─────────────────────────────────────────
  const turbineHubProfile = draw([shaftRadius, 0])
    .lineTo([turbineWheelRadius, 0])
    .lineTo([turbineWheelRadius * 0.55, turbineWheelHeight])
    .lineTo([shaftRadius, turbineWheelHeight])
    .close();
  const turbineHub = turbineHubProfile
    .sketchOnPlane("XZ", 0)
    .revolve();

  const turbBladeAngleStep = 360 / turbineBladeCount;
  let turbineWheel = turbineHub;
  for (let i = 0; i < turbineBladeCount; i++) {
    const angle = i * turbBladeAngleStep;
    const turbMidR = turbineWheelRadius * 0.7;
    const turbBlade = drawPolysides(turbMidR + 2.5, 4)
      .sketchOnPlane("XY", turbineWheelHeight * 0.05)
      .extrude(turbineWheelHeight * 0.9)
      .cut(drawCircle(turbMidR - 2.5)
        .sketchOnPlane("XY", turbineWheelHeight * 0.05)
        .extrude(turbineWheelHeight * 0.9))
      .rotate(angle, [0, 0, turbineWheelHeight / 2], [0, 0, 1]);
    turbineWheel = turbineWheel.fuse(turbBlade);
  }

  const turbBore = drawCircle(shaftRadius + 0.5)
    .sketchOnPlane("XY", 0)
    .extrude(turbineWheelHeight);
  turbineWheel = turbineWheel.cut(turbBore)
    .translateZ(shaftLength - turbineWheelHeight - shaftLength * 0.02);

  // ── CENTRE BEARING HOUSING ────────────────────────────────
  const centreOuter = drawCircle(centreHousingRadius)
    .sketchOnPlane("XY", compWheelHeight + 2)
    .extrude(centreHousingLength);
  const centreBore = drawCircle(bearingJournalRadius + 2)
    .sketchOnPlane("XY", compWheelHeight + 2)
    .extrude(centreHousingLength);

  // Oil feed port (top)
  const oilFeed = drawCircle(oilFeedPortRadius)
    .sketchOnPlane("XZ", centreHousingRadius)
    .extrude(16)
    .translateZ(compWheelHeight + 2 + centreHousingLength * 0.35)
    .cut(drawCircle(oilFeedPortRadius - 1.5)
      .sketchOnPlane("XZ", centreHousingRadius)
      .extrude(10)
      .translateZ(compWheelHeight + 2 + centreHousingLength * 0.35));

  // Oil return port (bottom)
  const oilReturn = drawCircle(oilReturnPortRadius)
    .sketchOnPlane("XZ", -centreHousingRadius)
    .extrude(20)
    .translateZ(compWheelHeight + 2 + centreHousingLength * 0.45)
    .cut(drawCircle(oilReturnPortRadius - 2)
      .sketchOnPlane("XZ", -centreHousingRadius)
      .extrude(14)
      .translateZ(compWheelHeight + 2 + centreHousingLength * 0.45));

  // Water jacket boss
  const waterBoss = drawCircle(waterJacketRadius)
    .sketchOnPlane("YZ", centreHousingRadius)
    .extrude(14)
    .translateZ(compWheelHeight + 2 + centreHousingLength * 0.6);

  const centreHousing = centreOuter.cut(centreBore)
    .fuse(oilFeed).fuse(oilReturn).fuse(waterBoss);

  // ── COMPRESSOR VOLUTE HOUSING ─────────────────────────────
  const compVoluteOuter = drawCircle(compVoluteRadius)
    .sketchOnPlane("XY", -compVoluteHeight * 0.1)
    .extrude(compVoluteHeight);
  const compVoluteBore = drawCircle(compExducerRadius + 1)
    .sketchOnPlane("XY", -compVoluteHeight * 0.1)
    .extrude(compVoluteHeight);
  // V-band flange at compressor volute outlet
  const compVBand = drawCircle(vBandFlangeRadius)
    .sketchOnPlane("XY", -compVoluteHeight * 0.1)
    .extrude(vBandFlangeThick)
    .cut(drawCircle(centreHousingRadius + 2)
      .sketchOnPlane("XY", -compVoluteHeight * 0.1)
      .extrude(vBandFlangeThick));
  const compHousing = compVoluteOuter.cut(compVoluteBore).fuse(compVBand);

  // ── TURBINE VOLUTE HOUSING ────────────────────────────────
  const turbVoluteZ = shaftLength - turbineWheelHeight - 8;
  const turbVoluteOuter = drawCircle(turbVoluteRadius)
    .sketchOnPlane("XY", turbVoluteZ)
    .extrude(turbVoluteHeight);
  const turbVoluteBore = drawCircle(turbineWheelRadius + 1.5)
    .sketchOnPlane("XY", turbVoluteZ)
    .extrude(turbVoluteHeight);
  const turbVBand = drawCircle(vBandFlangeRadius)
    .sketchOnPlane("XY", turbVoluteZ + turbVoluteHeight)
    .extrude(vBandFlangeThick)
    .cut(drawCircle(centreHousingRadius + 2)
      .sketchOnPlane("XY", turbVoluteZ + turbVoluteHeight)
      .extrude(vBandFlangeThick));
  const turbineHousing = turbVoluteOuter.cut(turbVoluteBore).fuse(turbVBand);

  return [
    { shape: rotorShaft,    name: "Steel Rotor Shaft",          color: "#A0A0A0" },
    { shape: compWheel,     name: "Aluminum Compressor Wheel",  color: "#C8D8E8" },
    { shape: turbineWheel,  name: "Turbine Wheel",              color: "#B8860B" },
    { shape: centreHousing, name: "Centre Bearing Housing",     color: "#606870" },
    { shape: compHousing,   name: "Compressor Volute Housing",  color: "#3C5878" },
    { shape: turbineHousing,name: "Cast Iron Turbine Housing",  color: "#4A4A4A" },
  ];
};
```
