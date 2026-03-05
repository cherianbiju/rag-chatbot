---
source_file: intake_manifold_throttle_body_v2.md
category: assembly
type: annotated_code
use_case: 4-cylinder intake air distribution, injection-molded polymer manifold with CNC aluminum throttle body
related: camshaft_tappet_valve_v2.md, turbocharger_v2.md
---

# Intake Manifold and Throttle Body Assembly — Nylon-Reinforced / Aluminum Ø60 mm

## Description
A 4-runner intake manifold in nylon-reinforced polymer with 230 mm equal-length runners, an aluminum throttle body (Ø60 mm bore) with butterfly valve, integrated MAP sensor boss, vacuum port, and gasket grooves at cylinder head flanges. Each runner terminates in a 4×M8 bolt flange to the cylinder head. Throttle body has a TPS (throttle position sensor) boss and idle air bypass port.

## Keywords
intake manifold, throttle body, 4-runner manifold, runner length 230mm, polymer manifold, aluminum throttle body, MAP sensor boss, butterfly valve, vacuum port, gasket groove, M8 flange bolt, cylinder head port, injection molded, CNC throttle, idle air bypass, TPS boss, plenum chamber, charge air, volumetric efficiency, Ø60 throttle bore

## Parameters
| Variable           | Value  | Unit | Meaning                                   |
|--------------------|--------|------|-------------------------------------------|
| plenumLength       | 220.0  | mm   | Plenum chamber length                     |
| plenumWidth        | 110.0  | mm   | Plenum chamber width                      |
| plenumHeight       | 90.0   | mm   | Plenum chamber height                     |
| plenumWallThick    | 4.5    | mm   | Manifold wall thickness                   |
| runnerRadius       | 24.0   | mm   | Runner bore radius                        |
| runnerWallThick    | 3.5    | mm   | Runner wall thickness                     |
| runnerLength       | 230.0  | mm   | Runner length (centre)                    |
| runnerSpacing      | 50.0   | mm   | Runner centre-to-centre spacing           |
| flangeWidth        | 60.0   | mm   | Cylinder head flange width                |
| flangeHeight       | 55.0   | mm   | Cylinder head flange height               |
| flangeThick        | 9.0    | mm   | Flange mounting thickness                 |
| gasketGrooveWidth  | 2.5    | mm   | Gasket groove width                       |
| gasketGrooveDepth  | 1.8    | mm   | Gasket groove depth                       |
| flangeBoltRadius   | 4.0    | mm   | M8 flange bolt hole radius                |
| tbBodyRadius       | 40.0   | mm   | Throttle body housing outer radius        |
| tbBoreRadius       | 30.0   | mm   | Throttle body bore radius (Ø60mm)         |
| tbLength           | 90.0   | mm   | Throttle body housing length              |
| butterflyRadius    | 29.0   | mm   | Butterfly valve disc radius               |
| butterflyThick     | 3.0    | mm   | Butterfly disc thickness                  |
| mapBossRadius      | 10.0   | mm   | MAP sensor boss outer radius              |
| mapBossHeight      | 18.0   | mm   | MAP sensor boss height                    |
| vacuumPortRadius   | 5.5    | mm   | Vacuum port radius                        |
| tpsBossRadius      | 12.0   | mm   | TPS sensor boss radius                    |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle } = replicad;

  const plenumLength      = 220.0;
  const plenumWidth       = 110.0;
  const plenumHeight      = 90.0;
  const plenumWallThick   = 4.5;
  const runnerRadius      = 24.0;
  const runnerWallThick   = 3.5;
  const runnerLength      = 230.0;
  const runnerSpacing     = 50.0;
  const flangeWidth       = 60.0;
  const flangeHeight      = 55.0;
  const flangeThick       = 9.0;
  const gasketGrooveWidth = 2.5;
  const gasketGrooveDepth = 1.8;
  const flangeBoltRadius  = 4.0;
  const tbBodyRadius      = 40.0;
  const tbBoreRadius      = 30.0;
  const tbLength          = 90.0;
  const butterflyRadius   = 29.0;
  const butterflyThick    = 3.0;
  const mapBossRadius     = 10.0;
  const mapBossHeight     = 18.0;
  const vacuumPortRadius  = 5.5;
  const tpsBossRadius     = 12.0;

  // ── PLENUM CHAMBER ────────────────────────────────────────
  const plenumOuter = drawRectangle(plenumLength, plenumWidth)
    .sketchOnPlane("XY", 0)
    .extrude(plenumHeight);
  const plenumInner = drawRectangle(plenumLength - plenumWallThick * 2, plenumWidth - plenumWallThick * 2)
    .sketchOnPlane("XY", plenumWallThick)
    .extrude(plenumHeight - plenumWallThick);
  const plenum = plenumOuter.cut(plenumInner).fillet(6, e => e.inPlane("XY", 0));

  // Throttle body inlet boss on plenum top
  const tbInletBoss = drawCircle(tbBodyRadius + 5)
    .sketchOnPlane("XY", plenumHeight)
    .extrude(15)
    .cut(drawCircle(tbBoreRadius)
      .sketchOnPlane("XY", plenumHeight)
      .extrude(15));

  // MAP sensor boss on plenum side
  const mapBoss = drawCircle(mapBossRadius)
    .sketchOnPlane("XZ", plenumWidth / 2)
    .extrude(mapBossHeight)
    .translateX(plenumLength * 0.2)
    .translateZ(plenumHeight * 0.55)
    .cut(drawCircle(mapBossRadius - 4)
      .sketchOnPlane("XZ", plenumWidth / 2)
      .extrude(mapBossHeight - 6)
      .translateX(plenumLength * 0.2)
      .translateZ(plenumHeight * 0.55));

  // Vacuum port boss
  const vacPort = drawCircle(vacuumPortRadius + 3)
    .sketchOnPlane("XZ", plenumWidth / 2)
    .extrude(14)
    .translateX(-plenumLength * 0.25)
    .translateZ(plenumHeight * 0.45)
    .cut(drawCircle(vacuumPortRadius)
      .sketchOnPlane("XZ", plenumWidth / 2)
      .extrude(14)
      .translateX(-plenumLength * 0.25)
      .translateZ(plenumHeight * 0.45));

  let manifold = plenum.fuse(tbInletBoss).fuse(mapBoss).fuse(vacPort);

  // ── RUNNERS (4 cylinders) ─────────────────────────────────
  const runnerXOffsets = [
    -runnerSpacing * 1.5,
    -runnerSpacing * 0.5,
     runnerSpacing * 0.5,
     runnerSpacing * 1.5,
  ];

  runnerXOffsets.forEach(xOff => {
    // Runner tube
    const runnerOuter = drawCircle(runnerRadius + runnerWallThick)
      .sketchOnPlane("XZ", plenumWidth / 2)
      .extrude(runnerLength)
      .translateX(xOff);
    const runnerBore = drawCircle(runnerRadius)
      .sketchOnPlane("XZ", plenumWidth / 2)
      .extrude(runnerLength)
      .translateX(xOff);

    // Cylinder head flange
    const flange = drawRectangle(flangeWidth, flangeHeight)
      .sketchOnPlane("XZ", plenumWidth / 2 + runnerLength)
      .extrude(flangeThick)
      .translateX(xOff - flangeWidth / 2)
      .translateZ(-flangeHeight / 2 + runnerRadius);

    const flangePortBore = drawCircle(runnerRadius)
      .sketchOnPlane("XZ", plenumWidth / 2 + runnerLength)
      .extrude(flangeThick)
      .translateX(xOff);

    // Gasket groove around port
    const gasketGroove = drawCircle(runnerRadius + 5)
      .sketchOnPlane("XZ", plenumWidth / 2 + runnerLength + flangeThick - gasketGrooveDepth)
      .extrude(gasketGrooveWidth)
      .cut(drawCircle(runnerRadius + 2)
        .sketchOnPlane("XZ", plenumWidth / 2 + runnerLength + flangeThick - gasketGrooveDepth)
        .extrude(gasketGrooveWidth))
      .translateX(xOff);

    // 4×M8 flange bolts
    const boltPositions = [
      [xOff - flangeWidth * 0.35, -flangeHeight * 0.35 + runnerRadius],
      [xOff + flangeWidth * 0.35, -flangeHeight * 0.35 + runnerRadius],
      [xOff - flangeWidth * 0.35,  flangeHeight * 0.35 + runnerRadius],
      [xOff + flangeWidth * 0.35,  flangeHeight * 0.35 + runnerRadius],
    ];
    let flangeWithBolts = flange.cut(flangePortBore).cut(gasketGroove);
    boltPositions.forEach(([bx, bz]) => {
      const boltHole = drawCircle(flangeBoltRadius)
        .sketchOnPlane("XZ", plenumWidth / 2 + runnerLength)
        .extrude(flangeThick)
        .translateX(bx).translateZ(bz);
      flangeWithBolts = flangeWithBolts.cut(boltHole);
    });

    manifold = manifold.fuse(runnerOuter.cut(runnerBore)).fuse(flangeWithBolts);
  });

  // ── THROTTLE BODY ─────────────────────────────────────────
  const tbHousing = drawCircle(tbBodyRadius)
    .sketchOnPlane("XY", plenumHeight + 15)
    .extrude(tbLength)
    .cut(drawCircle(tbBoreRadius)
      .sketchOnPlane("XY", plenumHeight + 15)
      .extrude(tbLength));

  // TPS boss on throttle body side
  const tpsBoss = drawCircle(tpsBossRadius)
    .sketchOnPlane("XZ", tbBodyRadius)
    .extrude(16)
    .translateZ(plenumHeight + 15 + tbLength * 0.5)
    .cut(drawCircle(tpsBossRadius - 4)
      .sketchOnPlane("XZ", tbBodyRadius)
      .extrude(10)
      .translateZ(plenumHeight + 15 + tbLength * 0.5));

  const throttleBody = tbHousing.fuse(tpsBoss);

  // ── BUTTERFLY VALVE ───────────────────────────────────────
  const butterfly = drawCircle(butterflyRadius)
    .sketchOnPlane("XY", plenumHeight + 15 + tbLength * 0.5)
    .extrude(butterflyThick)
    .translateZ(-butterflyThick / 2);

  return [
    { shape: manifold,     name: "Polymer Intake Manifold", color: "#E8E0D0" },
    { shape: throttleBody, name: "Aluminum Throttle Body",  color: "#A8B8C8" },
    { shape: butterfly,    name: "Butterfly Valve",         color: "#708090" },
  ];
};
```
