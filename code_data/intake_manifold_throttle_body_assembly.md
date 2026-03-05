---
source_file: intake_manifold_throttle_body_assembly.md
category: assembly
type: annotated_code
use_case: distributes air-fuel mixture from throttle body to engine cylinders via individual runners
related: camshaft_assembly.md, engine_block.md, turbocharger_assembly.md
---

# Intake Manifold and Throttle Body Assembly

## Description
A 4-cylinder intake manifold featuring a plenum chamber that receives air from the throttle body and distributes it through four equal-length runners to the cylinder head ports. The throttle body contains a butterfly valve disc that controls airflow volume based on accelerator pedal position. Optimised runner length ensures equal charge distribution and tuned inertia charging.

## Keywords
intake manifold, throttle body, plenum, runner, inlet port, butterfly valve, airflow, charge distribution, 4-cylinder, air intake, manifold runner, throttle plate, induction, volumetric efficiency, engine breathing

## Parameters
| Variable          | Value | Unit | Meaning                          |
|-------------------|-------|------|----------------------------------|
| plenumWidth       | 200   | mm   | Plenum chamber width             |
| plenumHeight      | 80    | mm   | Plenum chamber height            |
| plenumDepth       | 100   | mm   | Plenum chamber depth             |
| runnerRadius      | 22    | mm   | Intake runner bore radius        |
| runnerLength      | 130   | mm   | Runner length                    |
| runnerSpacing     | 48    | mm   | Centre-to-centre runner spacing  |
| throttleBodyRadius| 38    | mm   | Throttle body bore radius        |
| throttleBodyLength| 80    | mm   | Throttle body housing length     |
| butterflyRadius   | 36    | mm   | Butterfly valve disc radius      |
| butterflyThickness| 3     | mm   | Butterfly valve disc thickness   |
| flangeThickness   | 8     | mm   | Mounting flange thickness        |
| flangeWidth       | 55    | mm   | Flange width per runner          |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle } = replicad;

  const plenumWidth        = 200;
  const plenumHeight       = 80;
  const plenumDepth        = 100;
  const runnerRadius       = 22;
  const runnerLength       = 130;
  const runnerSpacing      = 48;
  const throttleBodyRadius = 38;
  const throttleBodyLength = 80;
  const butterflyRadius    = 36;
  const butterflyThickness = 3;
  const flangeThickness    = 8;
  const flangeWidth        = 55;

  // ── PLENUM CHAMBER ────────────────────────────────────────
  const plenumOuter = drawRectangle(plenumWidth, plenumDepth)
    .sketchOnPlane("XY", 0)
    .extrude(plenumHeight);

  const plenumInner = drawRectangle(plenumWidth - 16, plenumDepth - 16)
    .sketchOnPlane("XY", flangeThickness)
    .extrude(plenumHeight - flangeThickness);

  const plenum = plenumOuter.cut(plenumInner);

  // ── INTAKE RUNNERS (4 cylinders) ─────────────────────────
  const runnerOffsets = [
    -runnerSpacing * 1.5,
    -runnerSpacing * 0.5,
     runnerSpacing * 0.5,
     runnerSpacing * 1.5,
  ];

  let manifold = plenum;
  const runners = [];

  runnerOffsets.forEach(xOffset => {
    const runnerOuter = drawCircle(runnerRadius + 4)
      .sketchOnPlane("XZ", plenumDepth / 2)
      .extrude(runnerLength)
      .translateX(xOffset);

    const runnerBore = drawCircle(runnerRadius)
      .sketchOnPlane("XZ", plenumDepth / 2)
      .extrude(runnerLength)
      .translateX(xOffset);

    // Mounting flange at cylinder head end
    const flange = drawRectangle(flangeWidth, flangeThickness)
      .sketchOnPlane("XZ", plenumDepth / 2 + runnerLength)
      .extrude(flangeWidth)
      .translateX(xOffset - flangeWidth / 2)
      .translateY(-flangeWidth / 2);

    const flangePortBore = drawCircle(runnerRadius)
      .sketchOnPlane("XZ", plenumDepth / 2 + runnerLength)
      .extrude(flangeThickness)
      .translateX(xOffset);

    const runner = runnerOuter.cut(runnerBore)
      .fuse(flange.cut(flangePortBore));

    runners.push(runner);
    manifold = manifold.fuse(runner);
  });

  // Throttle body inlet boss on plenum top
  const tbBoss = drawCircle(throttleBodyRadius + 6)
    .sketchOnPlane("XY", plenumHeight)
    .extrude(20);

  const tbBore = drawCircle(throttleBodyRadius)
    .sketchOnPlane("XY", plenumHeight)
    .extrude(20);

  manifold = manifold.fuse(tbBoss.cut(tbBore));

  // ── THROTTLE BODY ─────────────────────────────────────────
  const tbHousing = drawCircle(throttleBodyRadius + 8)
    .sketchOnPlane("XY", plenumHeight + 20)
    .extrude(throttleBodyLength);

  const tbBoreFull = drawCircle(throttleBodyRadius)
    .sketchOnPlane("XY", plenumHeight + 20)
    .extrude(throttleBodyLength);

  const throttleBody = tbHousing.cut(tbBoreFull);

  // ── BUTTERFLY VALVE DISC ──────────────────────────────────
  const butterfly = drawCircle(butterflyRadius)
    .sketchOnPlane("XY", plenumHeight + 20 + throttleBodyLength / 2)
    .extrude(butterflyThickness)
    .translateZ(-butterflyThickness / 2);

  return [
    { shape: manifold,     name: "Intake Manifold",  color: "#C0C0C0" },
    { shape: throttleBody, name: "Throttle Body",    color: "#2F4F4F" },
    { shape: butterfly,    name: "Butterfly Valve",  color: "#708090" },
  ];
};
```
