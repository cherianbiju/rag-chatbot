---
source_file: camshaft_tappet_valve_v2.md
category: assembly
type: annotated_code
use_case: valvetrain timing and lift control, chilled iron camshaft with hydraulic lifters and stainless steel valves
related: intake_manifold_throttle_body_v2.md, piston_crank_assembly_v2.md
---

# Camshaft, Tappet and Valve Assembly — Chilled Iron / Stainless Steel

## Description
A 4-lobe chilled iron camshaft (ground and hardened) with 8.5 mm lift, 110° lobe separation angle, timing sprocket interface, and oil feed passages. Hydraulic bucket tappets (lifter bore Ø25 mm ±0.02) ride on cam lobes. Stainless steel poppet valves (intake/exhaust) feature a 45° seat angle, valve stem, spring retainer groove and collet groove. Cam bearings pressed into block bores. Valve springs with retainers shown.

## Keywords
camshaft, cam lobe, hydraulic lifter, bucket tappet, valve lift 8.5mm, lobe separation 110°, valve seat 45°, lifter bore 25mm, chilled iron cam, stainless valve, valve spring, retainer, collet groove, timing sprocket, cam bearing, oil feed passage, intake valve, exhaust valve, overhead cam, valvetrain, engine timing

## Parameters
| Variable           | Value  | Unit | Meaning                               |
|--------------------|--------|------|---------------------------------------|
| camShaftRadius     | 20.0   | mm   | Camshaft base circle radius           |
| camShaftLength     | 380.0  | mm   | Camshaft total length                 |
| journalRadius      | 24.0   | mm   | Cam bearing journal radius            |
| journalWidth       | 20.0   | mm   | Journal width                         |
| lobeBaseRadius     | 20.0   | mm   | Cam lobe base circle radius           |
| lobeLiftRadius     | 28.5   | mm   | Cam lobe max radius (base + 8.5 lift) |
| lobeWidth          | 22.0   | mm   | Cam lobe face width                   |
| lobe1Z             | 42.0   | mm   | Lobe 1 Z position                     |
| lobeSpacing        | 74.0   | mm   | Lobe centre-to-centre spacing         |
| sprocketRadius     | 38.0   | mm   | Timing sprocket radius                |
| sprocketTeeth      | 19     | -    | Sprocket tooth count                  |
| sprocketWidth      | 16.0   | mm   | Sprocket width                        |
| oilHoleRadius      | 3.0    | mm   | Oil feed hole radius                  |
| tappetOuterRadius  | 12.5   | mm   | Tappet outer radius (bore Ø25mm / 2)  |
| tappetHeight       | 32.0   | mm   | Tappet total height                   |
| tappetWallThick    | 2.5    | mm   | Tappet wall thickness                 |
| valveStemRadius    | 4.0    | mm   | Valve stem radius                     |
| valveStemLength    | 105.0  | mm   | Valve stem length                     |
| valveHeadRadius    | 24.0   | mm   | Intake valve head radius              |
| exhValveHeadRadius | 20.0   | mm   | Exhaust valve head radius             |
| valveHeadThick     | 3.5    | mm   | Valve head thickness                  |
| seatAngleMm        | 3.0    | mm   | Seat taper face width                 |
| springOuterRadius  | 17.0   | mm   | Valve spring outer coil radius        |
| springHeight       | 48.0   | mm   | Valve spring free height              |
| retainerRadius     | 14.0   | mm   | Spring retainer radius                |
| retainerHeight     | 8.0    | mm   | Spring retainer height                |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawPolysides } = replicad;

  const camShaftRadius    = 20.0;
  const camShaftLength    = 380.0;
  const journalRadius     = 24.0;
  const journalWidth      = 20.0;
  const lobeBaseRadius    = 20.0;
  const lobeLiftRadius    = 28.5;
  const lobeWidth         = 22.0;
  const lobe1Z            = 42.0;
  const lobeSpacing       = 74.0;
  const sprocketRadius    = 38.0;
  const sprocketTeeth     = 19;
  const sprocketWidth     = 16.0;
  const oilHoleRadius     = 3.0;
  const tappetOuterRadius = 12.5;
  const tappetHeight      = 32.0;
  const tappetWallThick   = 2.5;
  const valveStemRadius   = 4.0;
  const valveStemLength   = 105.0;
  const valveHeadRadius   = 24.0;
  const exhValveHeadRadius= 20.0;
  const valveHeadThick    = 3.5;
  const springOuterRadius = 17.0;
  const springHeight      = 48.0;
  const retainerRadius    = 14.0;
  const retainerHeight    = 8.0;

  // ── CAMSHAFT BASE ─────────────────────────────────────────
  let camshaft = drawCircle(camShaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(camShaftLength);

  // Bearing journals (×3)
  const journalPositions = [0, camShaftLength * 0.45, camShaftLength - journalWidth];
  journalPositions.forEach(zPos => {
    const journal = drawCircle(journalRadius)
      .sketchOnPlane("XY", zPos)
      .extrude(journalWidth);
    camshaft = camshaft.fuse(journal);
  });

  // Cam lobes (×4) — eccentric profile
  const lobeZPositions = [
    lobe1Z,
    lobe1Z + lobeSpacing,
    lobe1Z + lobeSpacing * 2,
    lobe1Z + lobeSpacing * 3,
  ];
  // Lobe separation offsets (110° pattern for 4-cyl)
  const lobeAngles = [0, 180, 90, 270];
  lobeZPositions.forEach((zPos, i) => {
    const lobeProfile = draw([0, 0])
      .ellipseTo([lobeLiftRadius * 2, 0], lobeLiftRadius, lobeBaseRadius + 3, 0, false, true)
      .close()
      .rotate(lobeAngles[i]);
    const lobe = lobeProfile
      .sketchOnPlane("XY", zPos)
      .extrude(lobeWidth);
    camshaft = camshaft.fuse(lobe);
  });

  // Axial oil feed passages
  const oilGallery = drawCircle(oilHoleRadius)
    .sketchOnPlane("XY", 0)
    .extrude(camShaftLength)
    .translateX(camShaftRadius * 0.5);
  camshaft = camshaft.cut(oilGallery);

  // Timing sprocket at drive end
  const sprocketTeethShape = drawPolysides(sprocketRadius + 3, sprocketTeeth)
    .sketchOnPlane("XY", camShaftLength)
    .extrude(sprocketWidth);
  const sprocketBody = drawCircle(sprocketRadius)
    .sketchOnPlane("XY", camShaftLength)
    .extrude(sprocketWidth);
  const sprocketBore = drawCircle(camShaftRadius + 1)
    .sketchOnPlane("XY", camShaftLength)
    .extrude(sprocketWidth);
  const sprocket = sprocketTeethShape.intersect(sprocketBody).cut(sprocketBore);
  camshaft = camshaft.fuse(sprocket);

  // ── HYDRAULIC BUCKET TAPPETS ──────────────────────────────
  const makeTappet = (zOff, yOff) => {
    const outer = drawCircle(tappetOuterRadius)
      .sketchOnPlane("XY", zOff)
      .extrude(tappetHeight);
    const inner = drawCircle(tappetOuterRadius - tappetWallThick)
      .sketchOnPlane("XY", zOff + tappetWallThick)
      .extrude(tappetHeight - tappetWallThick * 2);
    return outer.cut(inner).translateY(yOff);
  };

  const tappetYOffset = lobeLiftRadius + tappetHeight + 2;
  const tappets = lobeZPositions.map(z => makeTappet(z + (lobeWidth - tappetHeight) / 2, tappetYOffset));

  // ── VALVES (intake stainless) ─────────────────────────────
  const makeValve = (headRadius, zOff, yOff) => {
    const stem = drawCircle(valveStemRadius)
      .sketchOnPlane("XY", zOff)
      .extrude(valveStemLength);
    // 45° seat taper head
    const head = draw([0, 0])
      .lineTo([headRadius, 0])
      .lineTo([headRadius - 3, valveHeadThick])
      .lineTo([0, valveHeadThick])
      .close()
      .sketchOnPlane("XZ", 0)
      .revolve()
      .translateZ(zOff);
    // Collet groove near stem top
    const colletGroove = drawCircle(valveStemRadius + 1)
      .sketchOnPlane("XY", zOff + valveStemLength - 10)
      .extrude(4)
      .cut(drawCircle(valveStemRadius - 1)
        .sketchOnPlane("XY", zOff + valveStemLength - 10)
        .extrude(4));
    return stem.fuse(head).fuse(colletGroove).translateY(yOff);
  };

  const valveYOffset = tappetYOffset + tappetHeight + 5;
  const intakeValves = lobeZPositions.map(z => makeValve(valveHeadRadius, z, valveYOffset));
  const exhaustValves = lobeZPositions.map(z => makeValve(exhValveHeadRadius, z, -valveYOffset));

  // ── VALVE SPRINGS + RETAINERS ─────────────────────────────
  const makeSpring = (zOff, yOff) => {
    const springBody = drawCircle(springOuterRadius)
      .sketchOnPlane("XY", zOff + valveHeadThick)
      .extrude(springHeight)
      .cut(drawCircle(springOuterRadius - 3)
        .sketchOnPlane("XY", zOff + valveHeadThick)
        .extrude(springHeight));
    const retainer = drawCircle(retainerRadius)
      .sketchOnPlane("XY", zOff + valveHeadThick + springHeight)
      .extrude(retainerHeight)
      .cut(drawCircle(valveStemRadius + 1)
        .sketchOnPlane("XY", zOff + valveHeadThick + springHeight)
        .extrude(retainerHeight));
    return springBody.fuse(retainer).translateY(yOff);
  };

  const intakeSprings  = lobeZPositions.map(z => makeSpring(z, valveYOffset));
  const exhaustSprings = lobeZPositions.map(z => makeSpring(z, -valveYOffset));

  return [
    { shape: camshaft,          name: "Chilled Iron Camshaft",    color: "#606870" },
    { shape: tappets[0],        name: "Hydraulic Tappet 1",       color: "#A8A8A8" },
    { shape: tappets[1],        name: "Hydraulic Tappet 2",       color: "#A8A8A8" },
    { shape: tappets[2],        name: "Hydraulic Tappet 3",       color: "#A8A8A8" },
    { shape: tappets[3],        name: "Hydraulic Tappet 4",       color: "#A8A8A8" },
    { shape: intakeValves[0],   name: "Intake Valve 1",           color: "#C0C0C0" },
    { shape: intakeValves[1],   name: "Intake Valve 2",           color: "#C0C0C0" },
    { shape: intakeValves[2],   name: "Intake Valve 3",           color: "#C0C0C0" },
    { shape: intakeValves[3],   name: "Intake Valve 4",           color: "#C0C0C0" },
    { shape: exhaustValves[0],  name: "Exhaust Valve 1",          color: "#B87050" },
    { shape: intakeSprings[0],  name: "Valve Spring 1",           color: "#D4AF37" },
    { shape: intakeSprings[1],  name: "Valve Spring 2",           color: "#D4AF37" },
    { shape: intakeSprings[2],  name: "Valve Spring 3",           color: "#D4AF37" },
    { shape: intakeSprings[3],  name: "Valve Spring 4",           color: "#D4AF37" },
  ];
};
```
