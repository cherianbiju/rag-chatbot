---
source_file: camshaft_tappet_valve_assembly.md
category: assembly
type: annotated_code
use_case: controls intake and exhaust valve timing and lift in a 4-stroke internal combustion engine
related: intake_manifold_throttle_body_assembly.md, piston_crank_assembly.md, engine_block.md
---

# Camshaft, Tappet (Lifter) and Valve Assembly

## Description
A valvetrain assembly comprising a camshaft with multiple cam lobes, bucket tappets (hydraulic lifters) that ride on the cam lobes, and poppet valves with valve springs and retainers. As the camshaft rotates, the eccentric cam lobes push the tappets down, opening the valves against spring pressure. Valve timing determines engine breathing efficiency and performance characteristics.

## Keywords
camshaft, cam lobe, tappet, hydraulic lifter, bucket lifter, poppet valve, valve spring, retainer, valve stem, valvetrain, valve timing, lift, duration, overlap, intake valve, exhaust valve, overhead cam, engine timing

## Parameters
| Variable          | Value | Unit | Meaning                          |
|-------------------|-------|------|----------------------------------|
| camShaftRadius    | 18    | mm   | Camshaft base circle radius      |
| camShaftLength    | 360   | mm   | Camshaft total length            |
| camLobeRadius     | 28    | mm   | Cam lobe max radius (base+lift)  |
| camLobeWidth      | 20    | mm   | Cam lobe width                   |
| camLift           | 10    | mm   | Valve lift (lobe - base radius)  |
| liftCount         | 4     | -    | Number of cam lobes              |
| lobeSpacing       | 72    | mm   | Lobe centre spacing              |
| tappetRadius      | 20    | mm   | Tappet bucket outer radius       |
| tappetHeight      | 30    | mm   | Tappet height                    |
| valveStemRadius   | 5     | mm   | Valve stem radius                |
| valveStemLength   | 100   | mm   | Valve stem length                |
| valveHeadRadius   | 22    | mm   | Valve head radius                |
| valveHeadThick    | 4     | mm   | Valve head thickness             |
| springRadius      | 16    | mm   | Valve spring outer radius        |
| springHeight      | 55    | mm   | Valve spring free height         |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle } = replicad;

  const camShaftRadius = 18;
  const camShaftLength = 360;
  const camLobeRadius  = 28;
  const camLobeWidth   = 20;
  const liftCount      = 4;
  const lobeSpacing    = 72;
  const tappetRadius   = 20;
  const tappetHeight   = 30;
  const valveStemRadius= 5;
  const valveStemLength= 100;
  const valveHeadRadius= 22;
  const valveHeadThick = 4;
  const springRadius   = 16;
  const springHeight   = 55;

  // ── CAMSHAFT ──────────────────────────────────────────────
  const camShaftBase = drawCircle(camShaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(camShaftLength);

  // Cam lobes — eccentric circular profile
  const lobeOffsets = [40, 40 + lobeSpacing, 40 + lobeSpacing * 2, 40 + lobeSpacing * 3];
  let camshaft = camShaftBase;

  lobeOffsets.forEach((zOff, i) => {
    const lobe = draw([0, 0])
      .ellipseTo([camLobeRadius * 2, 0], camLobeRadius, camShaftRadius + 4, 0, false, true)
      .close()
      .sketchOnPlane("XY", zOff)
      .extrude(camLobeWidth);

    camshaft = camshaft.fuse(lobe);
  });

  // Bearing journals
  const journal1 = drawCircle(camShaftRadius + 4)
    .sketchOnPlane("XY", 0)
    .extrude(20);

  const journal2 = drawCircle(camShaftRadius + 4)
    .sketchOnPlane("XY", camShaftLength - 20)
    .extrude(20);

  camshaft = camshaft.fuse(journal1).fuse(journal2);

  // ── TAPPETS ───────────────────────────────────────────────
  const makeTappet = (zOffset) => {
    const outer = drawCircle(tappetRadius)
      .sketchOnPlane("XY", zOffset)
      .extrude(tappetHeight);
    const inner = drawCircle(tappetRadius - 3)
      .sketchOnPlane("XY", zOffset + 3)
      .extrude(tappetHeight - 3);
    return outer.cut(inner);
  };

  const tappet1 = makeTappet(lobeOffsets[0] + camLobeWidth + 2).translateY(camLobeRadius + tappetHeight / 2);
  const tappet2 = makeTappet(lobeOffsets[1] + camLobeWidth + 2).translateY(camLobeRadius + tappetHeight / 2);
  const tappet3 = makeTappet(lobeOffsets[2] + camLobeWidth + 2).translateY(camLobeRadius + tappetHeight / 2);
  const tappet4 = makeTappet(lobeOffsets[3] + camLobeWidth + 2).translateY(camLobeRadius + tappetHeight / 2);

  // ── VALVES ────────────────────────────────────────────────
  const makeValve = (zOffset) => {
    const stem = drawCircle(valveStemRadius)
      .sketchOnPlane("XY", zOffset)
      .extrude(valveStemLength);
    const head = drawCircle(valveHeadRadius)
      .sketchOnPlane("XY", zOffset)
      .extrude(valveHeadThick);
    return stem.fuse(head);
  };

  const valveYOffset = camLobeRadius + tappetHeight + valveStemLength * 0.1;
  const valve1 = makeValve(lobeOffsets[0]).translateY(valveYOffset);
  const valve2 = makeValve(lobeOffsets[1]).translateY(valveYOffset);
  const valve3 = makeValve(lobeOffsets[2]).translateY(valveYOffset);
  const valve4 = makeValve(lobeOffsets[3]).translateY(valveYOffset);

  // ── VALVE SPRINGS ─────────────────────────────────────────
  const makeSpring = (zOffset) => {
    const springOuter = drawCircle(springRadius)
      .sketchOnPlane("XY", zOffset + valveHeadThick)
      .extrude(springHeight);
    const springInner = drawCircle(springRadius - 3)
      .sketchOnPlane("XY", zOffset + valveHeadThick)
      .extrude(springHeight);
    return springOuter.cut(springInner);
  };

  const spring1 = makeSpring(lobeOffsets[0]).translateY(valveYOffset);
  const spring2 = makeSpring(lobeOffsets[1]).translateY(valveYOffset);
  const spring3 = makeSpring(lobeOffsets[2]).translateY(valveYOffset);
  const spring4 = makeSpring(lobeOffsets[3]).translateY(valveYOffset);

  return [
    { shape: camshaft, name: "Camshaft",   color: "#708090" },
    { shape: tappet1,  name: "Tappet 1",  color: "#A9A9A9" },
    { shape: tappet2,  name: "Tappet 2",  color: "#A9A9A9" },
    { shape: tappet3,  name: "Tappet 3",  color: "#A9A9A9" },
    { shape: tappet4,  name: "Tappet 4",  color: "#A9A9A9" },
    { shape: valve1,   name: "Valve 1",   color: "#C0C0C0" },
    { shape: valve2,   name: "Valve 2",   color: "#C0C0C0" },
    { shape: valve3,   name: "Valve 3",   color: "#C0C0C0" },
    { shape: valve4,   name: "Valve 4",   color: "#C0C0C0" },
    { shape: spring1,  name: "Spring 1",  color: "#B8860B" },
    { shape: spring2,  name: "Spring 2",  color: "#B8860B" },
    { shape: spring3,  name: "Spring 3",  color: "#B8860B" },
    { shape: spring4,  name: "Spring 4",  color: "#B8860B" },
  ];
};
```
