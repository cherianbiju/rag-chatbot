---
source_file: piston_crank_assembly.md
category: assembly
type: annotated_code
use_case: engine reciprocating mechanism, converts linear piston motion to rotary crankshaft motion
related: connecting_rod.md, engine_block.md, camshaft_assembly.md
---

# Piston Crank Assembly

## Description
A piston-crank mechanism consisting of a cylindrical piston with ring grooves, a connecting rod with big-end and small-end bores, and a simplified crankshaft with a single throw. This assembly represents the core reciprocating-to-rotary conversion unit found in internal combustion engines, compressors, and pumps.

## Keywords
piston, crankshaft, connecting rod, crank throw, big end bore, small end bore, ring groove, wrist pin, engine assembly, reciprocating motion, rotary motion, internal combustion, cylinder, bore, stroke

## Parameters
| Variable         | Value | Unit | Meaning                          |
|------------------|-------|------|----------------------------------|
| pistonRadius     | 40    | mm   | Piston outer radius              |
| pistonHeight     | 60    | mm   | Total piston height              |
| ringGrooveDepth  | 3     | mm   | Depth of each piston ring groove |
| ringGrooveWidth  | 4     | mm   | Width of each piston ring groove |
| pinBoreRadius    | 10    | mm   | Wrist pin bore radius            |
| rodLength        | 120   | mm   | Connecting rod centre-to-centre  |
| bigEndRadius     | 22    | mm   | Big end bore radius              |
| smallEndRadius   | 11    | mm   | Small end bore radius            |
| rodWidth         | 18    | mm   | Rod body width                   |
| rodThickness     | 12    | mm   | Rod body thickness               |
| crankRadius      | 30    | mm   | Crankshaft main journal radius   |
| crankThrow       | 50    | mm   | Crank throw (stroke/2)           |
| crankWebWidth    | 40    | mm   | Crank web width                  |
| crankWebThick    | 20    | mm   | Crank web thickness              |
| crankLength      | 80    | mm   | Total crankshaft length          |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle } = replicad;

  const pistonRadius    = 40;
  const pistonHeight    = 60;
  const ringGrooveDepth = 3;
  const ringGrooveWidth = 4;
  const pinBoreRadius   = 10;
  const rodLength       = 120;
  const bigEndRadius    = 22;
  const smallEndRadius  = 11;
  const rodWidth        = 18;
  const rodThickness    = 12;
  const crankRadius     = 30;
  const crankThrow      = 50;
  const crankWebWidth   = 40;
  const crankWebThick   = 20;
  const crankLength     = 80;

  // ── PISTON ───────────────────────────────────────────────
  const pistonOuter = drawCircle(pistonRadius)
    .sketchOnPlane("XY", 0)
    .extrude(pistonHeight);

  const groove1 = drawCircle(pistonRadius + 1)
    .sketchOnPlane("XY", pistonHeight * 0.7)
    .extrude(ringGrooveWidth)
    .cut(
      drawCircle(pistonRadius - ringGrooveDepth)
        .sketchOnPlane("XY", pistonHeight * 0.7)
        .extrude(ringGrooveWidth)
    );

  const groove2 = drawCircle(pistonRadius + 1)
    .sketchOnPlane("XY", pistonHeight * 0.55)
    .extrude(ringGrooveWidth)
    .cut(
      drawCircle(pistonRadius - ringGrooveDepth)
        .sketchOnPlane("XY", pistonHeight * 0.55)
        .extrude(ringGrooveWidth)
    );

  const pinBore = drawCircle(pinBoreRadius)
    .sketchOnPlane("XZ", pistonHeight * 0.25)
    .extrude(pistonRadius * 2)
    .translateX(-pistonRadius);

  const piston = pistonOuter
    .cut(groove1)
    .cut(groove2)
    .cut(pinBore);

  // ── CONNECTING ROD ────────────────────────────────────────
  const rodBody = drawRectangle(rodWidth, rodLength)
    .sketchOnPlane("XY", 0)
    .extrude(rodThickness);

  const bigEnd = drawCircle(bigEndRadius)
    .sketchOnPlane("XY", 0)
    .extrude(rodThickness);

  const bigEndBore = drawCircle(bigEndRadius - 6)
    .sketchOnPlane("XY", 0)
    .extrude(rodThickness);

  const smallEnd = drawCircle(smallEndRadius)
    .sketchOnPlane("XY", rodLength)
    .extrude(rodThickness);

  const smallEndBore = drawCircle(smallEndRadius - 4)
    .sketchOnPlane("XY", rodLength)
    .extrude(rodThickness);

  const connectingRod = rodBody
    .fuse(bigEnd)
    .fuse(smallEnd)
    .cut(bigEndBore)
    .cut(smallEndBore)
    .translateX(-rodWidth / 2)
    .translateY(-bigEndRadius)
    .translateZ(pistonHeight * 0.25 - rodThickness / 2);

  // ── CRANKSHAFT ────────────────────────────────────────────
  const mainJournal = drawCircle(crankRadius)
    .sketchOnPlane("XY", 0)
    .extrude(crankLength);

  const crankWeb1 = drawRectangle(crankWebWidth, crankWebThick)
    .sketchOnPlane("XY", crankLength * 0.3)
    .extrude(crankLength * 0.1)
    .translateX(-crankWebWidth / 2)
    .translateY(-crankWebThick / 2);

  const crankPin = drawCircle(bigEndRadius - 4)
    .sketchOnPlane("XY", crankLength * 0.3)
    .extrude(crankLength * 0.1)
    .translateY(crankThrow);

  const crankWeb2 = drawRectangle(crankWebWidth, crankWebThick)
    .sketchOnPlane("XY", crankLength * 0.4)
    .extrude(crankLength * 0.1)
    .translateX(-crankWebWidth / 2)
    .translateY(-crankWebThick / 2);

  const crankshaft = mainJournal
    .fuse(crankWeb1)
    .fuse(crankPin)
    .fuse(crankWeb2)
    .translateZ(-crankLength / 2);

  return [
    { shape: piston,        name: "Piston",         color: "#C0C0C0" },
    { shape: connectingRod, name: "Connecting Rod",  color: "#A0522D" },
    { shape: crankshaft,    name: "Crankshaft",      color: "#708090" },
  ];
};
```
