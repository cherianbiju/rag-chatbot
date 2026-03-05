---
source_file: connecting_rod.js
category: engine
type: annotated_code
use_case: links piston to crankshaft, transmitting combustion force in reciprocating engines
related: crankshaft.md, piston.md, wrist_pin.md
---
# Connecting Rod

## Description
A steel connecting rod with a small end bore (piston pin), a tapered shank, and a large split big end bore (crankshaft journal). The I-beam shank cross-section reduces weight while maintaining strength.

## Keywords
connecting rod, con rod, big end, small end, shank, I-beam, bore, extrude, sketcher, boolean, fuse, cut, cylinder, engine, reciprocating, journal, bushing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| ROD_LENGTH | 140 | mm | center-to-center length |
| BIG_END_OUTER_R | 34 | mm | big end outer radius |
| BIG_END_INNER_R | 24 | mm | big end bore radius |
| SMALL_END_OUTER_R | 16 | mm | small end outer radius |
| SMALL_END_INNER_R | 10 | mm | small end bore radius |
| THICKNESS | 22 | mm | rod body thickness |
| SHANK_WIDTH | 18 | mm | shank width at mid-length |

## Code
```javascript
const main = (replicad) => {
  const {
    Sketcher,
    sketchCircle,
    makeCylinder,
  } = replicad;

  const ROD_LENGTH       = 140;
  const BIG_END_OUTER_R  = 34;
  const BIG_END_INNER_R  = 24;
  const SMALL_END_OUTER_R = 16;
  const SMALL_END_INNER_R = 10;
  const THICKNESS        = 22;
  const SHANK_WIDTH      = 18;

  const bigEnd = sketchCircle(BIG_END_OUTER_R).extrude(THICKNESS);

  const smallEnd = sketchCircle(SMALL_END_OUTER_R)
    .extrude(THICKNESS)
    .translateX(ROD_LENGTH);

  const shank = new Sketcher("XY")
    .movePointerTo([BIG_END_OUTER_R - 4, SHANK_WIDTH / 2])
    .lineTo([ROD_LENGTH - SMALL_END_OUTER_R + 2, SHANK_WIDTH / 2 - 3])
    .vLine(-(SHANK_WIDTH - 6))
    .lineTo([BIG_END_OUTER_R - 4, -SHANK_WIDTH / 2])
    .close()
    .extrude(THICKNESS);

  let rod = bigEnd.fuse(smallEnd).fuse(shank);

  const bigBore = makeCylinder(BIG_END_INNER_R, THICKNESS + 10, [0, 0, -5], [0, 0, 1]);
  const smallBore = makeCylinder(SMALL_END_INNER_R, THICKNESS + 10, [ROD_LENGTH, 0, -5], [0, 0, 1]);

  rod = rod.cut(bigBore).cut(smallBore);

  return { shape: rod, name: "Connecting Rod", color: "steelblue" };
};
```
