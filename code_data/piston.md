---
source_file: piston.js
category: engine
type: annotated_code
use_case: seals combustion chamber and transmits force to connecting rod in reciprocating engines
related: connecting_rod.md, wrist_pin.md, crankshaft.md
---
# Piston

## Description
An aluminum alloy piston with ring grooves, wrist pin bore, and valve reliefs. The crown seals combustion gases while the skirt guides the piston in the cylinder bore.

## Keywords
piston, crown, skirt, ring groove, wrist pin bore, compression ring, oil ring, cylinder, revolve, cut, boolean, engine, aluminum, bore, clearance

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| BORE | 86 | mm | cylinder bore diameter |
| PISTON_HEIGHT | 80 | mm | total piston height |
| CROWN_THICKNESS | 10 | mm | thickness of piston crown |
| SKIRT_HEIGHT | 40 | mm | height of piston skirt |
| WALL_THICKNESS | 6 | mm | piston wall thickness |
| PIN_BORE_R | 11 | mm | wrist pin bore radius |
| RING_GROOVE_DEPTH | 3.5 | mm | depth of compression ring groove |
| RING_GROOVE_WIDTH | 2 | mm | width of ring groove |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
  } = replicad;

  const BORE              = 86;
  const PISTON_R          = BORE / 2 - 0.04;
  const PISTON_HEIGHT     = 80;
  const CROWN_THICKNESS   = 10;
  const WALL_THICKNESS    = 6;
  const PIN_BORE_R        = 11;
  const RING_GROOVE_DEPTH = 3.5;
  const RING_GROOVE_WIDTH = 2;
  const RING1_Z           = PISTON_HEIGHT - CROWN_THICKNESS - 4;
  const RING2_Z           = RING1_Z - 6;
  const RING3_Z           = RING2_Z - 6;

  // Outer piston body — revolve profile
  const profile = draw([0, 0])
    .vLine(PISTON_HEIGHT)
    .hLine(PISTON_R)
    .vLine(-PISTON_HEIGHT)
    .close();

  let piston = profile.sketchOnPlane("XZ").revolve();

  // Hollow interior
  const interior = draw([0, 0])
    .vLine(PISTON_HEIGHT - CROWN_THICKNESS)
    .hLine(PISTON_R - WALL_THICKNESS)
    .vLine(-(PISTON_HEIGHT - CROWN_THICKNESS))
    .close();

  const hole = interior.sketchOnPlane("XZ").revolve();
  piston = piston.cut(hole);

  // Ring grooves
  const makeRingGroove = (z) => {
    const groove = draw([PISTON_R - RING_GROOVE_DEPTH, z])
      .hLine(RING_GROOVE_DEPTH)
      .vLine(RING_GROOVE_WIDTH)
      .hLine(-RING_GROOVE_DEPTH)
      .close();
    return groove.sketchOnPlane("XZ").revolve();
  };

  piston = piston.cut(makeRingGroove(RING1_Z));
  piston = piston.cut(makeRingGroove(RING2_Z));
  piston = piston.cut(makeRingGroove(RING3_Z));

  // Wrist pin bore — horizontal through piston
  const pinBore = makeCylinder(PIN_BORE_R, BORE + 10, [-BORE / 2 - 5, 0, PISTON_HEIGHT * 0.35], [1, 0, 0]);
  piston = piston.cut(pinBore);

  return { shape: piston, name: "Piston", color: "silver" };
};
```
