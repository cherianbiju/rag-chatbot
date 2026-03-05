---
source_file: brake_pad.js
category: brakes
type: annotated_code
use_case: friction element pressed against rotor to generate braking force
related: brake_rotor.md, brake_caliper_housing.md
---
# Brake Pad

## Description
A steel backing plate with bonded friction material block. The pad sits in the caliper and is pressed against the rotor face by hydraulic pistons. Includes shim slot for anti-squeal shim.

## Keywords
brake pad, friction material, backing plate, shim, caliper, rotor, braking, box, extrude, cut, boolean, steel, friction block

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PAD_LENGTH | 120 | mm | length of pad |
| PAD_WIDTH | 55 | mm | width of pad |
| PLATE_THICKNESS | 6 | mm | steel backing plate thickness |
| FRICTION_THICKNESS | 14 | mm | friction material thickness |
| SHIM_SLOT_DEPTH | 1.5 | mm | anti-squeal shim slot depth |
| CORNER_NOTCH | 8 | mm | corner notch for caliper ears |

## Code
```javascript
const main = (replicad) => {
  const {
    makeBaseBox,
  } = replicad;

  const PAD_LENGTH         = 120;
  const PAD_WIDTH          = 55;
  const PLATE_THICKNESS    = 6;
  const FRICTION_THICKNESS = 14;
  const SHIM_SLOT_DEPTH    = 1.5;
  const CORNER_NOTCH       = 8;
  const TOTAL_THICKNESS    = PLATE_THICKNESS + FRICTION_THICKNESS;

  // Steel backing plate
  let plate = makeBaseBox(PAD_LENGTH, PAD_WIDTH, PLATE_THICKNESS);

  // Corner notches for caliper ears
  const notch1 = makeBaseBox(CORNER_NOTCH, CORNER_NOTCH, PLATE_THICKNESS + 2).translate(-1, -1, -1);
  const notch2 = makeBaseBox(CORNER_NOTCH, CORNER_NOTCH, PLATE_THICKNESS + 2).translate(PAD_LENGTH - CORNER_NOTCH + 1, -1, -1);
  const notch3 = makeBaseBox(CORNER_NOTCH, CORNER_NOTCH, PLATE_THICKNESS + 2).translate(-1, PAD_WIDTH - CORNER_NOTCH + 1, -1);
  const notch4 = makeBaseBox(CORNER_NOTCH, CORNER_NOTCH, PLATE_THICKNESS + 2).translate(PAD_LENGTH - CORNER_NOTCH + 1, PAD_WIDTH - CORNER_NOTCH + 1, -1);
  plate = plate.cut(notch1).cut(notch2).cut(notch3).cut(notch4);

  // Shim slot on back face
  const shimSlot = makeBaseBox(PAD_LENGTH - 20, PAD_WIDTH - 10, SHIM_SLOT_DEPTH).translate(10, 5, -SHIM_SLOT_DEPTH);
  plate = plate.cut(shimSlot);

  // Friction material block
  const friction = makeBaseBox(PAD_LENGTH - 10, PAD_WIDTH - 6, FRICTION_THICKNESS)
    .translate(5, 3, PLATE_THICKNESS);

  const assembly = plate.fuse(friction);

  return { shape: assembly, name: "Brake Pad", color: "dimgrey" };
};
```
