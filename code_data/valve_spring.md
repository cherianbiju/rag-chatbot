---
source_file: valve_spring.js
category: engine
type: annotated_code
use_case: returns valve to closed position after cam lobe releases lifter pressure
related: camshaft.md, hydraulic_lifter.md, valve_stem.md
---
# Valve Spring

## Description
A helical compression spring that seats between the cylinder head spring pocket and the valve spring retainer. Returns the valve to closed position. Modelled as a helical coil solid for visual representation.

## Keywords
valve spring, coil spring, helical spring, compression spring, spring retainer, valve seat, cylinder head, helix, revolve, torus, fuse, engine valve train

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| COIL_OUTER_R | 18 | mm | outer coil radius |
| WIRE_R | 2.5 | mm | wire cross-section radius |
| FREE_LENGTH | 50 | mm | free (uncompressed) length |
| NUM_COILS | 7 | — | number of active coils |
| PITCH | 7 | mm | coil pitch |

## Code
```javascript
const main = (replicad) => {
  const {
    makeHelix,
    drawCircle,
  } = replicad;

  const COIL_OUTER_R = 18;
  const WIRE_R       = 2.5;
  const FREE_LENGTH  = 50;
  const NUM_COILS    = 7;
  const PITCH        = FREE_LENGTH / NUM_COILS;
  const HELIX_R      = COIL_OUTER_R - WIRE_R;

  // makeHelix(pitch, height, radius) returns a wire path
  const helixPath  = makeHelix(PITCH, FREE_LENGTH, HELIX_R);

  // Sweep circular cross-section along helix
  const wireSection = drawCircle(WIRE_R);
  const spring = helixPath.sweepSketch((plane) => wireSection.sketchOnPlane(plane));

  return { shape: spring, name: "Valve Spring", color: "steelblue" };
};
```
