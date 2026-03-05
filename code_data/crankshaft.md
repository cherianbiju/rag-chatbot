---
source_file: crankshaft.js
category: engine
type: annotated_code
use_case: converts reciprocating piston motion to rotational output in internal combustion engines
related: connecting_rod.md, piston.md, wrist_pin.md
---
# Crankshaft

## Description
A forged steel crankshaft with main journals, crank throws, and counterweights. The throws offset from the main axis convert linear piston force into torque. Used in all reciprocating piston engines.

## Keywords
crankshaft, crank throw, main journal, crank pin, counterweight, engine, reciprocating, revolve, extrude, boolean, fuse, cylinder, offset journal, balance weight, forged steel

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| MAIN_JOURNAL_RADIUS | 25 | mm | radius of main bearing journals |
| CRANK_PIN_RADIUS | 20 | mm | radius of connecting rod journal |
| STROKE | 86 | mm | full piston stroke (throw = stroke/2) |
| JOURNAL_LENGTH | 28 | mm | length of each journal |
| THROW_WIDTH | 20 | mm | width of crank web/throw |
| COUNTERWEIGHT_RADIUS | 45 | mm | radius of counterweight |
| NUM_CYLINDERS | 4 | — | number of crank throws |

## Code
```javascript
const main = (replicad) => {
  const {
    makeCylinder,
    makeBaseBox,
  } = replicad;

  const MAIN_JOURNAL_RADIUS  = 25;
  const CRANK_PIN_RADIUS     = 20;
  const STROKE               = 86;
  const THROW                = STROKE / 2;
  const JOURNAL_LENGTH       = 28;
  const THROW_WIDTH          = 20;
  const COUNTERWEIGHT_RADIUS = 45;
  const COUNTERWEIGHT_THICK  = 18;
  const NUM_CYLINDERS        = 4;
  const SPACING              = JOURNAL_LENGTH + THROW_WIDTH * 2;

  // Build main shaft journals
  let crank = makeCylinder(MAIN_JOURNAL_RADIUS, JOURNAL_LENGTH, [0, 0, 0], [0, 0, 1]);

  for (let i = 0; i < NUM_CYLINDERS; i++) {
    const zBase = i * SPACING + JOURNAL_LENGTH;

    // Crank web left
    const webLeft = makeCylinder(COUNTERWEIGHT_RADIUS, THROW_WIDTH, [0, 0, zBase], [0, 0, 1]);
    crank = crank.fuse(webLeft);

    // Crank pin offset by throw
    const pinZ = zBase + THROW_WIDTH;
    const crankPin = makeCylinder(CRANK_PIN_RADIUS, JOURNAL_LENGTH, [THROW, 0, pinZ], [0, 0, 1]);
    crank = crank.fuse(crankPin);

    // Crank web right
    const webRight = makeCylinder(COUNTERWEIGHT_RADIUS, THROW_WIDTH, [0, 0, pinZ + JOURNAL_LENGTH], [0, 0, 1]);
    crank = crank.fuse(webRight);

    // Next main journal
    const nextJournal = makeCylinder(MAIN_JOURNAL_RADIUS, JOURNAL_LENGTH, [0, 0, pinZ + JOURNAL_LENGTH + THROW_WIDTH], [0, 0, 1]);
    crank = crank.fuse(nextJournal);
  }

  return { shape: crank, name: "Crankshaft", color: "steelblue" };
};
```
