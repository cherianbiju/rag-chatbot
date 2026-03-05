---
source_file: wrist_pin.js
category: engine
type: annotated_code
use_case: connects piston to connecting rod small end, allowing relative rotation
related: piston.md, connecting_rod.md
---
# Wrist Pin

## Description
A hollow hardened steel cylinder that pivots through the piston bosses and connecting rod small end. Also called gudgeon pin or piston pin. The hollow bore reduces mass while retaining bending strength.

## Keywords
wrist pin, gudgeon pin, piston pin, hollow cylinder, hardened steel, piston boss, small end, revolve, cylinder, bore, engine

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PIN_OUTER_R | 11 | mm | outer radius |
| PIN_INNER_R | 6 | mm | inner bore radius |
| PIN_LENGTH | 58 | mm | total pin length |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
  } = replicad;

  const PIN_OUTER_R = 11;
  const PIN_INNER_R = 6;
  const PIN_LENGTH  = 58;

  const outer = draw([0, 0])
    .hLine(PIN_OUTER_R)
    .vLine(PIN_LENGTH)
    .hLine(-PIN_OUTER_R)
    .close();

  let pin = outer.sketchOnPlane("XZ").revolve();

  const inner = draw([0, 0])
    .hLine(PIN_INNER_R)
    .vLine(PIN_LENGTH)
    .hLine(-PIN_INNER_R)
    .close();

  const bore = inner.sketchOnPlane("XZ").revolve();
  pin = pin.cut(bore);

  return { shape: pin, name: "Wrist Pin", color: "dimgrey" };
};
```
