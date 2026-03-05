---
source_file: helical_gear.js
category: transmission
type: annotated_code
use_case: transmits torque between parallel shafts with smooth engagement in manual gearboxes
related: transmission_shaft.md, synchro_hub.md
---
# Helical Gear

## Description
A helical gear blank with hub bore, keyway, and tooth profile approximated as a polygon extrusion with twist. Used in manual transmission gearsets for smooth quiet operation. Module 2.0, 30 teeth.

## Keywords
helical gear, gear blank, module, teeth, hub bore, keyway, extrude, revolve, polygon, transmission, torque, power, shaft, involute, helical

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| MODULE | 2 | mm | gear module |
| NUM_TEETH | 30 | — | number of teeth |
| GEAR_WIDTH | 28 | mm | face width |
| HUB_R | 15 | mm | hub bore radius |
| OUTER_R | 31 | mm | tip circle radius (m*z/2 + m) |
| ROOT_R | 28 | mm | root circle radius |
| HUB_LENGTH | 35 | mm | hub extension length |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const MODULE     = 2;
  const NUM_TEETH  = 30;
  const GEAR_WIDTH = 28;
  const HUB_R      = 15;
  const OUTER_R    = (MODULE * NUM_TEETH) / 2 + MODULE;
  const ROOT_R     = (MODULE * NUM_TEETH) / 2 - 1.25 * MODULE;
  const HUB_LENGTH = 35;

  // Gear blank body — root circle extrude
  const gearBlank = drawCircle(ROOT_R).sketchOnPlane("XY").extrude(GEAR_WIDTH);

  // Approximate teeth as thin rectangular extrusions around circumference
  let gearBody = gearBlank;
  const TOOTH_WIDTH = (2 * Math.PI * ROOT_R) / NUM_TEETH * 0.5;
  const TOOTH_HEIGHT = OUTER_R - ROOT_R;

  for (let i = 0; i < NUM_TEETH; i++) {
    const angle = (i / NUM_TEETH) * 360;
    const tooth = draw([-TOOTH_WIDTH / 2, ROOT_R])
      .hLine(TOOTH_WIDTH)
      .vLine(TOOTH_HEIGHT)
      .hLine(-TOOTH_WIDTH)
      .close()
      .sketchOnPlane("XY")
      .extrude(GEAR_WIDTH)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    gearBody = gearBody.fuse(tooth);
  }

  // Hub extension
  const hubExt = drawCircle(HUB_R + 8).sketchOnPlane("XY").extrude(HUB_LENGTH).translateZ(-HUB_LENGTH + GEAR_WIDTH);
  gearBody = gearBody.fuse(hubExt);

  // Bore
  const bore = makeCylinder(HUB_R, HUB_LENGTH + GEAR_WIDTH + 2, [0, 0, -HUB_LENGTH + GEAR_WIDTH - 1], [0, 0, 1]);
  gearBody = gearBody.cut(bore);

  // Keyway
  const keyway = draw([-3, HUB_R - 3])
    .hLine(6)
    .vLine(5)
    .hLine(-6)
    .close()
    .sketchOnPlane("XY")
    .extrude(HUB_LENGTH + GEAR_WIDTH + 2)
    .translateZ(-HUB_LENGTH + GEAR_WIDTH - 1);
  gearBody = gearBody.cut(keyway);

  return { shape: gearBody, name: "Helical Gear", color: "steelblue" };
};
```
