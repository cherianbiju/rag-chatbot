---
source_file: textplate.js
category: signage, hardware
type: annotated_code
use_case: engraved text plate, label, nameplate
related: drawText.md, drawRectangle.md
---

# Engraved Text Plate

## Description
Creates a thin rectangular plate with engraved text and a circular hole, suitable for nameplates, labels, or identification tags. The text is centered on the plate using bounding-box normalization, then combined with the hole cutout and subtracted from the plate via a single boolean cut operation.

## Keywords
text plate, drawText, engraving, drawRectangle, drawCircle, boolean cut, bounding box, center, extrude, nameplate, label, signage, fuse, sketch, 3D print

## Parameters
| Variable       | Value | Unit | Meaning                                      |
|----------------|-------|------|----------------------------------------------|
| plate width    | 5     | mm   | Width of the rectangular base plate          |
| plate height   | 5     | mm   | Height of the rectangular base plate         |
| plate depth    | 0.2   | mm   | Extrusion thickness of the plate             |
| hole radius    | 0.5   | mm   | Radius of the mounting / hanging hole        |
| fontSize       | 2     | mm   | Font size for the engraved text              |
| cutter depth   | 1.0   | mm   | Depth of the engraving cut into the plate    |

## Code
```javascript
// FILE: textplate.js
// Flat plate with engraved text and mounting hole.

const { draw, drawRectangle, drawCircle, drawText } = replicad;

const main = (r) => {

  // --- Helper: center a drawing around the origin using its bounding box ---
  function center(drawing) {
    const boundingBox = drawing.boundingBox;
    drawing = drawing.translate(
      -boundingBox.center[0],
      -boundingBox.center[1]
    );
    return drawing;
  }

  // --- Base plate: 5 × 5 mm, 0.2 mm thick ---
  let plate = drawRectangle(5, 5)
    .sketchOnPlane("XY")
    .extrude(0.2);

  // --- Cutter: circle hole + centered text, fused into one 2D profile ---
  let cutter = drawCircle(0.5)          // mounting hole at origin
    .fuse(
      center(drawText('test', { fontSize: 2 }))  // text centered by bounding box
        .translate(0, 0.8)             // shift text slightly upward from center
        .rotate(90)                    // rotate text upright (drawText is horizontal by default)
    );

  // Extrude cutter deeper than plate so it punches fully through
  cutter = cutter.sketchOnPlane("XY").extrude(1.0);

  // --- Boolean cut: engrave text and hole into plate ---
  plate = plate.cut(cutter);

  return plate;
};
```
