---
source_file: camshaft.js
category: engine
type: annotated_code
use_case: opens and closes intake and exhaust valves in sequence via rotating lobes
related: hydraulic_lifter.md, valve_stem.md, valve_spring.md
---
# Camshaft

## Description
A 4-lobe chilled iron camshaft with bearing journals between each lobe. The eccentric lobe profile pushes lifters to open valves. Lobe separation angle of 110° and 8.5mm lift. Timing sprocket interface at front.

## Keywords
camshaft, cam lobe, bearing journal, lobe separation, valve lift, timing sprocket, chilled iron, revolve, fuse, cylinder, eccentric, extrude, engine timing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| JOURNAL_R | 15 | mm | bearing journal radius |
| JOURNAL_LENGTH | 22 | mm | journal length |
| LOBE_BASE_R | 18 | mm | base circle radius |
| LOBE_LIFT | 8.5 | mm | cam lift (max - base radius) |
| LOBE_WIDTH | 16 | mm | lobe face width |
| NUM_LOBES | 4 | — | number of cam lobes |
| SPROCKET_R | 24 | mm | timing sprocket flange radius |
| SPROCKET_THICK | 10 | mm | sprocket flange thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const JOURNAL_R      = 15;
  const JOURNAL_LENGTH = 22;
  const LOBE_BASE_R    = 18;
  const LOBE_LIFT      = 8.5;
  const LOBE_WIDTH     = 16;
  const NUM_LOBES      = 4;
  const SPROCKET_R     = 24;
  const SPROCKET_THICK = 10;
  const SPACING        = JOURNAL_LENGTH + LOBE_WIDTH;

  // Build camshaft: alternating journals and lobes
  let cam = makeCylinder(JOURNAL_R, JOURNAL_LENGTH, [0, 0, 0], [0, 0, 1]);

  for (let i = 0; i < NUM_LOBES; i++) {
    const zBase = i * SPACING + JOURNAL_LENGTH;
    const lobeAngle = i * 90; // 4 lobes at 90 degree intervals

    // Lobe base circle
    const lobeBase = drawCircle(LOBE_BASE_R).sketchOnPlane("XY").extrude(LOBE_WIDTH).translateZ(zBase);
    cam = cam.fuse(lobeBase);

    // Lobe nose — eccentric bump
    const noseProfile = draw([0, LOBE_BASE_R])
      .hLine(LOBE_LIFT)
      .vLine(LOBE_WIDTH * 0.5)
      .hLine(-LOBE_LIFT)
      .close();
    const nose = noseProfile.sketchOnPlane("XY")
      .extrude(LOBE_WIDTH * 0.5)
      .translateZ(zBase + LOBE_WIDTH * 0.25)
      .rotate(lobeAngle, [0, 0, 0], [0, 0, 1]);
    cam = cam.fuse(nose);

    // Next journal
    const nextJournal = makeCylinder(JOURNAL_R, JOURNAL_LENGTH, [0, 0, zBase + LOBE_WIDTH], [0, 0, 1]);
    cam = cam.fuse(nextJournal);
  }

  // Timing sprocket flange at front
  const sprocket = drawCircle(SPROCKET_R).sketchOnPlane("XY").extrude(SPROCKET_THICK).translateZ(-SPROCKET_THICK);
  cam = cam.fuse(sprocket);

  return { shape: cam, name: "Camshaft", color: "dimgrey" };
};
```
