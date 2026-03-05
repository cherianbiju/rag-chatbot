---
source_file: throttle_body.js
category: engine
type: annotated_code
use_case: controls airflow into intake manifold via a rotating butterfly valve
related: intake_manifold.md, runner_tube.md
---
# Throttle Body

## Description
A 60mm aluminum throttle body housing with circular bore, butterfly valve disc, shaft bore, and four-bolt flange for manifold attachment. The butterfly plate rotates on a cross-shaft to vary airflow.

## Keywords
throttle body, butterfly valve, throttle plate, airflow, intake, flange, shaft bore, butterfly disc, revolve, draw, cylinder, extrude, cut, fuse, aluminum, 60mm bore

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| BORE_R | 30 | mm | throttle bore radius |
| BODY_LENGTH | 60 | mm | throttle body length |
| WALL_THICK | 8 | mm | housing wall thickness |
| FLANGE_THICK | 8 | mm | mounting flange thickness |
| FLANGE_SIZE | 80 | mm | flange square size |
| BOLT_OFFSET | 30 | mm | bolt hole offset from center |
| BOLT_R | 4.5 | mm | bolt hole radius |
| SHAFT_R | 5 | mm | butterfly shaft radius |
| PLATE_THICK | 3 | mm | butterfly plate thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeBaseBox,
    makeCylinder,
  } = replicad;

  const BORE_R       = 30;
  const BODY_LENGTH  = 60;
  const WALL_THICK   = 8;
  const FLANGE_THICK = 8;
  const FLANGE_SIZE  = 80;
  const BOLT_OFFSET  = 30;
  const BOLT_R       = 4.5;
  const SHAFT_R      = 5;
  const PLATE_THICK  = 3;
  const OUTER_R      = BORE_R + WALL_THICK;

  // Housing tube
  const profile = draw([BORE_R, 0])
    .hLine(WALL_THICK)
    .vLine(BODY_LENGTH)
    .hLine(-WALL_THICK)
    .close();
  let tb = profile.sketchOnPlane("XZ").revolve();

  // Inlet flange
  const flange = makeBaseBox(FLANGE_SIZE, FLANGE_SIZE, FLANGE_THICK)
    .translate(-FLANGE_SIZE / 2, -FLANGE_SIZE / 2, -FLANGE_THICK);
  const flangeBore = makeCylinder(BORE_R, FLANGE_THICK + 2, [0, 0, -FLANGE_THICK - 1], [0, 0, 1]);
  const flangeBody = flange.cut(flangeBore);
  tb = tb.fuse(flangeBody);

  // Flange bolt holes
  const offsets = [[BOLT_OFFSET, BOLT_OFFSET], [-BOLT_OFFSET, BOLT_OFFSET],
                   [BOLT_OFFSET, -BOLT_OFFSET], [-BOLT_OFFSET, -BOLT_OFFSET]];
  for (const [bx, by] of offsets) {
    const bolt = makeCylinder(BOLT_R, FLANGE_THICK + 2, [bx, by, -FLANGE_THICK - 1], [0, 0, 1]);
    tb = tb.cut(bolt);
  }

  // Butterfly shaft bore — horizontal through body
  const shaft = makeCylinder(SHAFT_R, OUTER_R * 2 + 4, [-OUTER_R - 2, 0, BODY_LENGTH / 2], [1, 0, 0]);
  tb = tb.cut(shaft);

  // Butterfly plate disc
  const plate = makeCylinder(BORE_R - 1, PLATE_THICK, [0, 0, BODY_LENGTH / 2 - PLATE_THICK / 2], [0, 0, 1]);
  tb = tb.fuse(plate);

  return { shape: tb, name: "Throttle Body", color: "silver" };
};
```
