---
source_file: synchro_hub.js
category: transmission
type: annotated_code
use_case: synchronizes shaft and gear speeds before engagement in manual gearbox
related: helical_gear.md, transmission_shaft.md
---
# Synchro Hub

## Description
A splined synchronizer hub that slides on the transmission shaft to engage gears. The outer dog teeth engage the gear's dog ring while inner splines slide on the shaft. Includes detent groove for shift fork.

## Keywords
synchro hub, synchronizer, dog teeth, spline, shift fork, gearbox, engagement, transmission, extrude, revolve, cylinder, cut, fuse, manual transmission

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| HUB_INNER_R | 16 | mm | inner spline radius |
| HUB_BODY_R | 28 | mm | main hub outer radius |
| DOG_RING_R | 34 | mm | dog teeth outer radius |
| HUB_WIDTH | 24 | mm | total hub width |
| DOG_TOOTH_W | 3 | mm | width of each dog tooth |
| NUM_DOG_TEETH | 18 | — | number of dog teeth |
| FORK_GROOVE_R | 31 | mm | shift fork groove radius |
| FORK_GROOVE_W | 5 | mm | shift fork groove width |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
  } = replicad;

  const HUB_INNER_R   = 16;
  const HUB_BODY_R    = 28;
  const DOG_RING_R    = 34;
  const HUB_WIDTH     = 24;
  const DOG_TOOTH_W   = 3;
  const NUM_DOG_TEETH = 18;
  const FORK_GROOVE_R = 31;
  const FORK_GROOVE_W = 5;

  // Hub body
  let hub = drawCircle(HUB_BODY_R).sketchOnPlane("XY").extrude(HUB_WIDTH);

  // Inner bore
  const bore = makeCylinder(HUB_INNER_R, HUB_WIDTH + 2, [0, 0, -1], [0, 0, 1]);
  hub = hub.cut(bore);

  // Dog teeth around outer ring
  const TOOTH_H = DOG_RING_R - HUB_BODY_R;
  for (let i = 0; i < NUM_DOG_TEETH; i++) {
    const angle = (i / NUM_DOG_TEETH) * 360;
    const tooth = draw([-DOG_TOOTH_W / 2, HUB_BODY_R])
      .hLine(DOG_TOOTH_W)
      .vLine(TOOTH_H)
      .hLine(-DOG_TOOTH_W)
      .close()
      .sketchOnPlane("XY")
      .extrude(HUB_WIDTH)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    hub = hub.fuse(tooth);
  }

  // Shift fork groove
  const forkGroove = draw([FORK_GROOVE_R, (HUB_WIDTH - FORK_GROOVE_W) / 2])
    .hLine(HUB_BODY_R + 2 - FORK_GROOVE_R)
    .vLine(FORK_GROOVE_W)
    .hLine(-(HUB_BODY_R + 2 - FORK_GROOVE_R))
    .close();
  const groove = forkGroove.sketchOnPlane("XZ").revolve();
  hub = hub.cut(groove);

  return { shape: hub, name: "Synchro Hub", color: "slategrey" };
};
```
