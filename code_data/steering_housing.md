---
source_file: steering_housing.js
category: steering
type: annotated_code
use_case: encloses rack and pinion, mounts to subframe, provides fluid seal and rack guidance
related: steering_rack.md, pinion_gear.md, tie_rod.md
---
# Steering Housing / Tube

## Description
A cylindrical aluminum rack housing tube with pinion inlet bore, two bushing lands for rack guidance, and two subframe bracket flanges. Bellows grooves at each end seal the rack from contamination.

## Keywords
steering housing, rack tube, pinion bore, bushing land, subframe mount, bracket flange, bellows groove, cylinder, fuse, cut, extrude, draw, aluminum, rack and pinion

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| TUBE_LENGTH | 380 | mm | rack tube total length |
| TUBE_OUTER_R | 26 | mm | tube outer radius |
| TUBE_INNER_R | 16 | mm | rack bore radius |
| PINION_BORE_R | 22 | mm | pinion inlet bore radius |
| PINION_BORE_H | 50 | mm | pinion bore depth |
| MOUNT_FLANGE_W | 50 | mm | subframe bracket flange width |
| MOUNT_FLANGE_H | 30 | mm | bracket flange height |
| MOUNT_BOLT_R | 6 | mm | mounting bolt radius |
| BELLOWS_GROOVE_W | 8 | mm | bellows groove width |
| BELLOWS_GROOVE_D | 4 | mm | bellows groove depth |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
    makeCylinder,
    makeBaseBox,
  } = replicad;

  const TUBE_LENGTH    = 380;
  const TUBE_OUTER_R   = 26;
  const TUBE_INNER_R   = 16;
  const PINION_BORE_R  = 22;
  const PINION_BORE_H  = 50;
  const MOUNT_FLANGE_W = 50;
  const MOUNT_FLANGE_H = 30;
  const MOUNT_BOLT_R   = 6;
  const BELLOWS_W      = 8;
  const BELLOWS_D      = 4;

  // Main tube
  let housing = drawCircle(TUBE_OUTER_R).sketchOnPlane("YZ").extrude(TUBE_LENGTH);

  // Inner rack bore
  const innerBore = makeCylinder(TUBE_INNER_R, TUBE_LENGTH + 2, [0, 0, -1], [1, 0, 0]);
  housing = housing.cut(innerBore);

  // Pinion inlet bore on top
  const pinionBore = makeCylinder(PINION_BORE_R, PINION_BORE_H, [TUBE_LENGTH / 2, 0, 0], [0, 0, 1]);
  housing = housing.cut(pinionBore);

  // Subframe mount flanges
  const flange1 = makeBaseBox(TUBE_OUTER_R * 2, MOUNT_FLANGE_W, MOUNT_FLANGE_H)
    .translate(-TUBE_OUTER_R, -MOUNT_FLANGE_W / 2, -MOUNT_FLANGE_H)
    .translateX(TUBE_LENGTH * 0.2);
  const flange2 = makeBaseBox(TUBE_OUTER_R * 2, MOUNT_FLANGE_W, MOUNT_FLANGE_H)
    .translate(-TUBE_OUTER_R, -MOUNT_FLANGE_W / 2, -MOUNT_FLANGE_H)
    .translateX(TUBE_LENGTH * 0.8);
  housing = housing.fuse(flange1).fuse(flange2);

  // Mounting bolt holes in flanges
  const bolt1 = makeCylinder(MOUNT_BOLT_R, MOUNT_FLANGE_H + 2, [TUBE_LENGTH * 0.2, 0, -MOUNT_FLANGE_H - 1], [0, 0, 1]);
  const bolt2 = makeCylinder(MOUNT_BOLT_R, MOUNT_FLANGE_H + 2, [TUBE_LENGTH * 0.8, 0, -MOUNT_FLANGE_H - 1], [0, 0, 1]);
  housing = housing.cut(bolt1).cut(bolt2);

  // Bellows grooves at ends
  const groove1 = draw([TUBE_OUTER_R - BELLOWS_D, 10])
    .hLine(BELLOWS_D)
    .vLine(BELLOWS_W)
    .hLine(-BELLOWS_D)
    .close();
  const bellows1 = groove1.sketchOnPlane("YZ").revolve().translateX(10);
  housing = housing.cut(bellows1);

  const groove2 = draw([TUBE_OUTER_R - BELLOWS_D, 10])
    .hLine(BELLOWS_D)
    .vLine(BELLOWS_W)
    .hLine(-BELLOWS_D)
    .close();
  const bellows2 = groove2.sketchOnPlane("YZ").revolve().translateX(TUBE_LENGTH - 10 - BELLOWS_W);
  housing = housing.cut(bellows2);

  return { shape: housing, name: "Steering Housing", color: "silver" };
};
```
