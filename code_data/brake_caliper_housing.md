---
source_file: brake_caliper_housing.js
category: brakes
type: annotated_code
use_case: houses brake pistons and applies clamping force to brake pads on rotor
related: brake_rotor.md, brake_pad.md
---
# Brake Caliper Housing

## Description
A 4-piston aluminum monobloc brake caliper housing. The body spans the rotor with two piston bores on each side, a fluid channel bridge, and mounting boss flanges for attachment to the upright.

## Keywords
brake caliper, caliper housing, piston bore, mounting boss, fluid bridge, monobloc, aluminum, fuse, cut, cylinder, box, extrude, boolean, brakes, clamping

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| BODY_LENGTH | 140 | mm | length of caliper body |
| BODY_WIDTH | 60 | mm | total width across rotor |
| BODY_HEIGHT | 70 | mm | height of caliper body |
| BRIDGE_THICKNESS | 12 | mm | thickness of bridge over rotor |
| PISTON_BORE_R | 19 | mm | radius of piston bore |
| PISTON_DEPTH | 35 | mm | depth of piston bore |
| MOUNT_BOSS_R | 9 | mm | radius of mounting boss |
| ROTOR_SLOT_WIDTH | 34 | mm | slot width for rotor clearance |

## Code
```javascript
const main = (replicad) => {
  const {
    makeBaseBox,
    makeCylinder,
  } = replicad;

  const BODY_LENGTH      = 140;
  const BODY_WIDTH       = 60;
  const BODY_HEIGHT      = 70;
  const BRIDGE_THICKNESS = 12;
  const PISTON_BORE_R    = 19;
  const PISTON_DEPTH     = 35;
  const MOUNT_BOSS_R     = 9;
  const MOUNT_BOSS_H     = 20;
  const ROTOR_SLOT_WIDTH = 34;
  const ROTOR_SLOT_DEPTH = BODY_HEIGHT - BRIDGE_THICKNESS;

  // Main body block
  let body = makeBaseBox(BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT);

  // Rotor slot — cut through middle
  const rotorSlot = makeBaseBox(BODY_LENGTH + 10, ROTOR_SLOT_WIDTH, ROTOR_SLOT_DEPTH)
    .translate(-5, (BODY_WIDTH - ROTOR_SLOT_WIDTH) / 2, 0);
  body = body.cut(rotorSlot);

  // Piston bores — 2 on each side, front
  const boreOffsets = [BODY_LENGTH * 0.28, BODY_LENGTH * 0.72];
  for (const xOff of boreOffsets) {
    const bore1 = makeCylinder(PISTON_BORE_R, PISTON_DEPTH, [xOff, 0, BODY_HEIGHT / 2], [0, 1, 0]);
    const bore2 = makeCylinder(PISTON_BORE_R, PISTON_DEPTH, [xOff, BODY_WIDTH, BODY_HEIGHT / 2], [0, -1, 0]);
    body = body.cut(bore1).cut(bore2);
  }

  // Mounting bosses
  const boss1 = makeCylinder(MOUNT_BOSS_R, MOUNT_BOSS_H, [15, BODY_WIDTH / 2, BODY_HEIGHT], [0, 0, 1]);
  const boss2 = makeCylinder(MOUNT_BOSS_R, MOUNT_BOSS_H, [BODY_LENGTH - 15, BODY_WIDTH / 2, BODY_HEIGHT], [0, 0, 1]);
  body = body.fuse(boss1).fuse(boss2);

  // Mounting bolt holes
  const bolt1 = makeCylinder(5, MOUNT_BOSS_H + 5, [15, BODY_WIDTH / 2, BODY_HEIGHT - 2], [0, 0, 1]);
  const bolt2 = makeCylinder(5, MOUNT_BOSS_H + 5, [BODY_LENGTH - 15, BODY_WIDTH / 2, BODY_HEIGHT - 2], [0, 0, 1]);
  body = body.cut(bolt1).cut(bolt2);

  return { shape: body, name: "Brake Caliper Housing", color: "silver" };
};
```
