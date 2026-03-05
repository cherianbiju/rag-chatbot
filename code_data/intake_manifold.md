---
source_file: intake_manifold.js
category: engine
type: annotated_code
use_case: distributes air-fuel mixture equally to all cylinder intake ports
related: runner_tube.md, throttle_body.md
---
# Intake Manifold Body

## Description
A 4-runner polymer intake manifold plenum with a central throttle body flange, four runner outlets to cylinder head ports, MAP sensor boss, and vacuum port. Injection-molded nylon-reinforced construction.

## Keywords
intake manifold, plenum, runner, throttle body flange, MAP sensor, vacuum port, cylinder head, air distribution, extrude, fuse, cut, box, cylinder, flange, polymer

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| PLENUM_LENGTH | 280 | mm | plenum box length |
| PLENUM_WIDTH | 120 | mm | plenum box width |
| PLENUM_HEIGHT | 80 | mm | plenum box height |
| WALL_THICK | 4 | mm | wall thickness |
| RUNNER_SPACING | 90 | mm | center-to-center runner spacing |
| RUNNER_R | 22 | mm | runner bore radius |
| TB_BORE_R | 30 | mm | throttle body bore radius |
| MAP_BOSS_R | 8 | mm | MAP sensor boss radius |
| MAP_BOSS_H | 15 | mm | MAP sensor boss height |

## Code
```javascript
const main = (replicad) => {
  const {
    makeBaseBox,
    makeCylinder,
  } = replicad;

  const PLENUM_LENGTH  = 280;
  const PLENUM_WIDTH   = 120;
  const PLENUM_HEIGHT  = 80;
  const WALL_THICK     = 4;
  const RUNNER_SPACING = 90;
  const RUNNER_R       = 22;
  const TB_BORE_R      = 30;
  const MAP_BOSS_R     = 8;
  const MAP_BOSS_H     = 15;
  const NUM_RUNNERS    = 4;

  // Plenum outer shell
  let manifold = makeBaseBox(PLENUM_LENGTH, PLENUM_WIDTH, PLENUM_HEIGHT);

  // Hollow interior
  const interior = makeBaseBox(
    PLENUM_LENGTH - WALL_THICK * 2,
    PLENUM_WIDTH - WALL_THICK * 2,
    PLENUM_HEIGHT - WALL_THICK
  ).translate(WALL_THICK, WALL_THICK, WALL_THICK);
  manifold = manifold.cut(interior);

  // Runner outlets on bottom face
  const RUNNER_START_X = (PLENUM_LENGTH - RUNNER_SPACING * (NUM_RUNNERS - 1)) / 2;
  for (let i = 0; i < NUM_RUNNERS; i++) {
    const xPos = RUNNER_START_X + i * RUNNER_SPACING;
    const runnerBore = makeCylinder(RUNNER_R, WALL_THICK + 2, [xPos, PLENUM_WIDTH / 2, -1], [0, 0, 1]);
    manifold = manifold.cut(runnerBore);
  }

  // Throttle body flange on top
  const tbFlange = makeBaseBox(80, 80, 10).translate(PLENUM_LENGTH / 2 - 40, PLENUM_WIDTH / 2 - 40, PLENUM_HEIGHT);
  manifold = manifold.fuse(tbFlange);
  const tbBore = makeCylinder(TB_BORE_R, WALL_THICK + 12, [PLENUM_LENGTH / 2, PLENUM_WIDTH / 2, PLENUM_HEIGHT - 2], [0, 0, 1]);
  manifold = manifold.cut(tbBore);

  // MAP sensor boss
  const mapBoss = makeCylinder(MAP_BOSS_R, MAP_BOSS_H, [PLENUM_LENGTH * 0.7, PLENUM_WIDTH - 10, PLENUM_HEIGHT], [0, 0, 1]);
  manifold = manifold.fuse(mapBoss);
  const mapBore = makeCylinder(MAP_BOSS_R - 3, MAP_BOSS_H + 2, [PLENUM_LENGTH * 0.7, PLENUM_WIDTH - 10, PLENUM_HEIGHT - 1], [0, 0, 1]);
  manifold = manifold.cut(mapBore);

  // Vacuum port on side
  const vacPort = makeCylinder(5, 20, [0, PLENUM_WIDTH * 0.3, PLENUM_HEIGHT * 0.6], [-1, 0, 0]);
  manifold = manifold.cut(vacPort);

  return { shape: manifold, name: "Intake Manifold", color: "dimgrey" };
};
```
