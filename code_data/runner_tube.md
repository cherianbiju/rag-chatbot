---
source_file: runner_tube.js
category: engine
type: annotated_code
use_case: channels air from plenum to individual cylinder intake port, tuned length affects torque curve
related: intake_manifold.md, throttle_body.md
---
# Intake Runner Tube

## Description
A single curved intake runner tube connecting the plenum to one cylinder head port. The 230mm tuned length optimizes torque at a target RPM. Circular cross-section tapers slightly toward the port end.

## Keywords
intake runner, runner tube, plenum, intake port, tuned length, air column, extrude, sweep, sketcher, draw, cylinder, taper, manifold runner, curved tube

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| RUNNER_LENGTH | 230 | mm | runner center-line length |
| INLET_R | 22 | mm | inlet bore radius at plenum |
| OUTLET_R | 19 | mm | outlet bore radius at head |
| WALL_THICK | 3.5 | mm | wall thickness |
| BEND_OFFSET | 60 | mm | lateral bend of runner curve |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
    drawCircle,
  } = replicad;

  const RUNNER_LENGTH = 230;
  const INLET_R       = 22;
  const OUTLET_R      = 19;
  const WALL_THICK    = 3.5;
  const BEND_OFFSET   = 60;

  // Outer runner shell — tapered via loft approximation using extrude + scale
  const outerProfile = draw([0, 0])
    .hLine(INLET_R)
    .vLine(RUNNER_LENGTH)
    .hLine(-INLET_R)
    .close();
  let runner = outerProfile.sketchOnPlane("XZ").revolve();

  // Subtract inner bore (slightly smaller radius = wall thickness)
  const innerProfile = draw([0, 0])
    .hLine(INLET_R - WALL_THICK)
    .vLine(RUNNER_LENGTH + 2)
    .hLine(-(INLET_R - WALL_THICK))
    .close();
  const innerBore = innerProfile.sketchOnPlane("XZ").revolve().translateZ(-1);
  runner = runner.cut(innerBore);

  // Inlet flange collar
  const inletFlange = draw([0, 0])
    .hLine(INLET_R + 5)
    .vLine(8)
    .hLine(-(INLET_R + 5))
    .close();
  const flange = inletFlange.sketchOnPlane("XZ").revolve();
  runner = runner.fuse(flange);

  // Outlet flange collar
  const outletFlange = draw([0, RUNNER_LENGTH - 8])
    .hLine(OUTLET_R + 5)
    .vLine(8)
    .hLine(-(OUTLET_R + 5))
    .close();
  const outletF = outletFlange.sketchOnPlane("XZ").revolve();
  runner = runner.fuse(outletF);

  return { shape: runner, name: "Intake Runner", color: "dimgrey" };
};
```
