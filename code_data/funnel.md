---
source_file: funnel.js
category: structural
type: annotated_code
use_case: funnel, fluid transfer, revolve from profile, wall thickness
related: bottle.md, pipe_fittings.md
---

# Funnel

## Description
Parametric funnel created by drawing a 2D wall profile with inner and outer edges in one closed sketch, then revolving it 360° around the Z axis. Fillets are applied at the neck junction (stem-to-cone) and at the rim (top of cone). Clean minimal example of revolve-based hollow part design using draw() on XZ plane.

## Keywords
funnel, revolve, draw, lineTo, movePointerTo, close, fillet, inPlane, XZ plane, fluid transfer, parametric, hollow, wall thickness, stem, mouth, cone, solid of revolution

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| stemDiameter | 30 | mm | Outlet stem outer diameter |
| mouthDiameter | 100 | mm | Funnel mouth outer diameter |
| stemHeight | 30 | mm | Height of straight stem section |
| coneHeight | 40 | mm | Height of conical section |
| wallThickness | 2 | mm | Wall thickness (inner = outer - 2) |
| neckFilletRadius | 10 | mm | Fillet at stem-cone junction (Z=stemHeight) |
| rimFilletRadius | 0.5 | mm | Fillet at top rim (Z=totalHeight) |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw() | Starts 2D profile drawing at origin |
| .movePointerTo([x,y]) | Moves to first point of profile |
| .lineTo([x,y]) | Draws line to absolute point |
| .close() | Closes wall profile path |
| .sketchOnPlane("XZ") | Places profile on XZ plane for revolve around Z |
| .revolve() | Revolves 360° around Z axis |
| .fillet(r, edgeFinder) | Rounds edges at specific heights |
| .inPlane("XY", z) | Selects circular edges at height z |

## Code
```javascript
const main = (replicad) => {
  const { draw } = replicad;
  const stemDiameter=30, mouthDiameter=100, stemHeight=30;
  const coneHeight=40, wallThickness=2, neckFilletRadius=10, rimFilletRadius=0.5;
  const stemRadius=stemDiameter/2, mouthRadius=mouthDiameter/2;
  const totalHeight=stemHeight+coneHeight;
  const profile = draw()
    .movePointerTo([stemRadius-wallThickness,0])
    .lineTo([stemRadius-wallThickness,stemHeight])
    .lineTo([mouthRadius-wallThickness,totalHeight])
    .lineTo([mouthRadius,totalHeight])
    .lineTo([stemRadius,stemHeight])
    .lineTo([stemRadius,0])
    .close();
  let funnel = profile.sketchOnPlane("XZ").revolve();
  funnel = funnel.fillet(neckFilletRadius,(e)=>e.inPlane("XY",stemHeight));
  funnel = funnel.fillet(rimFilletRadius,(e)=>e.inPlane("XY",totalHeight));
  return funnel;
};
```
