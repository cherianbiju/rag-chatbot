---
source_file: cadquery-cycloidal-gear.js
category: gear
type: annotated_code
use_case: cycloidal gear, parametric gear, high torque transmission
related: cycloidal-gear.md, spur_gears.md, worm_bevel_gears.md
---

# Cadquery Cycloidal Gear

## Description
Parametric cycloidal gear generated using hypocycloid and epicycloid mathematical functions to define the tooth profile. The gear profile is created as a parametric curve, extruded with a twist angle to create a helical effect, and a center bore hole is cut. Based on the CadQuery cycloidal gear example.

## Keywords
cycloidal gear, hypocycloid, epicycloid, parametric function, drawParametricFunction, extrude, twistAngle, helical, bore hole, cut, gear profile, mathematical curve, involute alternative

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| height | 15 | mm | Height/thickness of gear |
| r1 | 6 | - | Outer radius ratio for gear profile |
| r2 | 1 | - | Inner radius ratio for gear profile |
| bore radius | 2 | mm | Center bore hole radius |
| twistAngle | 90 | deg | Twist applied during extrusion for helical effect |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawParametricFunction(fn) | Creates 2D profile from parametric math function |
| .sketchOnPlane() | Places parametric sketch on default XY plane |
| .extrude(h, {twistAngle}) | Extrudes with twist angle for helical gear |
| drawCircle(r) | Creates circular bore sketch |
| .cut(other) | Boolean subtract to create center bore |

## Code
```javascript
const { drawCircle, drawParametricFunction } = replicad;
const hypocycloid = (t,r1,r2) => [(r1-r2)*Math.cos(t)+r2*Math.cos((r1/r2)*t-t),(r1-r2)*Math.sin(t)+r2*Math.sin(-((r1/r2)*t-t))];
const epicycloid = (t,r1,r2) => [(r1+r2)*Math.cos(t)-r2*Math.cos((r1/r2)*t+t),(r1+r2)*Math.sin(t)-r2*Math.sin((r1/r2)*t+t)];
const gear = (t,r1=4,r2=1) => ((-1)**(1+Math.floor((t/2/Math.PI)*(r1/r2)))<0) ? epicycloid(t,r1,r2) : hypocycloid(t,r1,r2);
const defaultParams = { height: 15 };
const main = (r, { height }) => {
  const base = drawParametricFunction((t) => gear(2*Math.PI*t, 6, 1))
    .sketchOnPlane().extrude(height, { twistAngle: 90 });
  const hole = drawCircle(2).sketchOnPlane().extrude(height);
  return base.cut(hole);
};
```
