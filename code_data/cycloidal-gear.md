---
source_file: cycloidal-gear.js
category: gear
type: annotated_code
use_case: cycloidal gear, helical gear, high torque, precision drive
related: cadquery-cycloidal-gear.md, spur_gears.md
---

# Cycloidal Gear

## Description
Parametric cycloidal gear with 12 teeth generated from hypocycloid and epicycloid parametric equations. The gear profile is extruded with a 90° twist angle to produce a helical effect, and a center bore is cut. Uses higher point count (600) for smooth tooth profile compared to the CadQuery version.

## Keywords
cycloidal gear, hypocycloid, epicycloid, sketchParametricFunction, extrude, twistAngle, helical, bore, cut, parametric curve, gear profile, smooth, high resolution, 600 points

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| height | 30 | mm | Height/thickness of gear |
| r1 | 12 | - | Outer radius ratio (number of teeth = r1/r2) |
| r2 | 1 | - | Inner radius ratio |
| bore radius | 2 | mm | Centre bore radius |
| twistAngle | 90 | deg | Twist for helical effect |
| pointsCount | 600 | - | Resolution of parametric curve |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| sketchParametricFunction(fn, options) | Creates sketch from parametric math function with point count and range |
| .extrude(h, {twistAngle}) | Extrudes with helical twist |
| sketchCircle(r) | Creates circular bore sketch |
| .extrude(h) | Extrudes bore to height |
| .cut(other) | Boolean subtract for center bore |

## Code
```javascript
const hypocycloid = (t,r1,r2) => [(r1-r2)*Math.cos(t)+r2*Math.cos((r1/r2)*t-t),(r1-r2)*Math.sin(t)+r2*Math.sin(-((r1/r2)*t-t))];
const epicycloid = (t,r1,r2) => [(r1+r2)*Math.cos(t)-r2*Math.cos((r1/r2)*t+t),(r1+r2)*Math.sin(t)-r2*Math.sin((r1/r2)*t+t)];
const gear = (t,r1=12,r2=1) => ((-1)**(1+Math.floor((t/2/Math.PI)*(r1/r2)))<0) ? epicycloid(t,r1,r2) : hypocycloid(t,r1,r2);
const defaultParams = { height: 30 };
const main = ({ sketchCircle, sketchParametricFunction }, { height }) => {
  const base = sketchParametricFunction((t) => gear(2*Math.PI*t, 12, 1),
    {pointsCount:600, start:0, stop:1}).extrude(height, {twistAngle:90});
  const hole = sketchCircle(2).extrude(height);
  return base.cut(hole);
};
```
