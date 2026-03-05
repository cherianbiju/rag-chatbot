---
source_file: extrude.js
category: replicad_example
type: annotated_code
use_case: s-curve taper extrusion, endFactor taper, sagittaArc profile
related: extrude-examples.md, bezier-extrude.md
---

# Extrude (S-Curve Taper)

## Description
Creates an organic tapering shape by extruding a curved profile (built with hLine, vSagittaArc, and sagittaArc) using an s-curve extrusionProfile with endFactor 0.5 — meaning the top of the extrusion is 50% the size of the base. Minimal but powerful demonstration of replicad's extrusion profile options.

## Keywords
extrude, extrusionProfile, s-curve, endFactor, vSagittaArc, sagittaArc, taper, Sketcher, movePointerTo, hLine, close, organic shape, taper extrusion, scale

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| start point | [50,50] | mm | Starting corner of sketch |
| hLine | -120 | mm | Horizontal line length |
| vSagittaArc dy | -80 | mm | Vertical drop of arc |
| vSagittaArc sagitta | -20 | mm | Bulge of vertical arc |
| sagittaArc dx | 100 | mm | Horizontal span of arc |
| sagittaArc dy | 20 | mm | Vertical offset of arc end |
| sagittaArc sagitta | 60 | mm | Bulge of this arc |
| extrude height | 100 | mm | Total extrusion height |
| endFactor | 0.5 | - | Top is 50% scale of base |
| profile | s-curve | - | Smooth s-curve taper profile |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher() | Creates sketcher on default XY plane |
| .movePointerTo([x,y]) | Moves cursor to start point |
| .hLine(d) | Horizontal line segment |
| .vSagittaArc(dy, sagitta) | Vertical arc defined by endpoint offset and sagitta |
| .sagittaArc(dx, dy, sagitta) | Arc defined by endpoint offsets and sagitta |
| .close() | Closes the sketch |
| .extrude(h, options) | Extrudes with profile options |
| extrusionProfile: {profile, endFactor} | Controls taper shape and scale at top |

## Code
```javascript
const main = ({ Sketcher }) => {
  const shape = new Sketcher()
    .movePointerTo([50,50])
    .hLine(-120)
    .vSagittaArc(-80,-20)
    .sagittaArc(100,20,60)
    .close()
    .extrude(100, { extrusionProfile: { profile:"s-curve", endFactor:0.5 } });
  return shape;
};
```
