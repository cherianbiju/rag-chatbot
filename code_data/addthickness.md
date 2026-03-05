---
source_file: addthickness.js
category: replicad_example
type: annotated_code
use_case: geometry experimentation, sketch operations
related: cannedSketches.md, bezier-extrude.md
---

# Add Thickness

## Description
Experiments with combining revolved and extruded sketches in replicad. Creates a box, a revolved arc profile, and a circular wheel shape then fuses them together into a single solid. Useful for understanding how different sketch operations combine.

## Keywords
revolve, extrude, fuse, Sketcher, sketchRectangle, sketchCircle, threePointsArc, hLine, translateY, boolean union

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| length | 60 | mm | Length of extruded rectangle |
| width | 15 | mm | Width of rectangle and revolved profile |
| height | 10 | mm | Height parameter |
| radius | 20 | mm | Radius of circular wheel sketch |
| fillet | 4 | mm | Arc fillet size in revolved profile |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XZ") | Creates a 2D sketch on the XZ plane |
| .hLine() | Draws a horizontal line segment |
| .threePointsArc() | Draws an arc through three points |
| .close() | Closes the sketch path |
| .revolve() | Revolves sketch 360° around axis |
| sketchRectangle() | Creates a rectangular sketch |
| .extrude() | Extrudes sketch into 3D solid |
| sketchCircle() | Creates a circular sketch |
| .translateY() | Moves shape along Y axis |
| .fuse() | Boolean union of two shapes |

## Code
```javascript
function main({ Sketcher, sketchRectangle, sketchCircle }) {
  let length = 60, width = 15, height = 10, radius = 20, fillet = 4;
  let sketch = new Sketcher("XZ").hLine(width).threePointsArc(0,2*fillet,fillet,fillet).hLine(-width).close();
  let revolved = sketch.revolve();
  let rectangle = sketchRectangle(length, width, {plane: "XZ"});
  let box = rectangle.extrude(30);
  let wheel = sketchCircle(radius, {plane:"XZ", origin:[30,0,0]}).extrude(2).translateY(-10);
  box = box.fuse(wheel);
  box = box.fuse(revolved);
  return box;
}
```
