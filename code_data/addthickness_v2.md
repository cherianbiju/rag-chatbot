---
source_file: addthickness_v2.js
category: replicad_example
type: annotated_code
use_case: geometry experimentation, donut/torus shape, edge highlighting
related: addthickness.md, cannedSketches.md
---

# Add Thickness V2

## Description
Extended version of addthickness experiments, adding a torus (donut) shape created by revolving a circle around a distant axis. Also demonstrates returning an EdgeFinder highlight alongside the shape for visual debugging of edges in a specific direction.

## Keywords
revolve, torus, donut, EdgeFinder, extrude, fuse, Sketcher, sketchRectangle, sketchCircle, highlight edges, inDirection

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| length | 120 | mm | Length of extruded rectangle |
| width | 20 | mm | Width of rectangle |
| height | 10 | mm | Height of extruded box |
| radius | 20 | mm | Radius of circle for torus |
| fillet | 4 | mm | Arc fillet in revolved profile |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XZ") | Creates 2D sketch on XZ plane |
| .revolve([0,0,1], {origin}) | Revolves sketch around Z axis at origin to make torus |
| sketchRectangle() | Creates rectangular sketch |
| .extrude() | Extrudes sketch to 3D solid |
| sketchCircle() | Creates circular sketch |
| .fuse() | Boolean union of shapes |
| EdgeFinder | Utility to find and highlight specific edges |
| .inDirection("X") | Filters edges running in X direction |

## Code
```javascript
function main({ Sketcher, sketchRectangle, sketchCircle, EdgeFinder }) {
  let length = 120, width = 20, height = 10, radius = 20, fillet = 4;
  let sketch = new Sketcher("XZ").hLine(width).threePointsArc(0,2*fillet,fillet,fillet).hLine(-width).close();
  let revolved = sketch.revolve();
  let rectangle = sketchRectangle(length, width/2, {plane: "XZ"});
  let box = rectangle.extrude(height);
  let donut = sketchCircle(radius/2, {plane:"XZ", origin:[length/2,0,0]}).revolve([0,0,1], {origin:[0,0,0]});
  box = box.fuse(donut);
  return {shape: box, highlight: new EdgeFinder().inDirection("X")};
}
```
