---
source_file: edgefinder.js
category: replicad_example
type: annotated_code
use_case: selective fillet, edge selection reference, EdgeFinder API
related: edges-inlist.md, finder_combination.md
---

# EdgeFinder

## Description
Demonstrates the EdgeFinder API for selecting specific edges to apply different fillet radii on the same box. Top X-direction edges get a 5mm fillet, and front face (YZ plane at X=0) non-horizontal edges get a 2mm fillet. Essential reference for selective edge operations in replicad.

## Keywords
EdgeFinder, fillet, inDirection, inPlane, not, edge selection, selective fillet, draw, hLine, vLine, sketchOnPlane, extrude, reference, multi-radius fillet

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| box width | 75 | mm | Rectangle width (X) |
| box height | 40 | mm | Rectangle height (Y) |
| box depth | 20 | mm | Extrude height (Z) |
| top fillet | 5 | mm | Fillet on top X-direction edges |
| front fillet | 2 | mm | Fillet on front YZ face non-horizontal edges |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw() | Starts 2D freeform drawing |
| .hLine(d) | Horizontal line segment |
| .vLine(d) | Vertical line segment |
| .close() | Closes 2D path |
| .sketchOnPlane("XY") | Places sketch on XY plane |
| .extrude(h) | Extrudes to 3D solid |
| new EdgeFinder() | Creates an edge selector |
| .inDirection("X") | Selects edges running in X direction |
| .inPlane("XY", z) | Selects edges lying in XY plane at height z |
| .inPlane("YZ", x) | Selects edges in YZ plane at X position |
| .not(fn) | Inverts the edge filter condition |
| .fillet(r, fn) | Applies fillet to edges matching the finder |

## Code
```javascript
const {draw, EdgeFinder} = replicad;
function main() {
  let rectangleSketch = draw().hLine(75).vLine(40).hLine(-75).close().sketchOnPlane("XY");
  let box = rectangleSketch.extrude(20);
  let selectedEdges = new EdgeFinder().inDirection("X").inPlane("XY",20);
  let frontEdges = new EdgeFinder().inPlane("YZ",0).not((e)=>e.inPlane("XY"));
  box = box.fillet(5, test => selectedEdges);
  box = box.fillet(2, test => frontEdges);
  return box;
}
```
