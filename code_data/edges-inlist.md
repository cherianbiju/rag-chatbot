---
source_file: edges-inlist.js
category: replicad_example
type: annotated_code
use_case: edge list capture, fillet all edges, inList reference
related: edgefinder.md, finder_combination.md
---

# Edges In List

## Description
Demonstrates the inList() EdgeFinder method which applies fillets to a pre-captured list of edges. All edges of the box are captured into a variable before any operation, then fillet uses inList to match exactly those edges. Useful when you want to apply operations to a frozen set of edges captured at a specific point in the modeling history.

## Keywords
inList, EdgeFinder, edges, fillet, drawRectangle, sketchOnPlane, extrude, edge list, edge capture, selective fillet, reference, pre-capture

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| box width | 40 | mm | Rectangle width |
| box length | 80 | mm | Rectangle length |
| box height | 20 | mm | Extrude height |
| fillet radius | 3 | mm | Fillet applied to all captured edges |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawRectangle(w, h) | Creates rectangular 2D sketch |
| .sketchOnPlane("XY") | Places sketch on XY plane |
| .extrude(h) | Extrudes to 3D solid |
| .edges | Property returning all edges of the shape |
| .inList(edgeArray) | EdgeFinder matching only edges from a pre-captured list |
| .fillet(r, fn) | Rounds matched edges |

## Code
```javascript
const main = ({ draw, drawRectangle, Plane, makeOffset, makeSolid }, {}) => {
  let baseBox = drawRectangle(40,80).sketchOnPlane("XY");
  baseBox = baseBox.extrude(20);
  let box = baseBox;
  let edgesAll = baseBox.edges;
  box = baseBox.fillet(3, e => e.inList(edgesAll));
  return box;
};
```
