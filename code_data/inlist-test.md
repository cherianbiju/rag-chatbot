---
source_file: inlist-test.js
category: edge-selection
type: annotated_code
use_case: demonstrating edge filtering with inList, inDirection, and inPlane selectors for selective filleting
related: holderv7.md, holder3.md
---
# Edge Selection with inList

## Description
Demonstrates replicad's edge selection API by creating a simple rectangular box and selectively applying fillets using `inList`, `inDirection`, and `inPlane` edge finders. Shows how to capture all edges of a shape into a list and use that list to target specific fillet operations.

## Keywords
EdgeFinder, inList, inDirection, inPlane, fillet, edge-selection, draw, hLine, vLine, extrude, sketchOnPlane, replicad, selective-fillet, box, parametric, edges

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| box width | 75 | mm | Width of the rectangle (X direction) |
| box depth | 40 | mm | Depth of the rectangle (Y direction) |
| box height | 20 | mm | Extrusion height (Z direction) |
| fillet radius | 5 | mm | Fillet applied to edges from the inList selection |

## Code
```javascript
const {draw, EdgeFinder} = replicad

function main()
{
let rectangleSketch = draw().hLine(75).vLine(40).hLine(-75).close()
rectangleSketch = rectangleSketch.sketchOnPlane("XY")
let box = rectangleSketch.extrude(20)

let selectedEdges = new EdgeFinder().inDirection("X").inPlane("XY",20)
let frontEdges = new EdgeFinder().inPlane("YZ",0).not((e)=>e.inPlane("XY"))

let list= box.edges
console.log(list)

box = box.fillet(5,e=>e.inList(list))

return box
}
```
