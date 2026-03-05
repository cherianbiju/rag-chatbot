---
source_file: loft-pipe.js
category: geometry
type: annotated_code
use_case: creating a pipe or tube by lofting circular cross-sections through 3D space along an implied curved path
related: loft-examples.md, loft-ruled.md, loft-ruled_v2.md, loft-ruled_v3.md, loft-rules.md
---
# Loft Pipe — Circular Cross-Section Pipe Through 3D Space

## Description
Creates a smooth pipe by lofting five circular cross-sections placed at different positions and orientations in 3D space. The circles are sketched on different planes and at different origins, so the loft naturally follows a curved path through the waypoints — useful for hoses, pipes, conduits, or organic tube forms.

## Keywords
loft, loftWith, pipe, tube, sketchCircle, cross-section, 3D-path, plane, origin, XZ-plane, XY-plane, replicad, swept-surface, curved-tube, organic-form, 3d-printing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| radius | 5 | mm | Radius of all circular cross-sections |
| xSection1 | plane XZ, origin [0,0,0] | — | Start circle, vertical plane |
| xSection2 | plane XY, origin [0,50,30] | — | First bend waypoint |
| xSection3 | plane XY, origin [0,50,50] | — | Intermediate waypoint |
| xSection4 | plane XY, origin [0,50,70] | — | Second intermediate waypoint |
| xSection5 | plane XZ, origin [0,100,100] | — | End circle, vertical plane |

## Code
```javascript
function main( 
{
    Sketcher,
    sketchCircle,
    sketchRoundedRectangle,
    supportExtrude
})
{

let radius = 5
let xSection1 = sketchCircle(radius,{plane:"XZ",origin: [0,0,0]})
let xSection2 = sketchCircle(radius,{plane:"XY",origin: [0,50,30]})
let xSection3 = sketchCircle(radius,{plane:"XY",origin: [0,50,50]})
let xSection4 = sketchCircle(radius,{plane:"XY",origin: [0,50,70]})
let xSection5 = sketchCircle(radius,{plane:"XZ",origin: [0,100,100]})
let pipe = xSection1.loftWith([xSection2,xSection3,xSection4,xSection5])

return pipe
}
```
