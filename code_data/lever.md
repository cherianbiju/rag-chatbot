---
source_file: lever.js
category: mechanical
type: annotated_code
use_case: parametric lever arm connecting two pivot circles with tangent lines and optional drilled holes
related: keyway.md, knob8.md
---
# Lever — Parametric Two-Circle Lever Arm

## Description
Provides two reusable functions for constructing a mechanical lever: `Lever()` builds the basic solid body connecting two circles of different radii with straight tangent lines and arc ends, while `leverHoles()` adds coaxial through-holes to both pivot points with a configurable wall thickness. Geometry is computed analytically from the angle between the circles.

## Keywords
lever, mechanical-arm, pivot, tangent-lines, threePointsArcTo, Sketcher, sketchCircle, extrude, cut, Boolean, parametric, lever-arm, replicad, 3d-printing, two-circle, analytic-geometry

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| r1 | 30 | mm | Radius of the large (origin) pivot circle |
| r2 | 12 | mm | Radius of the small (end) pivot circle |
| d | 90 | mm | Distance between the two circle centres |
| t | 5 | mm | Wall thickness around holes (hole radius = pivot radius − t) |
| h | 20 | mm | Lever extrusion height (thickness in Z) |
| hole overcut | +10 | mm | Extra length added to hole cylinders to ensure clean Boolean cut |

## Code
```javascript
function main({Sketcher, sketchCircle,Lever,leverHoles},{})
{

let r1  = 30;
let r2  = 12;
let d   = 90;
let t   = 5;
let h   = 20;

function Lever(radius1, radius2, distance, leverHeight)
{
    let sinus_angle = (radius1 - radius2) / distance
    let angle = Math.asin(sinus_angle);

    let p1 = [radius1 * Math.sin(angle), radius1 * Math.cos(angle)];
    let p2 = [distance + radius2 * Math.sin(angle), radius2 * Math.cos(angle)];
    let p3 = [distance + radius2, 0];
    let p4 = [distance + radius2 * Math.sin(angle), - radius2 * Math.cos(angle)];
    let p5 = [radius1 * Math.sin(angle), - radius1 * Math.cos(angle)];
    let p6 = [- radius1, 0 ];

    let sketchLever = new Sketcher("XY").movePointerTo(p1)
                    .lineTo(p2)
                    .threePointsArcTo(p4,p3)
                    .lineTo(p5)
                    .threePointsArcTo(p1,p6)
                    .close();
              
    let leverBody = sketchLever.extrude(leverHeight);
    return leverBody
}

function leverHoles(radius1,radius2,distance,leverHeight,wallThickness)
{ 
    let leverBody = Lever(radius1,radius2,distance,leverHeight);
    let orig_hole  = sketchCircle(radius1-wallThickness).extrude(leverHeight + 10);
    let dist_hole =  sketchCircle(radius2-wallThickness).extrude(leverHeight + 10).translate([distance,0,0]);
    let lever   = leverBody.cut(orig_hole)
    lever       = lever.cut(dist_hole);
    return lever
}

let shape = leverHoles(r1,r2,d,h,t);
let shapeArray =[{shape: shape, color: "steelblue"}]
return shapeArray;
}
```
