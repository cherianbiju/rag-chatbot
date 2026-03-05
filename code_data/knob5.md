---
source_file: knob5.js
category: mechanical
type: annotated_code
use_case: rotary control knob with dome top, ribbed internal stem, and hollow shell — early iteration
related: knob6.md, knob7.md, knob8.md, knob10.md, knob11_pretty.md
---
# Knob v5 — Rotary Knob (Early Iteration)

## Description
Early version of a rotary control knob modeled after the SolidWorks Model Mania 2006 challenge. Builds the knob finger by extruding a side-view profile drawn with elliptic arcs and smooth splines, fuses it with a hemispherical dome, shells the result hollow, then adds three radial ribs and a hollow central stem on the interior.

## Keywords
knob, rotary, dome, shell, rib, stem, makeSphere, makeCylinder, ellipse, smoothSplineTo, extrude, fuse, cut, intersect, fillet, inDirection, containsPoint, replicad, 3d-printing, SolidWorks-Model-Mania

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| sphere radius | 30 | mm | Radius of the dome hemisphere |
| extrude width | 32 | mm | Width of the extruded finger |
| finger translate | [-16,0,0] | mm | Centers the extruded finger |
| cutBall box | 80×80×40 | mm | Box used to cut hemisphere from full sphere |
| rib size | 4×40×40 | mm | Dimensions of each internal rib |
| rib rotations | 0°, 120°, 240° | ° | Three equally-spaced ribs around Z axis |
| stem outer radius | 10 | mm | Outer radius of central stem cylinder |
| stem inner radius | 6 | mm | Inner (hollow) radius of stem |
| stem height | 30 | mm | Height of stem cylinder |
| shell thickness | -4 | mm | Wall thickness for hollowing (negative = inward) |
| fillet (dome join) | 5 | mm | Fillet at dome-finger junction edges |
| fillet (Z ribs) | 1 | mm | Fillet on Z-direction rib edges |

## Code
```javascript
const main = (
{draw,
makeSphere,
assembleWire,
EdgeFinder,
makeBaseBox,
makeOffset,
makeCylinder,
Sketcher},
{},    
) => {

let sideView = draw()
.movePointerTo([-60,0])
.ellipse(20*Math.sin(Math.PI/6),20*Math.cos(Math.PI/6),20,20,0,0,false)
.smoothSplineTo([0,32])
.ellipse(32,-32,32,32,Math.PI/2,0,false)
.close()

sideView = sideView.sketchOnPlane("YZ").extrude(32).translate([-16,0,0])
sideView = sideView.fillet(5,(e)=>e.containsPoint([16,0,32]))
sideView = sideView.fillet(5,(e)=>e.containsPoint([-16,0,32]))

let ball = makeSphere(30);
let cutBall = makeBaseBox(80,80,40).translate(0,0,-40)
ball = ball.cut(cutBall);
sideView  = sideView.fuse(ball)
sideView = sideView.fillet(5,(e)=>e.inBox([18,40,1],[-18,-40,50]))
let intersectBall = sideView;
sideView = sideView.shell(-4,(f)=>f.containsPoint([0,0,0]))

let rib =  makeBaseBox(4,40,40).translate(0,20,4);
let rib1 = makeBaseBox(4,40,40).translate(0,20,4).rotate(120,[0,0,0],[0,0,1]);
let rib2 = makeBaseBox(4,40,40).translate(0,20,4).rotate(240,[0,0,0],[0,0,1]);
let stem = makeCylinder(10,30,[0,0,0],[0,0,1]).translate(0,0,4);

rib = rib.fuse(rib1).fuse(rib2)
stem = rib.fuse(stem)
let stemHole = makeCylinder(6,40,[0,0,0],[0,0,1]);
stem = stem.cut(stemHole);
stem = stem.intersect(intersectBall)

sideView  = sideView.fuse(stem)
sideView  = sideView.fillet(1,(e)=>e.inDirection("Z"));

let shapeArray = [
{shape: sideView, name: "sideView"},
]; 

return shapeArray
}
```
