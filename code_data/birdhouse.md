---
source_file: birdhouse.js
category: replicad_example
type: annotated_code
use_case: enclosure design, shell modeling, parametric design
related: creditCardTray.md, boolean.md
---

# Birdhouse

## Description
Parametric birdhouse model built by extruding a triangular (toblerone) profile, shelling it to create hollow walls, and cutting a circular entry hole. Includes a decorative hook at the top for hanging. Demonstrates shell operations, fillet with edge finders, and smooth spline curves.

## Keywords
shell, fillet, Sketcher, sketchCircle, extrude, cut, fuse, rotate, smoothSplineTo, closeWithMirror, enclosure, parametric, hook, birdhouse

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| height | 85 | mm | Height of birdhouse body |
| width | 120 | mm | Width of birdhouse |
| thickness | 2 | mm | Wall thickness after shell |
| holeDia | 50 | mm | Diameter of entry hole |
| hookHeight | 10 | mm | Height of hanging hook |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XZ", offset) | Sketch on XZ plane with Y offset |
| .lineTo([x,y]) | Draw line to absolute point |
| .close() | Close sketch path |
| .extrude(depth) | Extrude sketch to 3D |
| .shell(t, faceFinder) | Hollow out solid with wall thickness t |
| .fillet(r, edgeFinder) | Round edges matching finder condition |
| .either([f1,f2]) | EdgeFinder OR condition |
| sketchCircle(r, options) | Create circular sketch at position |
| .cut(tool) | Boolean subtract tool from shape |
| .clone() | Duplicate shape |
| .fuse(other) | Boolean union |
| .rotate(angle) | Rotate shape around Z axis |
| .smoothSplineTo(pt, angle) | Draw smooth spline to point |
| .closeWithMirror() | Close sketch by mirroring |
| .translate([x,y,z]) | Move shape by vector |

## Code
```javascript
const defaultParams = { height:85, width:120, thickness:2, holeDia:50, hookHeight:10 };
function main({ Sketcher, sketchCircle }, { width:inputWidth, height, thickness, holeDia, hookHeight }) {
  const length = inputWidth, width = inputWidth * 0.9;
  const tobleroneShape = new Sketcher("XZ", -length/2)
    .movePointerTo([-width/2,0]).lineTo([0,height]).lineTo([width/2,0]).close()
    .extrude(length)
    .shell(thickness, (f) => f.parallelTo("XZ"))
    .fillet(thickness/2, (e) => e.inDirection("Y").either([(f)=>f.inPlane("XY"),(f)=>f.inPlane("XY",height)]));
  const hole = sketchCircle(holeDia/2, {plane:"YZ", origin:[-length/2,0,height/3]}).extrude(length);
  const base = tobleroneShape.cut(hole);
  const body = base.clone().fuse(base.rotate(90));
  const hookWidth = length/2;
  const hook = new Sketcher("XZ")
    .movePointerTo([0,hookHeight/2]).smoothSplineTo([hookHeight/2,0],-45)
    .lineTo([hookWidth/2,0]).line(-hookWidth/4,hookHeight/2)
    .smoothSplineTo([0,hookHeight],{endTangent:180,endFactor:0.6})
    .closeWithMirror().extrude(thickness)
    .translate([0,thickness/2,height-thickness/2]);
  return body.fuse(hook);
}
```
