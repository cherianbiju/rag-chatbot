---
source_file: bottle.js
category: replicad_example
type: annotated_code
use_case: bottle design, shell modeling, thread creation, organic shapes
related: birdhouse.md, creditCardTray.md
---

# Bottle

## Description
Parametric bottle model with a rounded body created by mirroring a sketched profile, a cylindrical neck, and a helical thread on the neck modeled using FaceSketcher and loft. Demonstrates shelling for hollow bodies, face-based sketching, and makeOffset for thread placement.

## Keywords
bottle, shell, FaceSketcher, loftWith, makeOffset, FaceFinder, thread, neck, cylinder, threePointsArc, closeWithMirror, hollow, parametric, organic shape

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| width | 70 | mm | Width of bottle body |
| height | 70 | mm | Height of bottle body |
| thickness | 30 | mm | Depth/thickness of bottle |
| myNeckRadius | thickness/4 | mm | Radius of bottle neck |
| myNeckHeight | height/10 | mm | Height of neck cylinder |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher() | Creates 2D sketch on default plane |
| .threePointsArc(dx,dy,mx,my) | Draws arc through midpoint |
| .closeWithMirror() | Closes sketch by mirroring the path |
| .extrude(h) | Extrudes sketch to 3D |
| .fillet(r) | Rounds all edges by radius r |
| makeCylinder(r,h,origin,dir) | Creates cylinder at given position and direction |
| .fuse(other) | Boolean union |
| .shell(t, faceFinder) | Hollows out solid keeping wall thickness t |
| FaceFinder() | Tool to find specific faces |
| .containsPoint([x,y,z]) | Finds face containing given point |
| .ofSurfaceType("CYLINDRE") | Finds cylindrical faces |
| makeOffset(face, delta) | Offsets a face inward or outward |
| new FaceSketcher(face) | Creates sketch on an existing face |
| .halfEllipse(l,r1,r2) | Draws half ellipse for thread profile |
| .loftWith(other) | Lofts between two sketches |

## Code
```javascript
const defaultParams = { width:70, height:70, thickness:30 };
const main = ({ Sketcher, FaceSketcher, makeCylinder, makeOffset, FaceFinder },
              { width:myWidth, height:myHeight, thickness:myThickness }) => {
  let shape = new Sketcher()
    .movePointerTo([-myWidth/2,0]).vLine(-myThickness/4)
    .threePointsArc(myWidth,0,myWidth/2,-myThickness/4)
    .vLine(myThickness/4).closeWithMirror()
    .extrude(myHeight).fillet(myThickness/12);
  const myNeckRadius = myThickness/4, myNeckHeight = myHeight/10;
  const neck = makeCylinder(myNeckRadius, myNeckHeight, [0,0,myHeight], [0,0,1]);
  shape = shape.fuse(neck);
  shape = shape.shell(myThickness/50, (f) => f.inPlane("XY",[0,0,myHeight+myNeckHeight]));
  const neckFace = new FaceFinder().containsPoint([0,myNeckRadius,myHeight])
    .ofSurfaceType("CYLINDRE").find(shape.clone(), {unique:true});
  const bottomThreadFace = makeOffset(neckFace,-0.01*myNeckRadius).faces[0];
  const baseThreadSketch = new FaceSketcher(bottomThreadFace)
    .movePointerTo([0.75,0.25]).halfEllipse(2,0.5,0.1).close();
  const topThreadFace = makeOffset(neckFace,0.05*myNeckRadius).faces[0];
  const topThreadSketch = new FaceSketcher(topThreadFace)
    .movePointerTo([0.75,0.25]).halfEllipse(2,0.5,0.05).close();
  const thread = baseThreadSketch.loftWith(topThreadSketch);
  return shape.fuse(thread);
};
```
