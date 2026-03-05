---
source_file: doorhandle.js
category: enclosure
type: annotated_code
use_case: screen door handle, hardware, countersunk screw holes, face highlight
related: creditCardTray.md, birdhouse.md
---

# Door Handle

## Description
Screen door handle with a trapezoidal extruded body, finger cutout relief, lock slot, small tab space, and two countersunk screw holes created via loftWith between circles at different heights. Returns the shape with a FaceFinder highlight on the top face for inspection.

## Keywords
door handle, screen door, loftWith, countersink, screw hole, Sketcher, sketchRectangle, sketchCircle, Plane, FaceFinder, inPlane, inDirection, containsPoint, inBox, fillet, cut, rotate, translate, highlight

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| handle length | 89.0 | mm | Total handle length (Y direction) |
| handle width | 57.0 | mm | Total handle width (X direction) |
| extrude depth | 9.0 | mm | Handle body thickness (Z) |
| border | 3.0 | mm | Border margin around finger cutout |
| fingerArea extrude | 30.0 | mm | Depth of finger cutout negative |
| lockNegative | 25×7 | mm | Lock slot rectangle dimensions |
| lockSmallTab | 15×5 | mm | Small tab space dimensions |
| screwHole top r | 4.0 | mm | Wide top radius of countersunk hole |
| screwHole bottom r | 1.5 | mm | Narrow bottom radius |
| screwHole depth | 9.0 | mm | Total depth of screw hole |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XY") | Creates sketch on XY plane |
| .vLine() .hLine() .line() | Draws handle profile lines |
| .close() | Closes sketch |
| .extrude(h) | Extrudes profile to 3D |
| .fillet(r, fn) | Rounds specific edges |
| .inDirection("Z") | Selects vertical edges |
| .containsPoint([x,y,z]) | Selects edge passing through a point |
| .inBox([p1],[p2]) | Selects edges within a bounding box |
| sketchRectangle(w,h) | Creates rectangular sketch for lock negatives |
| .rotate(angle, origin, axis) | Rotates shape to correct orientation |
| .translate([x,y,z]) | Positions shapes |
| sketchCircle(r, Plane) | Creates circle at a specific plane height |
| .loftWith([s1,s2]) | Lofts between circles to create countersink |
| .cut(other) | Boolean subtract |
| .clone() | Duplicates shape |
| FaceFinder().inPlane("XY",z) | Finds face at given Z height for highlight |

## Code
```javascript
const main = ({ Sketcher, sketchRectangle, sketchCircle, Plane, FaceFinder }, {}) => {
  const handleBase = new Sketcher("XY")
    .vLine(89.0).hLine(20.5).line(57.0-20.5,-3.5).vLine(-82.0).line(-57.0+20.5,-3.5).hLine(-20.5).close()
    .extrude(9.0)
    .fillet(5.0, e => e.inDirection('Z').containsPoint([57,3.5,0]))
    .fillet(5.0, e => e.inDirection('Z').containsPoint([57,89.0-3.5,0]))
    .fillet(1.0, e => e.inBox([0,0,9],[20.5,89.0,0]));
  const border = 3.0;
  const fingerAreaNegative = new Sketcher("XY")
    .line(57.0-20.5-border,-3.5).vLine(-82.0+(border*2)).line(-57.0+20.5+border,-3.5).close()
    .extrude(30.0).fillet(5.0, e => e.inDirection('Z')).fillet(1.5);
  const lockNegative = sketchRectangle(25.0,7.0).extrude(20.0)
    .rotate(90,undefined,[1,0,0]).rotate(90,undefined,[0,0,1]).translate([0,0,3.5]);
  const lockSmallTabSpaceNegative = sketchRectangle(15.0,5.0).extrude(3.0)
    .rotate(90,undefined,[1,0,0]).rotate(90,undefined,[0,0,1]).translate([0,0,2.5]);
  const screwHole = sketchCircle(4.0, new Plane([0,0,0]))
    .loftWith([sketchCircle(1.5,new Plane([0,0,-3.0])), sketchCircle(1.5,new Plane([0,0,-9.0]))])
    .translate([0.0,0.0,9.0]);
  let handle = handleBase
    .cut(fingerAreaNegative.translate([20.5,89-border,2.0]))
    .cut(lockNegative.translate([2.0,89.0/2,0]))
    .cut(lockSmallTabSpaceNegative.translate([57-3,89.0/2,0]))
    .cut(screwHole.clone().translate([10.0,6.0,0.0]))
    .cut(screwHole.translate([10.0,89.0-6.0,0.0]));
  return {shape: handle, highlight: new FaceFinder().inPlane("XY",9)};
};
```
