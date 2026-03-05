---
source_file: holder_ghost.js
category: enclosure
type: annotated_code
use_case: generic device holder prototype, lanyard holes, 3D printing
related: holder.md, holder_ghost_param.md, holder_flatbase.md
---

# Holder Ghost (Generic Device Holder Prototype)

## Description
Early prototype of the GPS holder with hardcoded generic dimensions (50×100×20mm). The holder is a rounded box with a lanyard lip fused on. A hollow cutout, side cutout, and top cutout are removed. Edge of the side cutout is filleted using containsPoint. Lanyard lip Z-edges are filleted. Two offset lanyard holes are drilled. Returns both the pre-fillet and post-fillet shapes in a named array for comparison.

## Keywords
device holder, prototype, makeBaseBox, makeCylinder, fillet, inDirection, inPlane, inBox, containsPoint, cut, fuse, lanyard hole, named shapes, comparison, shapeUnrounded, shapeRounded, ghost

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| holder width | 50 | mm | Outer holder X dimension |
| holder length | 100 | mm | Outer holder Y dimension |
| holder height | 20 | mm | Outer holder Z dimension |
| holder fillet | 9.5 | mm | Y-edge fillet radius (near full thickness) |
| lanyardholder width | 20 | mm | Lanyard lip X size |
| lanyardholder translate | [0,10,0] | mm | Lanyard lip position |
| hollow width | 46 | mm | Inner cavity X (holder − 4mm) |
| hollow height | 16 | mm | Inner cavity Z (holder − 4mm) |
| hollow fillet | 7.5 | mm | Hollow Y-edge fillet |
| hollow translate | [0,2,2] | mm | Hollow offset from corner |
| cutter Z | 4 | mm | Side cutout starts at Z=4 |
| cutterTop Z | 17 | mm | Top cutout starts at Z=17 (87% of height) |
| lanyard hole r | 2 | mm | Lanyard hole radius |
| lanyard hole centers | ±3.5 | mm | Two holes at Y=53 |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| r.makeBaseBox(l,w,h) | Creates holder, hollow, cutter, cutterTop boxes |
| .fillet(r, e=>e.inDirection("Y")) | Rounds Y-direction edges for pill shapes |
| .translate(x,y,z) | Positions shapes |
| .fuse(other) | Fuses lanyard lip onto holder |
| .cut(other) | Cuts hollow, side, top, lanyard holes |
| .fillet(r, e=>e.inDirection("X")) | Rounds cutter X-edges for smooth opening |
| .containsPoint([x,y,z]) | Selects the side cutout loop edge for fillet |
| .fillet(8, e=>e.inDirection("Z").inPlane("XZ",x)) | Rounds lanyard lip Z-edges at specific X |
| r.makeCylinder(r,h,[origin],[axis]) | Creates lanyard hole cylinders along Z |
| .fillet(0.6) | Final global small edge fillet |

## Code
```javascript
const r = replicad;
const {drawRoundedRectangle, EdgeFinder} = replicad;
const main = () => {
  let shape = r.makeBaseBox(50,100,20).fillet(9.5,(e)=>e.inDirection("Y"));
  let lanyardholder = r.makeBaseBox(20,100,2).translate([0,10,0]);
  shape = shape.fuse(lanyardholder);
  let hollow = r.makeBaseBox(46,100,16).fillet(7.5,(e)=>e.inDirection("Y")).translate(0,2,2);
  let cutter = r.makeBaseBox(60,60,30).translate([0,0,4]).fillet(5,(e)=>e.inDirection("X"));
  let cutterTop = r.makeBaseBox(60,120,20).translate([0,0,17]);
  let shape1 = shape.cut(hollow);
  let shapeUnrounded = shape1.cut(cutter);
  let shapeRounded = shapeUnrounded.fillet(1.0,(e)=>e.containsPoint([-15.50,30.00,20.00]));
  shapeRounded = shapeRounded.fillet(8,(e)=>e.inDirection("Z").inPlane("XZ",-60));
  let lanyardCutterL = r.makeCylinder(2,20,[3.5,53,-5],[0,0,1]);
  let lanyardCutterR = r.makeCylinder(2,20,[-3.5,53,-5],[0,0,1]);
  shapeRounded = shapeRounded.cut(lanyardCutterL).cut(lanyardCutterR);
  shapeRounded = shapeRounded.cut(cutterTop).fillet(0.6);
  return [{shape:shapeUnrounded,color:"grey",name:"shapeUnrounded"},
          {shape:shapeRounded,color:"slategrey",name:"shapeRounded"}];
};
```
