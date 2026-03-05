---
source_file: holder_flatbase.js
category: enclosure
type: annotated_code
use_case: GPS holder flat bottom, 3D printing no-support, lanyard holes
related: holder.md, holder_ghost_param.md, holder_ghost_param3.md
---

# Holder Flat Base (GPS Case, Flat Bottom)

## Description
GPS receiver holder variant with a flat bottom base — only the top Y-direction edges are filleted, leaving a flat base ideal for 3D printing without supports. Uses rlandist=7mm to produce two separate offset lanyard holes instead of one central hole. Otherwise follows the same structure as holder_ghost_param.js. Final edges get fillet(0.6) instead of chamfer(0.4).

## Keywords
GPS holder, flat base, flat bottom, makeBaseBox, makeCylinder, fillet, inDirection, inPlane, cut, containsPoint, two lanyard holes, rlandist, 3D printing, no support, fillet 0.6

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| lx | 45.75 | mm | GPS width + 0.5mm tolerance |
| ly | 79.75 | mm | GPS length + 0.5mm tolerance |
| lz | 11.9 | mm | GPS thickness + 0.5mm tolerance |
| th | 2 | mm | Wall thickness |
| wholder | 20 | mm | Lanyard lip width |
| yholder | 10 | mm | Lanyard lip protrusion |
| rlanhol | 2 | mm | Lanyard hole radius |
| rlandist | 7 | mm | Distance between two lanyard holes (non-zero = two holes) |
| ycut | 0.6 | - | Fraction of side length to cut open |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| r.makeBaseBox(l,w,h) | Creates box shapes |
| .fillet(r, e=>e.inDirection("Y").inPlane("XY",z)) | Fillets ONLY top Y-edges — gives flat base |
| .fillet(r, e=>e.inDirection("Y")) | Fillets receiver/hollow Y-edges (full) |
| .fillet(r, e=>e.inDirection("X")) | Fillets cutter X-edges |
| .translate([x,y,z]) | Positions shapes |
| .fuse(other) | Adds lanyard lip |
| .cut(other) | Cuts hollow, side cutter, top cutter, lanyard holes |
| .containsPoint([x,y,z]) | Selects cutout loop edge for fillet |
| .fillet(8, e=>e.inDirection("Z").inPlane("XZ",x)) | Rounds lanyard lip Z-edges |
| r.makeCylinder(r,h,[origin],[axis]) | Creates offset lanyard hole cylinders |
| .fillet(0.6) | Final global small edge fillet |

## Code
```javascript
const r = replicad;
const main = () => {
  let lx=45.25,ly=79.25,lz=11.4,lt=0.5,th=2,wholder=20,yholder=10;
  let rlanhol=2,ycut=0.6,rlandist=7;
  lx+=lt; ly+=lt; lz+=lt;
  let receiver = r.makeBaseBox(lx,ly,lz).fillet((lz-lt)/2,(e)=>e.inDirection("Y")).translate([0,0,th]);
  let hollow = r.makeBaseBox(lx,ly+2*th,lz).fillet((lz-lt)/2,(e)=>e.inDirection("Y")).translate([0,th,th]);
  // FLAT BASE: only fillet top Y-edges
  let shape = r.makeBaseBox(lx+2*th,ly+2*th,lz+2*th)
    .fillet((lz+2*th-lt)/2,(e)=>e.inDirection("Y").inPlane("XY",lz+2*th));
  shape = shape.fuse(r.makeBaseBox(wholder,ly+2*th,th).translate([0,yholder,0]));
  let cutter = r.makeBaseBox(lx*1.2,ly*ycut,lz*2).translate([0,0,2*th]).fillet(5,(e)=>e.inDirection("X"));
  let cutterTop = r.makeBaseBox(lx*1.2,ly*1.2,lz).translate([0,0,(lz+2*th)*0.87]);
  let shapeRounded = shape.cut(hollow).cut(cutter)
    .fillet(1.0,(e)=>e.containsPoint([0,ly*ycut/2,lz+2*th]))
    .fillet(8,(e)=>e.inDirection("Z").inPlane("XZ",-(((ly+2*th)/2)+yholder)))
    .cut(r.makeCylinder(rlanhol,th*4,[rlandist/2,ly/2+yholder/2+th,-2*th],[0,0,1]))
    .cut(r.makeCylinder(rlanhol,th*4,[-rlandist/2,ly/2+yholder/2+th,-2*th],[0,0,1]))
    .cut(cutterTop).fillet(0.6);
  return [{shape:receiver,name:"receiver",color:"dimgrey",opacity:0.8},
          {shape:shapeRounded,name:"holder",color:"steelblue",opacity:1.0}];
};
```
