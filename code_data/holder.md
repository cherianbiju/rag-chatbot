---
source_file: holder.js
category: enclosure
type: annotated_code
use_case: GPS receiver holder, parametric case, lanyard holes, 3D printing
related: holder_ghost_param.md, holder_flatbase.md, hexagon_holder_v3.md
---

# Holder (Parametric GPS Receiver Case)

## Description
Parametric GPS receiver holder with configurable device dimensions, fit tolerance, wall thickness, and height portion to cover. Creates the receiver body shape, builds an outer holder by adding thickness, then cuts four regions: top opening, side cutout, oval bottom cutout, and paired lanyard holes. Returns all shapes in a named array for inspection. Clean parametric design template for custom device holders.

## Keywords
GPS holder, makeBaseBox, makeCylinder, makeCompound, fillet, cut, inDirection, inBox, tolerance, lanyard hole, portion, parametric, 3D printing, device holder, named shapes, defaultParams

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| gnsLength | 79.25 | mm | GPS device length |
| gnsWidth | 45.25 | mm | GPS device width |
| gnsHeight | 11.4 | mm | GPS device height |
| fit | 0.5 | mm | Tolerance clearance around device |
| thickness | 2.0 | mm | Holder wall thickness |
| portion | 0.85 | - | Fraction of height that holder covers |
| radius | 5.7 | mm | Pill radius = gnsHeight/2 |
| cutterTop offset | 0.85×height | mm | Top opening starts at portion×height |
| cutterSide | portion×length | mm | Side opening width |
| cutterBottom | 80% width | mm | Bottom oval width |
| lanyard hole r | 2 | mm | Lanyard hole radius |
| lanyard hole offset | ±3.5 | mm | Two holes 7mm apart |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| r.makeBaseBox(l,w,h) | Creates device and holder boxes |
| .fillet(r, e=>e.inDirection("X")) | Rounds X-edges for pill shape |
| .translate(x,y,z) | Positions shapes |
| .cut(other) | Boolean subtract for all cavities and holes |
| r.makeCylinder(r,h,origin,axis) | Creates lanyard hole cylinders at [x,y,z] along Z |
| r.makeCompound([s1,s2]) | Groups two lanyard cylinders into one cutter |
| .fillet(r, e=>e.inBox([p1],[p2]).inDirection("Y")) | Fillets Y-edges only in bounding box zone |
| .fillet(0.5) | Global small fillet on all remaining edges |

## Code
```javascript
const defaultParams = {gnsLength:79.25,gnsWidth:45.25,gnsHeight:11.4,fit:0.5,thickness:2.0,portion:0.85};
const r = replicad;
function main({},{gnsLength,gnsWidth,gnsHeight,fit,thickness,portion}) {
  let length=gnsLength+fit, width=gnsWidth+fit, height=gnsHeight+fit, radius=gnsHeight/2;
  let receiverBody = r.makeBaseBox(length,width,height).fillet(radius,(e)=>e.inDirection("X"));
  let holder = r.makeBaseBox(length+2*thickness,width+2*thickness,height+2*thickness)
    .fillet(radius+thickness,(e)=>e.inDirection("X")).translate(0,0,-thickness);
  let cutterTop = r.makeBaseBox(length+4*thickness,width+4*thickness,height).translate(0,0,portion*height);
  let cutterSide = r.makeBaseBox(length*portion,width+4*thickness,height).translate(0,0,3);
  let cutterBottom = r.makeBaseBox(length,width*0.8,height).fillet(3,(e)=>e.inDirection("X")).translate(length/2,0,2.0);
  let cutterLanyardL = r.makeCylinder(2,20,[-length/2-10,3.5,5],[1,0,0]);
  let cutterLanyardR = r.makeCylinder(2,20,[-length/2-10,-3.5,5],[1,0,0]);
  let cutterLanyard = r.makeCompound([cutterLanyardL,cutterLanyardR]);
  holder = holder.cut(receiverBody).cut(cutterTop).cut(cutterSide).cut(cutterBottom).cut(cutterLanyard)
    .fillet(2.5,(e)=>e.inBox([length/2-5,50,3],[-length/2+5,-50,3+height]).inDirection("Y")).fillet(0.5);
  return [{shape:receiverBody,name:"receiver",color:"red"},
          {shape:cutterTop,name:"cutterTop",color:"green",opacity:0.5},
          {shape:cutterSide,name:"cutterSide",color:"green",opacity:0.5},
          {shape:cutterBottom,name:"cutterBottom",color:"green",opacity:0.5},
          {shape:cutterLanyard,name:"cutterLanyard",color:"green",opacity:0.5},
          {shape:holder,name:"holder",opacity:1.0}];
}
```
