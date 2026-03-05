---
source_file: hexagon_holder_v3.js
category: enclosure
type: annotated_code
use_case: GPS receiver holder, honeycomb pattern cutout, parametric grid, 3D printing
related: holder.md, holder_ghost_param.md, holder_flatbase.md
---

# Hexagon Holder V3 (Honeycomb GPS Case)

## Description
GPS receiver holder with a symmetric honeycomb cutout pattern cut through the side. Creates a flat-base holder (only top edges filleted), cuts the receiver cavity, side and top openings, and two lanyard holes. Then punches a parametric honeycomb grid of hexagonal columns using nested loops — each hexagonal column is intersected with a bounding box to clip it to the grid area. Uses a custom Hexagon() function built with Sketcher lineTo in a loop.

## Keywords
GPS holder, honeycomb, hexagon, Hexagon function, hexColumn, makeBaseBox, makeCylinder, Sketcher, lineTo, loop, intersect, chamfer, containsPoint, inDirection, inPlane, inBox, lanyard, flat base, 3D printing, parametric pattern

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
| rlandist | 0 | mm | Distance between two lanyard holes (0=single) |
| ycut | 0.6 | - | Fraction of side length to cut open |
| hc_width | 35 | mm | Honeycomb grid X extent |
| hc_length | 65 | mm | Honeycomb grid Y extent |
| hc_depth | 10 | mm | Honeycomb column extrude height |
| cellSize | 5 | mm | Hexagon circumradius |
| wallThickness | 1 | mm | Thickness of honeycomb walls |
| rowNumber | 5 | - | Honeycomb rows from center |
| colNumber | 2 | - | Honeycomb columns from center |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| r.makeBaseBox(l,w,h) | Creates receiver, hollow, holder, cutter boxes |
| .fillet(r, e=>e.inDirection("Y").inPlane("XY",z)) | Fillets only top Y-edges (flat base) |
| .fillet(r, e=>e.inDirection("Y")) | Fillets receiver/hollow Y-edges |
| .fillet(r, e=>e.inDirection("X")) | Fillets cutter X-edges |
| .translate([x,y,z]) | Positions shapes |
| .fuse(other) | Adds lanyard lip |
| .cut(other) | Cuts hollow, cutter, top, lanyard holes, hexagons |
| .containsPoint([x,y,z]) | Selects cutout edge loop for fillet |
| .inDirection("Z").inPlane("XZ",x) | Selects lanyard lip Z-edge |
| .fillet(r) .chamfer(r) | Rounds and chamfers final edges |
| new r.Sketcher("XY",-1) | Creates hexagon sketch just below surface |
| .movePointerTo([x,y]) | Moves to first hex vertex |
| .lineTo([x,y]) | Draws hex edges in loop |
| .close() | Closes hexagon sketch |
| .extrude(h) | Extrudes hexagon to column |
| .translate([x,y,z]) | Positions hex column at grid point |
| .intersect(r.makeBaseBox(...)) | Clips hex column to rectangular grid boundary |
| r.makeCylinder(r,h,origin,axis) | Creates lanyard hole cylinders |

## Code
```javascript
const r = replicad;
const main = () => {
  let lx=45.25,ly=79.25,lz=11.4,lt=0.5,th=2,wholder=20,yholder=10;
  let rlanhol=2,ycut=0.6,rlandist=0;
  lx+=lt; ly+=lt; lz+=lt;
  function Hexagon(size){
    let sketchHexagon;
    for(let i=0;i<=5;i++){
      const angle=i*2*Math.PI/6;
      const point=[size*Math.cos(angle),size*Math.sin(angle)];
      if(i===0) sketchHexagon=new r.Sketcher("XY",-1).movePointerTo(point);
      else sketchHexagon.lineTo(point);
    }
    return sketchHexagon.close();
  }
  function hexColumn(size,height){ return Hexagon(size).extrude(height); }
  let receiver = r.makeBaseBox(lx,ly,lz).fillet((lz-lt)/2,(e)=>e.inDirection("Y")).translate([0,0,th]);
  let hollow = r.makeBaseBox(lx,ly+2*th,lz).fillet((lz-lt)/2,(e)=>e.inDirection("Y")).translate([0,th,th]);
  // flat base: only fillet top edges
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
    .cut(cutterTop).chamfer(0.4);
  // Honeycomb grid — nested loop over rows and columns
  let hc_width=35,hc_length=65,hc_depth=10,wallThickness=1,cellSize=5,rowNumber=5,colNumber=2;
  let deg30=Math.PI/6;
  let delta_x=(1+Math.sin(deg30))*cellSize+wallThickness*Math.cos(deg30);
  let delta_y=0.5*wallThickness+Math.cos(deg30)*cellSize;
  let point=[];
  for(let rowCount=1;rowCount<=rowNumber;rowCount++){
    for(let colCount=1;colCount<=colNumber;colCount++){
      // 8 symmetrically placed points per row/col combination
      point[1]=[(colCount-1)*2*delta_x,(rowCount-1)*delta_y*2,0];
      // ... (points 2-8 mirror in 4 quadrants)
      for(let j=1;j<=8;j++){
        let cutColumn=hexColumn(cellSize,5*th).translate(point[j])
          .intersect(r.makeBaseBox(hc_width,hc_length,hc_depth));
        shapeRounded=shapeRounded.cut(cutColumn);
      }
    }
  }
  return [{shape:receiver,name:"receiver",color:"dimgrey",opacity:0.8},
          {shape:shapeRounded,name:"holder",color:"steelblue",opacity:1.0}];
};
```
