---
source_file: gridfinity.js
category: enclosure
type: annotated_code
use_case: gridfinity bin, modular storage, parametric grid, 3D printing
related: holder.md, birdhouse.md
---

# Gridfinity Bin

## Description
Full parametric Gridfinity-compatible storage bin. Builds three layers: a bottom socket grid with tapered walls (matching the Gridfinity spec's exact magic numbers), a hollow box body, and a top lip profile. Sockets can include magnet and/or screw cutouts at 13mm from center. Grid cloning uses a helper utility. Demonstrates advanced replicad patterns: sweepSketch with withContact, BlueprintSketcher, intersectBlueprints, cutBlueprints, makeSolid, and shell.

## Keywords
gridfinity, modular storage, socket, magnet cutout, screw cutout, sweepSketch, withContact, roundedRectangleBlueprint, BlueprintSketcher, intersectBlueprints, cutBlueprints, makeSolid, makeFace, assembleWire, EdgeFinder, shell, cloneOnGrid, fillet, inBox

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| xSize | 4 | cells | Grid cells in X |
| ySize | 2 | cells | Grid cells in Y |
| heigth | 0.5 | units | Height in Gridfinity units (×42mm) |
| withMagnet | true | - | Include magnet cutouts |
| withScrew | true | - | Include screw cutouts |
| magnetRadius | 3.25 | mm | Magnet hole radius |
| magnetHeight | 2 | mm | Magnet hole depth |
| screwRadius | 1.5 | mm | Screw hole radius |
| keepFull | false | - | Solid box (no shell) if true |
| wallThickness | 1.2 | mm | Shell wall thickness |
| SIZE | 42.0 | mm | Gridfinity standard cell pitch |
| CLEARANCE | 0.5 | mm | Standard fit clearance |
| SOCKET_HEIGHT | 5 | mm | Socket depth |
| SOCKET_BIG_TAPER | 2.4 | mm | Large chamfer on socket |
| SOCKET_SMALL_TAPER | 0.8 | mm | Small chamfer on socket |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| roundedRectangleBlueprint(w,h,r) | Creates 2D rounded rectangle blueprint |
| .sketchOnPlane() | Places blueprint on XY plane |
| .sweepSketch(fn, {withContact:true}) | Sweeps socket/top profile along outline |
| new BlueprintSketcher() | 2D blueprint sketcher for socket profile |
| .movePointerTo() .vLine() .lineTo() .close() | Draws socket taper profile |
| intersectBlueprints(a,b) | 2D blueprint intersection for top profile |
| cutBlueprints(a,b) | 2D blueprint subtract |
| makeSolid([faces]) | Creates solid from faces list |
| makeFace(wire) | Creates face from wire |
| assembleWire(edges) | Assembles edges into wire |
| new EdgeFinder().inPlane("XY",z).find(shape) | Finds edges at height for face construction |
| sketchCircle(r).extrude(h) | Creates magnet/screw cutout cylinders |
| .fuse(other) | Fuses magnet+screw cutout |
| .cut(cutout.clone().translate([x,y,z])) | Cuts 4 corner holes |
| .clone().translate([...]) | Clones socket to grid positions |
| .fuse(b, {optimisation:"commonFace"}) | Optimised fuse for adjacent coplanar faces |
| .shell(t, faceFinder) | Hollows box open at top |
| .fillet(r, e=>e.inBox([p1],[p2])) | Fillets top corner of socket profile |
| .translateZ(h) | Lifts top profile to box height |

## Code
```javascript
const defaultParams = {xSize:4,ySize:2,heigth:0.5,withMagnet:true,withScrew:true,
  magnetRadius:3.25,magnetHeight:2,screwRadius:1.5,keepFull:false,wallThickness:1.2};
const SIZE=42.0, CLEARANCE=0.5, AXIS_CLEARANCE=(CLEARANCE*Math.sqrt(2))/4;
const CORNER_RADIUS=4, TOP_FILLET=0.6, SOCKET_HEIGHT=5;
const SOCKET_SMALL_TAPER=0.8, SOCKET_BIG_TAPER=2.4;
const SOCKET_VERTICAL_PART=SOCKET_HEIGHT-SOCKET_SMALL_TAPER-SOCKET_BIG_TAPER;
const SOCKET_TAPER_WIDTH=SOCKET_SMALL_TAPER+SOCKET_BIG_TAPER;
function main({roundedRectangleBlueprint,sketchCircle,BlueprintSketcher,
               intersectBlueprints,cutBlueprints,makeSolid,assembleWire,makeFace,EdgeFinder}, config) {
  const socketProfile = (_,startPoint) => {
    const full = new BlueprintSketcher()
      .movePointerTo([-CLEARANCE/2,0]).vLine(-CLEARANCE/2)
      .lineTo([-SOCKET_BIG_TAPER,-SOCKET_BIG_TAPER]).vLine(-SOCKET_VERTICAL_PART)
      .line(-SOCKET_SMALL_TAPER,-SOCKET_SMALL_TAPER).done()
      .translate(CLEARANCE/2,0);
    return full?.sketchOnPlane("XZ",startPoint);
  };
  const buildSocket = ({magnetRadius=3.25,magnetHeight=2,screwRadius=1.5,withScrew=true,withMagnet=true}={}) => {
    const baseSocket = roundedRectangleBlueprint(SIZE-CLEARANCE,SIZE-CLEARANCE,CORNER_RADIUS).sketchOnPlane();
    const slotSide = baseSocket.sweepSketch(socketProfile,{withContact:true});
    let slot = makeSolid([slotSide,
      makeFace(assembleWire(new EdgeFinder().inPlane("XY",-SOCKET_HEIGHT).find(slotSide))),
      makeFace(assembleWire(new EdgeFinder().inPlane("XY",0).find(slotSide)))]);
    if(withScrew||withMagnet){
      const magnetCutout = withMagnet ? sketchCircle(magnetRadius).extrude(magnetHeight) : null;
      const screwCutout = withScrew ? sketchCircle(screwRadius).extrude(SOCKET_HEIGHT) : null;
      const cutout = magnetCutout&&screwCutout ? magnetCutout.fuse(screwCutout) : magnetCutout||screwCutout;
      slot=slot.cut(cutout.clone().translate([-13,-13,-5])).cut(cutout.clone().translate([-13,13,-5]))
               .cut(cutout.clone().translate([13,13,-5])).cut(cutout.clone().translate([13,-13,-5]));
    }
    return slot;
  };
  const cloneOnGrid=(shape,{xSteps=1,ySteps=1,span=10,xSpan=null,ySpan=null})=>{
    const xCorr=((xSteps-1)*(xSpan||span))/2, yCorr=((ySteps-1)*(ySpan||xSpan||span))/2;
    const translations=[...Array(xSteps).keys()].flatMap(i=>[...Array(ySteps).keys()].map(j=>[i*SIZE-xCorr,j*SIZE-yCorr,0]));
    return translations.map(t=>shape.clone().translate(t));
  };
  function run({xSize=2,ySize=1,heigth=0.5,keepFull=false,wallThickness=1.2,
                withMagnet=false,withScrew=false,magnetRadius=3.25,magnetHeight=2,screwRadius=1.5}={}) {
    const stdHeight=heigth*SIZE;
    let box=roundedRectangleBlueprint(xSize*SIZE-CLEARANCE,ySize*SIZE-CLEARANCE,CORNER_RADIUS).sketchOnPlane().extrude(stdHeight);
    if(!keepFull) box=box.shell(wallThickness,(f)=>f.inPlane("XY",stdHeight));
    const top=buildTopShape({xSize,ySize,includeLip:!keepFull}).translateZ(stdHeight);
    const socket=buildSocket({withMagnet,withScrew,magnetRadius,magnetHeight,screwRadius});
    let base=null;
    cloneOnGrid(socket,{xSteps:xSize,ySteps:ySize,span:SIZE}).forEach(s=>{
      if(base) base=base.fuse(s,{optimisation:"commonFace"}); else base=s;
    });
    return base.fuse(box,{optimisation:"commonFace"}).fuse(top,{optimisation:"commonFace"});
  }
  return run(config);
}
```
