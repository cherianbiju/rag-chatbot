---
source_file: handle_thorsten.js
category: structural
type: annotated_code
use_case: axle mount, mast handle, loftWith triangle fillet, assembly with debug
related: forked-lever-v4rc.md, shaft_design.md
---

# Handle Thorsten (UpwindBuddy Axle Mount)

## Description
Axle mount for an aluminum mast (UpwindBuddy windsurfing accessory). Combines an axle cylinder, a slanted handle with knob sphere, a rectangular sleeve with U-channel cutout and screw lock hole, and a strengthening triangular fillet (lofted between a rectangle and circle). Uses export default and named exports. Returns all parts individually colored plus test cross-section for verifying fit with aluminum extrusion.

## Keywords
axle mount, makeCylinder, makeBaseBox, makeSphere, loftWith, fillet, cut, fuseAll, intersect, DEG2RAD, export default, named exports, U-profile, sleeve, screw hole, test section, debug, named shapes, UpwindBuddy, windsurfing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| axleDia | 19 | mm | Axle cylinder diameter |
| axleLen | 50 | mm | Axle length (100/fct) |
| axleWall | 3 | mm | Axle wall thickness |
| handleLen | 35 | mm | Handle length (70/fct) |
| handlePos | 35 | mm | Handle Y-position along axle |
| handleAngle | 85 | deg | Handle slant angle from horizontal |
| sleeveLen | 37.5 | mm | Sleeve length (75/fct) |
| sleeveWall | 5 | mm | Sleeve wall thickness |
| uWidth | 10.5 | mm | U-channel internal width |
| uDepth | 13 | mm | U-channel depth |
| uThick | 1.5 | mm | U-channel wall thickness |
| triLen | 10 | mm | Strengthening triangle base length |
| screwDia | 5 | mm | Screw lock hole diameter |
| screwOff | 1.5 | mm | Screw offset from center |
| fct | 2 | - | Scale divisor applied to lengths |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| r.makeCylinder(r,len) | Creates axle and handle cylinders |
| .rotate(angle,[0,0,0],[axis]) | Rotates cylinder to correct axis |
| .translateZ(z) | Lifts to lay on XY plane |
| .translateY(y) | Positions handle along axle |
| r.makeBaseBox(l,w,h) | Creates box shapes (sleeve, keyway, test box) |
| .translate(x,y,z) | Positions box shapes |
| .fillet(r) | Rounds sleeve edges |
| .cut(other) | Cuts U-channel and screw hole from sleeve |
| r.makeSphere(r) | Creates knob sphere at handle end |
| r.drawRectangle(w,h) | Creates triangular fillet base sketch |
| r.drawCircle(r) | Creates triangular fillet top sketch |
| .translate(x,y) | Centers base/top sketches |
| .sketchOnPlane("XZ",z) | Places sketch at Z offset |
| .loftWith(other) | Lofts triangle between rectangle base and circle top |
| .fillet(r) | Rounds lofted triangle edges |
| fuseAll([shapes]) | Helper: reduces array via .fuse() |
| .intersect(other) | Creates cross-section test piece |
| .translateX(x) | Spreads named parts for display |
| r.DEG2RAD | Constant: Math.PI/180 |

## Code
```javascript
const r = replicad;
const fct=2;
export const defaultParams = {axleDia:19,axleLen:100/fct,axleWall:3,handleLen:70/fct,
  handlePos:70/fct,handleAngle:85,sleeveLen:75/fct,sleeveWall:5,uWidth:10.5,
  uDepth:13,uThick:1.5,triLen:10,screwDia:5,screwOff:1.5};
const fuseAll = (a) => a.slice(1).reduce((acc,s)=>acc.fuse(s.clone()),a[0].clone());
export default function main(p) {
  const axle = r.makeCylinder(p.axleDia/2,p.axleLen)
    .rotate(90,[0,0,0],[-1,0,0]).translateZ(p.axleDia/2);
  const handle = fuseAll([
    r.makeCylinder(p.axleDia/2,p.handleLen)
      .rotate(p.handleAngle,[0,0,0],[0,-1,0]).translateZ(p.axleDia/2),
    r.makeBaseBox(p.handleLen,p.axleDia/2,p.axleDia/2).translate(-p.handleLen/2,0,0),
    r.makeSphere(p.axleDia*0.55)
      .translate(-p.handleLen,0,p.axleDia/2+p.handleLen*Math.cos(p.handleAngle*r.DEG2RAD))
  ]).translateY(p.handlePos).rotate(90,[0,0,p.axleDia/2],[0,1,0]);
  const sleeveDepth=p.uDepth+2*p.sleeveWall, sleeveWidth=p.uWidth+2*p.sleeveWall;
  const screwHole = r.makeCylinder(p.screwDia/2,sleeveWidth+2).translate(0,-sleeveWidth/2+p.screwOff,-1);
  const sleeve = r.makeBaseBox(p.sleeveLen,sleeveDepth,sleeveWidth).fillet(2)
    .cut(r.makeBaseBox(p.sleeveLen+1,p.uDepth,p.uWidth).translate(0,0,p.sleeveWall))
    .translateY(-sleeveDepth/2).cut(screwHole);
  const triBase = r.drawRectangle(p.axleDia+p.triLen*2,sleeveWidth-2)
    .translate(0,sleeveWidth/2).sketchOnPlane("XZ",2);
  const triTop = r.drawCircle(p.axleDia/2).translate(0,p.axleDia/2).sketchOnPlane("XZ",-p.triLen);
  const tri = triTop.loftWith(triBase).fillet(1);
  const all = fuseAll([axle,handle,sleeve,tri]);
  const testSectBox = r.makeBaseBox(20,sleeveDepth+p.triLen+10,sleeveWidth+10).translate(0,-5,-5);
  const testSect = testSectBox.clone().intersect(all.clone());
  return [
    {shape:all.translateX(100),name:"all",color:"#ffeeee"},
    {shape:axle.translateX(-100),name:"axle",color:"orange"},
    {shape:handle.translateX(-100),name:"handle",color:"blue"},
    {shape:sleeve.translateX(-100),name:"sleeve",color:"green"},
    {shape:screwHole.translateX(-100),name:"screwHole",color:"yellow"},
    {shape:tri.translateX(-100),name:"tri",color:"olive"},
    {shape:testSect,name:"testSect",color:"purple"},
    {shape:testSectBox,name:"testSectBox",color:"steelblue"},
    {shape:all.clone(),name:"testSectAll",color:"lightgreen"}
  ];
}
```
