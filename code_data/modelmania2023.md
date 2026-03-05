---
source_file: modelmania2023.js
category: mechanical
type: annotated_code
use_case: SolidWorks Model Mania 2023 — split clamp assembly with cylindrical body, holder arm with counterbored hole, clamp arm, and gap slot
related: mm2001-v2.md, mm2008_v5.md, mm2016-v7.md, holderv7.md
---
# Model Mania 2023 — Split Clamp Assembly

## Description
Recreation of the SolidWorks Model Mania 2023 challenge part: a split cylindrical clamp with two opposing arms. The main cylinder with fillet is fused with a holder arm (rectangular body + semicircular end + hollow pocket + counterbored hole) and a clamp arm (similar geometry, mirrored), then the bore, a hollow slot pocket, and a thin gap cut are applied. Final selective fillets clean up interior and exterior edges.

## Keywords
Model-Mania-2023, SolidWorks, clamp, split-clamp, cylinder, bore, holder, counterbore, fillet, inBox, inDirection, inPlane, makeBaseBox, makeCylinder, fuse, cut, gapHole, replicad, mechanical, assembly, 3d-printing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| mainCylinder radius | 50/2 = 25 | mm | Outer radius of main cylinder |
| mainCylinder height | 30 | mm | Height of main cylinder |
| mainBore radius | 42/2 = 21 | mm | Bore radius |
| filletRadius | 2 | mm | General fillet applied throughout |
| holder body | 50.5×15×30 | mm | Rectangular holder arm body |
| holder round radius | 30/2 = 15 | mm | Semicircular end of holder arm |
| holder hole radius | 6.6/2 = 3.3 | mm | Through-hole in holder arm |
| holderCounterR/L radius | 11/2 = 5.5 | mm | Counterbore radius on holder |
| holderHollow | 30×50×14 | mm | Hollow pocket cut in holder back |
| clamp body | 30×30×25 | mm | Rectangular clamp arm body |
| clamp round radius | 12.5 | mm | Semicircular end of clamp arm |
| clamp hole radius | 6.6/2 = 3.3 | mm | Through-hole in clamp arm |
| clampCounterR/L radius | 11/2 = 5.5 | mm | Counterbore on clamp arm |
| gapHole | 42.5×2×50 | mm | Thin gap slot separating clamp halves |

## Code
```javascript
// Model mania 2023
const r = replicad

function main()
{
let filletRadius = 2;

let mainCylinder = r.makeCylinder(50/2,30,[0,0,0],[0,0,1])
.translate(0,0,-30/2).fillet(filletRadius)
let mainBore = r.makeCylinder(42/2,50,[0,0,-10],[0,0,1])
.translate(0,0,-30/2)

let holder = r.makeBaseBox(65.5-15,15,30)
.translate((65.5-15)/2,50/2-15/2,-30/2)
.fillet(filletRadius,(e)=>e.inDirection("X"))
let holderRound = r.makeCylinder(30/2,15,[65.5-15,50/2,0],[0,-1,0]).fillet(filletRadius)
let holderHole = r.makeCylinder(6.6/2,40,[65.5-15,50/2+30/2,0],[0,-1,0])
let holderCounterR = r.makeCylinder(11/2,12,[65.5-15,50/2+6,0],[0,-1,0])
let holderCounterL = r.makeCylinder(11/2,12,[65.5-15,50/2-15+6,0],[0,-1,0])
let holderHollow = r.makeBaseBox(30,50,14)
.translate(30/2,20,-14/2).fillet(filletRadius,(e)=>e.inPlane("YZ",30))

let clamp = r.makeBaseBox(42.5-12.5,30,25)
.translate(-(42.5-12.5)/2,0,-25/2)
.fillet(filletRadius,(e)=>e.inDirection("X"))
let clampRound = r.makeCylinder(12.5,30,[-42.5+12.5,30/2,0],[0,-1,0]).fillet(filletRadius)
let clampHole = r.makeCylinder(6.6/2,40,[-42.5+12.5,40/2,0],[0,-1,0])
let clampCounterR = r.makeCylinder(11/2,8,[-42.5+12.5,15-6,0],[0,1,0])
let clampCounterL = r.makeCylinder(11/2,8,[-42.5+12.5,-15+6,0],[0,-1,0])
let gapHole = r.makeBaseBox(42.5,2,50).translate(-42.5/2,0,-50/2)

holder = holder.fuse(holderRound).cut(holderHollow).cut(holderHole)
holder = holder.cut(holderCounterL).cut(holderCounterR)
clamp = clamp.fuse(clampRound)
clamp = clamp.cut(clampHole).cut(clampCounterL).cut(clampCounterR)
mainCylinder = mainCylinder.fuse(holder).fuse(clamp).cut(mainBore)
mainCylinder = mainCylinder.fillet(0.9,(e)=>e.inBox([-17,-30,-15],[-22,30,15]))
mainCylinder = mainCylinder.cut(gapHole)
mainCylinder = mainCylinder.fillet(2,(e)=>e.inBox([17,-30,-20],[22,30,20]))

let shapes= [
{shape: mainCylinder, name:"mainCylinder"},
]  

return shapes
}
```
