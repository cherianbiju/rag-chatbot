---
source_file: hairdryer_cs.js
category: enclosure
type: annotated_code
use_case: hair dryer model, legacy CadScript API, API migration reference
related: hairdryer_rc.md
---

# Hair Dryer (CadScript Legacy API)

## Description
Hair dryer model based on a 1970s Braun design using a legacy CadScript-style API (Cylinder, Box, Translate, FilletEdges, Difference, Union, Offset, ChamferEdges) rather than standard replicad. Includes fan housing with air intake cutout, lid, outlet duct, handle with button recesses, buttons, outlet vanes, and optional cross-section cut. Useful as a migration reference — see hairdryer_rc.js for the modern equivalent.

## Keywords
hair dryer, legacy CadScript, Cylinder, Box, Translate, FilletEdges, Difference, Union, Offset, ChamferEdges, fanhousing, outlet, handle, button, vane, migration reference, hollow shell, Braun

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| fanRadius | 30 | mm | Fan housing cylinder radius |
| fanHeight | 30 | mm | Fan housing height |
| fanhousingThickness | 1.5 | mm | Shell wall thickness |
| fanCutoutRadius | 20 | mm | Air intake cutout radius |
| fanCutoutDepth | 5 | mm | Air intake cutout depth |
| fanRounding | 8 | mm | Fan housing edge fillet |
| lidRadius | 19 | mm | Lid disk radius |
| lidThickness | 3 | mm | Lid thickness |
| outletLength | 60 | mm | Outlet duct length |
| outletWidth | 30 | mm | Outlet duct width |
| outletHeight | 18 | mm | Outlet duct height |
| handleLength | 80 | mm | Handle length |
| handleWidth | 25 | mm | Handle width |
| handleHeight | 16 | mm | Handle height |
| buttonLength | 8 | mm | Button depth |
| buttonWidth | 3 | mm | Button width |
| buttonHeight | 6 | mm | Button height |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| Cylinder(r, h) | Legacy: creates cylinder solid |
| Box(w, l, h) | Legacy: creates box solid |
| Translate([x,y,z], shape) | Legacy: moves shape to position |
| FilletEdges(shape, r, [indices]) | Legacy: fillets specific edges by index |
| Difference(shape, [tools]) | Legacy: boolean subtract |
| Union([shapes]) | Legacy: boolean union |
| Offset(shape, -t, tol, inner) | Legacy: creates shell/offset |
| ChamferEdges(shape, r, [indices]) | Legacy: chamfers specific edges by index |

## Code
```javascript
// NOTE: This file uses legacy CadScript API — see hairdryer_rc.js for modern replicad equivalent
let fanRadius=30, fanHeight=30, fanhousingThickness=1.5, fanCutoutRadius=20;
let fanCutoutDepth=5, fanRounding=8, lidRadius=19, lidThickness=3;
let outletLength=60, outletWidth=30, outletHeight=18, outletRounding=5;
let handleLength=80, handleWidth=25, handleHeight=16, handleRounding=5;
let buttonLength=8, buttonWidth=3, buttonHeight=6;
// Fan housing
let fanhousing = Cylinder(fanRadius,fanHeight);
fanhousing = FilletEdges(fanhousing,fanRounding,[0,2],false);
let cutout = Translate([0,0,fanHeight-fanCutoutDepth],Cylinder(fanCutoutRadius,fanCutoutDepth+10));
fanhousing = Difference(fanhousing,[cutout],false);
let fanhousing_inner = Offset(fanhousing,-fanhousingThickness,0.01,true);
// Lid
let lid = Translate([0,0,fanHeight-lidThickness],Cylinder(lidRadius,lidThickness));
lid = FilletEdges(lid,2,[0]);
// Outlet
let outlet = Translate([-0.2,0,(fanHeight-outletHeight)/2], Box(outletWidth,outletLength,outletHeight));
outlet = FilletEdges(outlet,outletRounding,[1,3,5,7]);
// Handle with button recess
let handle = Translate([0,-30,(fanHeight-handleHeight)/2],Box(-handleLength,handleWidth,handleHeight));
handle = FilletEdges(handle,handleRounding,[11,10,9,8]);
let button_cut = Translate([-50,-10,10],Box(recesWidth,recesDepth,recesHeight));
button_cut = FilletEdges(button_cut,recesRounding,[1,5,7,3]);
handle = Difference(handle,[button_cut]);
// Combine and hollow
let dryer_solid = Union([fanhousing,outlet,handle],false,0.01,false);
dryer_solid = FilletEdges(dryer_solid,outletJunctionRound,[49]);
let dryer_inner = Union([fanhousing_inner,outlet_in,handle_inner]);
let dryer_hollow = Difference(dryer_solid,[dryer_inner],false,0.5,false);
// Vanes
let vane = Translate([4,46,7.5],Box(1,15,15));
vane = ChamferEdges(vane,2,[10,11],false);
for(let j=0;j<=3;j++) vanes[j]=Translate([5*(j+1),0,0],vane,true);
```
