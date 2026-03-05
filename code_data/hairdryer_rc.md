---
source_file: hairdryer_rc.js
category: enclosure
type: annotated_code
use_case: hair dryer model, shell operation, fillet with not/inBox, complex assembly
related: hairdryer_cs.md, birdhouse.md, bottle.md
---

# Hair Dryer (Modern Replicad API)

## Description
Complete hair dryer model in modern replicad style, porting the legacy CadScript version. Fan housing is an extruded circle with an air intake cutout. An outlet duct and handle rectangles are fused on. Junction fillets use inDirection("Z").not(inBox(...)) to select only the relevant edges. The combined solid is shelled with one face open. A lid with stem is added separately after shelling.

## Keywords
hair dryer, shell, sketchCircle, sketchRectangle, fuse, cut, fillet, inDirection, not, inBox, inPlane, ofCurveType, translateZ, EdgeFinder, hollow, outlet, handle, lid, stem, Braun, junction fillet, shell open face

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| fanRadius | 30 | mm | Fan housing cylinder radius |
| fanHeight | 30 | mm | Fan housing height |
| fanhousingThickness | 1.5 | mm | Shell wall thickness |
| fanCutoutRadius | 20 | mm | Air intake cutout radius |
| fanRounding | 8 | mm | Fan housing fillet |
| lidRadius | 19 | mm | Lid disk radius |
| lidThickness | 3 | mm | Lid thickness |
| lidRounding | 2 | mm | Lid circular edge fillet |
| stemRadius | 5 | mm | Lid stem radius |
| outletLength | 60 | mm | Outlet duct length |
| outletWidth | 18 | mm | Outlet duct width (extrude) |
| outletHeight | 30 | mm | Outlet duct height (sketch) |
| outletRounding | 5 | mm | Outlet Y-edge fillet |
| outletJunctionRound | 5 | mm | Outlet-to-housing junction fillet |
| handleLength | 80 | mm | Handle length |
| handleWidth | 25 | mm | Handle width |
| handleHeight | 16 | mm | Handle height |
| handleRounding | 5 | mm | Handle X-edge fillet |
| handleBottomRound | 2 | mm | Handle end-face fillet |
| handleJunctionRound | 5 | mm | Handle-to-housing junction fillet |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| sketchCircle(r) | Creates circular cross-section |
| .extrude(h) | Extrudes to cylinder or disk |
| .fillet(r) | Rounds all edges of fan housing |
| .cut(other) | Subtracts air intake cutout |
| .translateZ(z) | Lifts cutout, lid to correct Z height |
| .ofCurveType("CIRCLE") | Selects circular edges for lid fillet |
| .inPlane("XY",z) | Selects edges at specific Z height |
| sketchRectangle(w,h) | Creates outlet and handle rectangles |
| .translate([x,y,z]) | Positions outlet and handle |
| .fillet(r, e=>e.inDirection("Y")) | Rounds outlet long-axis edges |
| .fillet(r, e=>e.inDirection("X")) | Rounds handle long-axis edges |
| .inDirection("Z") | Selects vertical junction edges |
| .not(fn) | Inverts the edge filter |
| .inBox([p1],[p2]) | Defines exclusion zone for junction fillet |
| .inPlane("YZ",x) | Selects handle end face at X position |
| .fuse(other) | Boolean union |
| .shell(-t, f=>f.inPlane("XZ",[pt])) | Shells solid, open at back/bottom face |

## Code
```javascript
const main = ({sketchCircle, sketchRectangle, EdgeFinder}) => {
  let fanRadius=30, fanHeight=30, fanhousingThickness=1.5;
  let fanCutoutRadius=20, fanCutoutDepth=5, fanRounding=8;
  let lidRadius=19, lidThickness=3, lidRounding=2, stemRadius=5;
  let outletLength=60, outletWidth=18, outletHeight=30, outletRounding=5;
  let handleLength=80, handleWidth=25, handleHeight=16;
  let handleRounding=5, handleBottomRound=2;
  let handleJunctionRound=5, outletJunctionRound=5;
  // Fan housing
  let fanhousing = sketchCircle(fanRadius).extrude(fanHeight).fillet(fanRounding);
  let cutout = sketchCircle(fanCutoutRadius).extrude(fanCutoutDepth+10).translateZ(fanHeight-fanCutoutDepth);
  fanhousing = fanhousing.cut(cutout);
  // Lid with circular edge fillet and stem
  let lid = sketchCircle(lidRadius).extrude(lidThickness)
    .fillet(lidRounding,(e)=>e.ofCurveType("CIRCLE").inPlane("XY",lidThickness))
    .translateZ(fanHeight-lidThickness);
  let lidstem = sketchCircle(stemRadius).extrude(fanHeight);
  lid = lid.fuse(lidstem);
  // Outlet duct
  let outletBase=(fanHeight-outletWidth)/2;
  let outlet = sketchRectangle(outletHeight,outletLength).extrude(outletWidth)
    .translate([(-fanRadius+(outletHeight/2.0)),-outletLength/2,outletBase])
    .fillet(outletRounding,(e)=>e.inDirection("Y"));
  fanhousing = fanhousing.fuse(outlet);
  let corner1=[0,-outletLength,-fanHeight/2], corner2=[-fanRadius,-fanRadius-5,fanHeight/2];
  fanhousing = fanhousing.fillet(outletJunctionRound,(e)=>e.inDirection("Z").not((f)=>f.inBox(corner1,corner2)));
  // Handle
  let handleBase=(fanHeight-handleHeight)/2;
  let handle = sketchRectangle(handleLength,handleWidth).extrude(handleHeight)
    .translate([handleLength/2,fanRadius-handleWidth/2,handleBase])
    .fillet(handleRounding,(e)=>e.inDirection("X"))
    .fillet(handleBottomRound,(e)=>e.inPlane("YZ",handleLength));
  fanhousing = fanhousing.fuse(handle);
  fanhousing = fanhousing.fillet(handleJunctionRound,(e)=>e.inDirection("Z").not((f)=>f.inBox(corner1,corner2)));
  // Shell — open at the back/bottom face
  fanhousing = fanhousing.shell(-fanhousingThickness,(f)=>f.inPlane("XZ",[-fanRadius,-outletLength,0]));
  fanhousing = fanhousing.fuse(lid);
  return fanhousing;
};
```
