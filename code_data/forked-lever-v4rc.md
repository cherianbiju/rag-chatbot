---
source_file: forked-lever-v4rc.js
category: structural
type: annotated_code
use_case: forked lever, intersect-based modeling, mechanical linkage, ModelMania
related: curveSlot2.md, crankshaft.md, shaft_design.md
---

# Forked Lever V4

## Description
Forked lever created by intersecting a front-view lever body (two hubs connected by tapered arms) with a side-view extrusion that defines the fork shape. Three helper functions build the lever body (Lever), the inner material cutout (Cutout), and the side profile (SideView). The intersect operation is the key modeling step that creates the fork geometry. Inspired by ModelMania 2010 and JokoEngineering exercise.

## Keywords
forked lever, intersect, Lever function, Cutout, SideView, Sketcher, sketchCircle, threePointsArcTo, lineTo, extrude, cut, fillet, inDirection, inBox, XZ plane, XY plane, PolarY helper, mechanical linkage, ModelMania

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| t1 | 8 | mm | Outer arm thickness |
| t2 | 10 | mm | Inner fork thickness |
| t3 | 5 | mm | Clearance gap between fork arms and hubs |
| sep | 40 | mm | Fork separation distance |
| radius_1 | 30 | mm | Large hub radius |
| radius_2 | 12 | mm | Small hub radius |
| distance | 120 | mm | Center-to-center distance |
| wall_thickness | 5 | mm | Hub bore wall thickness |
| lever_height | 96 | mm | Total lever height (2×sep + 4×t1) |
| fillet_cutout | 5 | mm | Fillet on inner material cutout edges |
| fillet_outer | 1 | mm | Overall outer edge fillet |
| fillet_arms | 5 | mm | Fillet on side-view arm edges |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| new Sketcher("XY") | Front view lever sketch |
| .movePointerTo([x,y]) | Moves to start point |
| .lineTo([x,y]) | Straight line to absolute point |
| .threePointsArcTo([end],[mid]) | Arc through midpoint to endpoint |
| .close() | Closes sketch |
| .extrude(h) | Extrudes to 3D |
| sketchCircle(r) | Creates hub bore circle |
| .cut(hole) | Cuts bore through lever |
| .fillet(r, edgeFinder) | Rounds edges |
| .inDirection("Z") | Selects vertical edges |
| .inDirection("Y") | Selects Y-direction edges |
| .inBox([p1],[p2]) | Selects edges within bounding box |
| new Sketcher("XZ") | Side view sketch on XZ plane |
| .intersect(sideExtrude) | Cuts lever to fork shape via intersection |
| PolarY(pt, ydist, angle) | Helper: computes point by Y-distance and angle |

## Code
```javascript
function main({Sketcher, sketchCircle}) {
  let t1=8,t2=10,t3=5,sep=40,radius_1=30,radius_2=12,distance=120,wall_thickness=5;
  let lever_height=2*sep+(4*t1), fillet_cutout=5, fillet_outer=1, fillet_arms=5;
  let angle=Math.atan((sep/2+t1-t2/2)/(distance-radius_1-radius_2-2*t3))*180/Math.PI;
  function PolarY(pt,ydist,angleDeg){
    let rad=angleDeg*Math.PI/180;
    return [pt[0]+ydist/Math.tan(rad), pt[1]+ydist];
  }
  function Lever(r1,r2,dist,height,wall){
    let sin_a=(r1-r2)/dist, a=Math.asin(sin_a);
    let p1=[r1*Math.sin(a),r1*Math.cos(a)], p2=[dist+r2*Math.sin(a),r2*Math.cos(a)];
    let p3=[dist+r2,0], p4=[dist+r2*Math.sin(a),-r2*Math.cos(a)];
    let p5=[r1*Math.sin(a),-r1*Math.cos(a)], p6=[-r1,0];
    let body = new Sketcher("XY").movePointerTo(p1).lineTo(p2)
      .threePointsArcTo(p4,p3).lineTo(p5).threePointsArcTo(p1,p6).close().extrude(height);
    return body.cut(sketchCircle(r1-wall).extrude(height+10))
               .cut(sketchCircle(r2-wall).extrude(height+10).translate([dist,0,0]));
  }
  function Cutout(r1,r2,dist,height,wall_t,iR1,iR2){ /* inner relief cutout */ }
  function SideView(r1,r2,dist,sep,t1,t2,t3){ /* fork side profile on XZ */ }
  let myLever = Lever(radius_1,radius_2,distance,lever_height,wall_thickness);
  let cutoutShape = Cutout(radius_1,radius_2,distance,lever_height,wall_thickness,radius_1,radius_2);
  cutoutShape = cutoutShape.fillet(fillet_cutout,(e)=>e.inDirection("Z"));
  myLever = myLever.cut(cutoutShape);
  let sideview = SideView(radius_1,radius_2,distance,sep,t1,t2,t3)
    .extrude(radius_1*2).translate([0,radius_1,lever_height/2])
    .fillet(fillet_arms,(e)=>e.inDirection("Y").inBox([0,-radius_1,0],[distance,radius_1,lever_height]));
  let fork = myLever.intersect(sideview).fillet(fillet_outer);
  return fork;
}
```
