---
source_file: earbud_holder.js
category: enclosure
type: annotated_code
use_case: earbud clip holder, wearable accessory, sweep along bezier curve
related: earbud2-script.md, earbud3-script.md, earbud4-script.md
---

# Earbud Holder

## Description
Parametric earbud holder with a swept rectangular hook along a 4-segment cubic bezier curve path. The holder body grips the earbud stem (modeled as an ellipse cross-section cutout) with a flexibility slit. Outputs left and right mirrored holders, the hook, and the stem preview shape as a named array.

## Keywords
earbud holder, cubicBezierCurveTo, sweepSketch, sketchEllipse, sketchRectangle, fillet, mirror, cut, fuse, bezier curve, sweep, wearable, tolerance, clone, YZ mirror, inDirection, inPlane

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| earbudStemWidth | 4.5 | mm | Stem ellipse X half-dimension |
| earbudStemThickness | 4.0 | mm | Stem ellipse Z half-dimension |
| earbudStemHeight | 20.0 | mm | Stem extrude length |
| tolerance | 0.5 | mm | Added clearance to stem dims for fit |
| holderThickness | 1.5 | mm | Wall thickness of holder body |
| hookWidth | 3 | mm | Hook cross-section width |
| hookHeight | 2 | mm | Hook cross-section height |
| holderLength | 8 | mm | Length of grip body |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw() | Starts 2D freeform drawing for hook path |
| .cubicBezierCurveTo(end, cp1, cp2) | Cubic bezier segment with two control points |
| .done() | Finalizes open 2D path |
| .sketchOnPlane("XY") | Places 2D path on XY plane |
| new Sketcher("XY", z) | Creates sketcher at Z height |
| .movePointerTo([x,y]) | Moves cursor to start point |
| .sweepSketch(fn) | Sweeps a profile sketch along a path |
| sketchRectangle(w, h, {plane, origin}) | Creates hook cross-section profile |
| sketchEllipse(rx, ry, {plane}) | Creates elliptical stem cross-section |
| .extrude(h) | Extrudes sketch to 3D |
| .fillet(r, edgeFinder) | Rounds edges at planes or directions |
| .inDirection("Z") | Selects vertical edges |
| .inPlane("XY", z) | Selects edges at Z height |
| .fuse(other) | Boolean union |
| .cut(other) | Boolean subtract |
| .translate([x,y,z]) | Moves shape |
| .clone() | Duplicates shape |
| .mirror("YZ", origin) | Mirrors across YZ plane for right holder |

## Code
```javascript
const defaultParams = { earbudStemWidth:4.5, earbudStemThickness:4, earbudStemHeight:20.0 };
function main({ draw, Sketcher, sketchRectangle, sketchEllipse },
              { earbudStemWidth, earbudStemThickness, earbudStemHeight }) {
  let tp1=[28,25], cp1s=[15,0], cp1e=[28,15];
  let tp2=[10,42], cp2s=[28,35], cp2e=[20,42];
  let tp3=[-12,32], cp3s=[0,42], cp3e=[-8,32];
  let tp4=[-20,35], cp4s=[-14,32], cp4e=[-16,32];
  let hookCurveSketch = new Sketcher("XY",1).movePointerTo([0,0])
    .cubicBezierCurveTo(tp1,cp1s,cp1e).cubicBezierCurveTo(tp2,cp2s,cp2e)
    .cubicBezierCurveTo(tp3,cp3s,cp3e).cubicBezierCurveTo(tp4,cp4s,cp4e).done();
  let hookWidth=3, hookHeight=2;
  let loftedHook = hookCurveSketch.sweepSketch((plane,origin) =>
    sketchRectangle(hookWidth,hookHeight,{plane,origin}));
  loftedHook = loftedHook.fillet(1,(e)=>e.inDirection("Z"))
                         .fillet(0.75,(e)=>e.inPlane("XY",2));
  let tolerance=0.5;
  earbudStemThickness+=tolerance; earbudStemWidth+=tolerance;
  let holderThickness=1.5, holderWidth=earbudStemWidth+(2*holderThickness);
  let holderLength=8, holderHeight=earbudStemThickness+holderThickness+hookHeight;
  let earPodHolder = sketchRectangle(holderWidth,holderLength).extrude(holderHeight)
    .translate([0,-((holderLength/2)-(hookWidth/2)),0]);
  let earPodStem = sketchEllipse(earbudStemWidth/2,earbudStemThickness/2,{plane:"XZ"})
    .extrude(earbudStemHeight).translate([0,earbudStemHeight/8,earbudStemThickness/2+hookHeight]);
  let slit = sketchRectangle(2,40,{plane:"XY"}).extrude(holderHeight).translate([0,0,holderHeight/2]);
  earPodHolder = earPodHolder.fuse(loftedHook.clone()).cut(earPodStem.clone()).cut(slit)
    .fillet(0.75,(e)=>e.inDirection("Y").inPlane("XY",holderHeight));
  let earPodHolderR = earPodHolder.clone().mirror("YZ",[-30,0]);
  return [
    {shape:loftedHook,name:"loftedHook",color:"gray"},
    {shape:earPodHolder,name:"holderL"},
    {shape:earPodHolderR,name:"holderR"},
    {shape:earPodStem,name:"earPodStem"}
  ];
}
```
