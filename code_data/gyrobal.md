---
source_file: gyrobal.js
category: replicad_example
type: annotated_code
use_case: torus shapes, off-axis revolve, gyroscope ring, steering wheel experiment
related: addthickness.md, addthickness_v2.md
---

# Gyrobal (Gyroscope Ring Experiment)

## Description
Steering wheel / gyroscope ring experiment. Creates two perpendicular torus (donut) shapes by revolving offset circles around different axes — one in XZ plane around Z axis, one in XY plane around X axis. The second donut is scaled to 99% to avoid coincident face issues when fusing. A box spoke section is present but commented out. Returns the fused donut pair.

## Keywords
revolve, torus, donut, sketchCircle, origin offset, off-axis revolve, scale, fuse, sketchRectangle, fillet, inDirection, XZ plane, XY plane, gyroscope, steering wheel, 0.99 scale trick

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| wheelDiameter | 120 | mm | Distance from center to tube center |
| width | 10 | mm | Width of spoke box |
| height | 10 | mm | Height of spoke box |
| radius | 10 | mm | Cross-section radius of each torus tube |
| filletSpoke | 1.5 | mm | Fillet on spoke X-edges (fails above 1.6) |
| filletWheelSpoke | 3 | mm | Junction fillet — commented out |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| sketchRectangle(w,h,{plane}) | Creates rectangular spoke on XZ plane |
| .extrude(h) | Extrudes to box |
| .fillet(r, e=>e.inDirection("X")) | Rounds spoke X-direction edges |
| sketchCircle(r,{plane:"XZ", origin:[x,0,0]}) | Circle offset in X for XZ-plane torus |
| .revolve([0,0,1],{origin:[0,0,0]}) | Revolves around Z axis → torus 1 |
| sketchCircle(r,{plane:"XY", origin:[0,y,0]}) | Circle offset in Y for XY-plane torus |
| .revolve([1,0,0],{origin:[0,0,0]}) | Revolves around X axis → torus 2 |
| .scale(0.99) | Scales 99% to avoid coincident faces on fuse |
| .fuse(other) | Boolean union of two perpendicular tori |

## Code
```javascript
function main({sketchRectangle, sketchCircle}) {
  let wheelDiameter=120, width=10, height=10, radius=10;
  let filletSpoke=1.5, filletWheelSpoke=3;
  let rectangle = sketchRectangle(wheelDiameter,width,{plane:"XZ"});
  let box = rectangle.extrude(height).fillet(filletSpoke,(e)=>e.inDirection("X"));
  // torus 1: circle in XZ, offset wheelDiameter/2 in X, revolve around Z
  let donut = sketchCircle(radius,{plane:"XZ",origin:[wheelDiameter/2,0,0]})
    .revolve([0,0,1],{origin:[0,0,0]});
  // torus 2: circle in XY, offset wheelDiameter/2 in Y, revolve around X, scaled 99%
  let donut2 = sketchCircle(radius,{plane:"XY",origin:[0,wheelDiameter/2,0]})
    .revolve([1,0,0],{origin:[0,0,0]}).scale(0.99);
  return donut.fuse(donut2);
}
```
