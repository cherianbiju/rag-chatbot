---
source_file: one-world-piramids-v2.js
category: architecture
type: annotated_code
use_case: One World Trade Center tower approximation — square-to-octagon transition using pyramid cutouts, tapering spire
related: one-world-tc.md, loft-examples.md, lofts-failed-antiprism.md
---
# One World Trade Center — Pyramid Cutout Method

## Description
Models the One World Trade Center tower by cutting four corner pyramids from a square prism to create the octagonal cross-section transition. The tower base is a plain square extrusion, the upper section is the prism minus four rotated pyramids, and a tapering cylinder and spike are added on top. Uses the `extrusionProfile: linear, endFactor` option to create tapered (pyramid) extrusions.

## Keywords
One-World-Trade-Center, tower, architecture, pyramid, extrusionProfile, endFactor, taper, makeCylinder, drawCircle, drawRectangle, fuse, cut, rotate, translate, scale, replicad, 3d-printing, landmark

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| scale | 1/10 | — | Converts feet to model units (~30mm per foot) |
| baseLength | 200 × scale = 20 | mm | Square base side length |
| topLength | √2 × baseLength/2 | mm | Width of corner pyramid base |
| totalHeight | 1368 × scale = 136.8 | mm | Total tower height |
| baseHeight | 196.85 × scale ≈ 19.7 | mm | Height of straight square base |
| heightTop | totalHeight − baseHeight | mm | Height of tapered upper section |
| cylinderHeight | 50 × scale = 5 | mm | Height of top cylinder platform |
| spikeHeight | 250 × scale = 25 | mm | Height of tapering spike |
| spikeBaseR | 10 × scale = 1 | mm | Base radius of spike |
| endFactor | 0.01 | — | Near-zero tip for pyramid and spike extrusions |
| pyramid rotation | 45° | ° | Rotate pyramid so corners align with tower edges |

## Code
```javascript
const {draw,drawRectangle,drawCircle,makeCylinder} = replicad

function main()
{
let scale = 1/10
let baseLength = 200*scale;
let topLength = Math.sqrt((2*Math.pow((baseLength/2),2)))
let totalHeight = 1368*scale
let baseHeight = 196.85*scale
let heightTop = (totalHeight-baseHeight)
let cylinderHeight = 50*scale
let spikeHeight = 250*scale
let spikeBaseR = 10*scale

// create 4 pyramids to cut from square prism
let pMid1 = drawRectangle(topLength,topLength).sketchOnPlane("XY")
.extrude(heightTop,{extrusionProfile: {profile: "linear", endFactor: 0.01}})
.rotate(45,[0,0]).translate(-baseLength/2,-baseLength/2)
let pMid2 = pMid1.clone().translate(baseLength,0)
let pMid3 = pMid1.clone().translate(0,baseLength)
let pMid4 = pMid1.clone().translate(baseLength,baseLength)

let baseProfile = drawRectangle(baseLength,baseLength).sketchOnPlane("XY")
let towerTopPrism = baseProfile.clone().extrude(heightTop)
let towerTop = towerTopPrism.clone().cut(pMid1).cut(pMid2).cut(pMid3).cut(pMid4)
towerTop = towerTop.rotate(180,[0,0,0],[0,1,0]).translate(0,0,totalHeight)
let towerBase = baseProfile.extrude(baseHeight)
let tower = towerBase.fuse(towerTop)
let topCylinder = makeCylinder(topLength/1.8,5).translate(0,0,totalHeight)
tower = tower.fuse(topCylinder);
let spike = drawCircle(spikeBaseR).sketchOnPlane("XY")
spike = spike.extrude(spikeHeight,{extrusionProfile: {profile: "linear", endFactor: 0.01}})
.translate(0,0,totalHeight+cylinderHeight)
tower = tower.fuse(spike).translate(40,0,0)

return [
{shape: pMid1, color: "yellow"},
{shape: pMid2, color: "yellow"},
{shape: pMid3, color: "yellow"},
{shape: pMid4, color: "yellow"},
{shape: towerTopPrism, color: "darkred"},
{shape: tower}
]
}
```
