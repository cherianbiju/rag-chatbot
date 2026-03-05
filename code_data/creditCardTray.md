---
source_file: creditCardTray.js
category: enclosure
type: annotated_code
use_case: credit card holder, tray design, everyday object
related: birdhouse.md, boolean.md
---

# Credit Card Tray

## Description
Parametric credit card tray designed to hold standard ISO/IEC 7810 ID-1 credit cards with tolerance. Features a rounded rectangular body with a card slot cavity, a cylindrical finger cutout for easy card removal, and two tape holes on the sides. Demonstrates practical tolerance-based design.

## Keywords
credit card tray, drawRoundedRectangle, makeCylinder, makeBaseBox, cut, translate, rotate, tolerance, enclosure, tray, holder, everyday object, shell, ISO 7810

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| cardLength | 85.60 | mm | Standard credit card length |
| cardWidth | 53.98 | mm | Standard credit card width |
| cardThickness | 0.9 | mm | Standard credit card thickness |
| cardRadius | 3.18 | mm | Corner radius of credit card |
| tolerance | 0.5 | mm | Assembly clearance added to card dims |
| wallThickness | 2.0 | mm | Tray wall thickness |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawRoundedRectangle(w,h,r) | Creates rounded rectangle sketch |
| .sketchOnPlane("XY") | Places sketch on XY plane |
| .extrude(h) | Extrudes to 3D solid |
| .translate(x,y,z) | Moves shape to position |
| makeCylinder(r,h,axis) | Creates cylinder along given axis |
| .rotate(angle,origin,axis) | Rotates shape around axis |
| makeBaseBox(l,w,h) | Creates box from origin |
| .cut(other) | Boolean subtract |

## Code
```javascript
const { draw, drawRoundedRectangle, makeBaseBox, makeCylinder } = replicad;
const main = () => {
  const cardLength=85.60, cardWidth=53.98, cardThickness=0.9, cardRadius=3.18, tolerance=0.5;
  let holderLength=cardLength+tolerance, holderWidth=cardWidth+tolerance;
  let holderRadius=cardRadius+(tolerance/2), wallThickness=2.0;
  let bodyLength=holderLength+2*wallThickness, bodyWidth=holderWidth+2*wallThickness;
  let bodyRadius=holderRadius+wallThickness, bodyHeight=4*cardThickness;
  let fingerHole = makeCylinder(20,30,"Z").rotate(90,[0,0,0],[1,0,0]).translate(0,-27,20+1.0);
  let creditCard = drawRoundedRectangle(holderLength,holderWidth,holderRadius)
    .sketchOnPlane("XY").extrude(bodyHeight).translate(0,0,1.0);
  let holderBody = drawRoundedRectangle(bodyLength,bodyWidth,bodyRadius)
    .sketchOnPlane("XY").extrude(bodyHeight);
  let tapeHoleL = makeBaseBox(10,40,0.5).translate(30,0,0);
  let tapeHoleR = makeBaseBox(10,40,0.5).translate(-30,0,0);
  holderBody = holderBody.cut(creditCard).cut(fingerHole).cut(tapeHoleR).cut(tapeHoleL);
  return holderBody;
};
```
