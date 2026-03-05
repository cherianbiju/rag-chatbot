---
source_file: garminBand.js
category: enclosure
type: annotated_code
use_case: watch band keeper, Garmin band, wearable, 3D printing
related: creditCardTray.md, birdhouse.md
---

# Garmin Watch Band Keeper

## Description
Parametric watch band keeper loop for a Garmin watch band. An inner rounded rectangle (the band cross-section plus tolerance) is extruded 2mm longer than the keeper width, then subtracted from an outer rounded rectangle extruded to exactly the keeper width. The result is a hollow loop that slips over the band. Designed for FDM 3D printing in PLA.

## Keywords
watch band, keeper, Garmin, sketchRoundedRectangle, extrude, cut, fillet, 3D printing, wearable, hollow loop, tolerance, band width, band thickness, PLA

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| bandWidth | 22.5 | mm | Width of watch band |
| keeperWidth | 6 | mm | Length of the keeper loop |
| bandThickness | 6.5 | mm | Thickness of band (3.25×2) |
| thickness | 1.5 | mm | Wall thickness of keeper |
| fillet | 0.5 | mm | Corner fillet for flex |
| oBw | 25.5 | mm | Outer body width (bandWidth + 2×thickness) |
| oBt | 9.5 | mm | Outer body thickness (bandThickness + 2×thickness) |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| sketchRoundedRectangle(w,h,r) | Creates rounded rectangle sketch |
| .extrude(h) | Extrudes to 3D solid |
| .cut(other) | Boolean subtract inner from outer to make loop |
| .fillet(r) | Rounds all edges for comfort and printing |

## Code
```javascript
const { sketchRoundedRectangle } = replicad;
function main() {
  const bandWidth=22.5, keeperWidth=6, bandThickness=3.25*2, thickness=1.5, fillet=0.5;
  let innerShape = sketchRoundedRectangle(bandWidth,bandThickness,fillet).extrude(keeperWidth+2);
  let oBw=bandWidth+2*thickness, oBt=bandThickness+2*thickness;
  let outerShape = sketchRoundedRectangle(oBw,oBt,fillet+thickness).extrude(keeperWidth)
    .cut(innerShape).fillet(0.5);
  return outerShape;
}
```
