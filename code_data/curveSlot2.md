---
source_file: curveSlot2.js
category: structural
type: annotated_code
use_case: slotted lever, mechanical linkage, cam follower slot
related: crankshaft.md, shaft_design.md, keyways.md
---

# Curve Slot 2

## Description
Slotted lever mechanism with a cylindrical axle hub, a complex plate profile, and a curved arc slot. The slot is created by intersecting a ring with a segment, rounding its ends, then offsetting inward to get a wall and extruding. Demonstrates advanced 2D boolean operations and arc-based slot creation.

## Keywords
slotted lever, curved slot, axle, keyway, ellipseTo, tangentArc, polarLineTo, drawCircle, drawRectangle, offset, intersect, fuse, cut, linkage, cam, mechanical

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| axleHoleRadius | 11 | mm | Radius of axle bore hole |
| axleRadius | 17.5 | mm | Outer radius of axle hub |
| keySlotHeight | 6 | mm | Height of keyway slot |
| keySlotWidth | 2.5 | mm | Width of keyway slot |
| axleWidth | 30 | mm | Axial width of axle hub |
| dist | 100 | mm | Distance from axle center to slot center |
| slotOuterRadius | 12 | mm | Outer radius of arc slot |
| slotAngle | 30 | deg | Angular span of arc slot |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawCircle(r) | Creates circular 2D sketch |
| .sketchOnPlane("XZ", offset) | Places sketch on XZ plane at Y offset |
| .extrude(h) | Extrudes to 3D solid |
| draw() | Starts freeform 2D drawing |
| .lineTo([x,y]) | Line to absolute point |
| .tangentArc(dx,dy) | Arc tangent to previous segment |
| .polarLineTo([r,angle]) | Line defined by polar coordinates |
| .ellipseTo([x,y],rx,ry) | Ellipse arc to point |
| .close() | Close sketch |
| .cut(other2D) | 2D boolean subtract |
| .intersect(other2D) | 2D boolean intersection |
| .fuse(other2D) | 2D boolean union |
| .offset(d) | Offsets 2D shape inward/outward |
| .translate([x,y]) | Moves 2D sketch |
| .fillet(r, edgeFinder) | Rounds 3D edges |
| .inPlane("XZ", y) | Finds edges in XZ plane at Y |

## Code
```javascript
const { draw, drawCircle, drawRectangle } = replicad;
const main = () => {
  let axleHoleRadius=11, axleRadius=17.5, keySlotHeight=6, keySlotWidth=2.5;
  let axleWidth=30, dist=100, slotOuterRadius=12, slotAngle=30/180*Math.PI;
  // compute slot geometry angles
  let dh=axleRadius-slotOuterRadius, minAngle=Math.asin(-dh/dist), maxAngle=minAngle+slotAngle;
  let startPoint=[Math.cos(minAngle)*(dist+slotOuterRadius),Math.sin(minAngle)*(dist+slotOuterRadius)];
  let endPoint=[Math.cos(maxAngle)*(dist+slotOuterRadius),Math.sin(maxAngle)*(dist+slotOuterRadius)];
  let axle = drawCircle(axleRadius).sketchOnPlane("XZ").extrude(axleWidth);
  let plate = draw().movePointerTo([0,-axleRadius+6])
    .lineTo([0,-axleRadius]).lineTo([100,-axleRadius]).lineTo([90,-axleRadius+6])
    .lineTo([38,-axleRadius+6]).tangentArc(-5,30).tangentArc(50,28)
    .polarLineTo([100,33.5]).ellipseTo([0,axleRadius],175,175)
    .lineTo([0,axleRadius-6]).ellipseTo([0,-axleRadius+6],11,11)
    .close().sketchOnPlane("XZ",3).extrude(14);
  // Create arc slot by ring+segment intersection
  let slotOuter=drawCircle(dist+slotOuterRadius), slotInner=drawCircle(dist-slotOuterRadius);
  let segment=draw().lineTo(startPoint).line(0,50).lineTo(endPoint).close();
  let slotSegment=slotOuter.cut(slotInner).intersect(segment)
    .fuse(drawCircle(slotOuterRadius).translate(startPoint))
    .fuse(drawCircle(slotOuterRadius).translate(endPoint));
  let slotSegmentOuter=slotSegment.cut(slotSegment.offset(-6)).sketchOnPlane("XZ",2).extrude(16);
  // Axle bore with keyway
  let axleHoleShape = drawCircle(axleHoleRadius)
    .fuse(drawRectangle(2*keySlotWidth,keySlotHeight).translate(-axleHoleRadius,0))
    .sketchOnPlane("XZ",-10);
  axle = axle.cut(axleHoleShape.extrude(50)).fuse(slotSegmentOuter).fuse(plate)
    .fillet(0.8,(e)=>e.inPlane("XZ",2)).fillet(0.8,(e)=>e.inPlane("XZ",18));
  return [{shape:axle, color:"steelblue"}];
};
```
