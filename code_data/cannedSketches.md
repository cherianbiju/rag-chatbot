---
source_file: cannedSketches.js
category: replicad_example
type: annotated_code
use_case: sketch primitives reference, learning replicad API
related: arc_ellipse.md, addthickness.md
---

# Canned Sketches

## Description
Reference file demonstrating all built-in replicad sketch primitives — circle, ellipse, rectangle, rounded rectangle, polysides, and parametric function sketches. All shapes are fused together into one combined solid. Ideal as a quick reference for available sketch functions.

## Keywords
sketchCircle, sketchEllipse, sketchRectangle, sketchRoundedRectangle, sketchPolysides, polysideInnerRadius, sketchParametricFunction, Plane, extrude, fuse, cut, primitives, reference

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| height | 10 | mm | Extrusion height for shapes |
| radius | 20 | mm | Base radius for shapes |
| fillet | 2 | mm | Corner fillet for rounded rectangle |
| sides | 6 | - | Number of sides for polygon |
| sagitta | -1 | mm | Sagitta (bulge) for polysides |
| thickness | 1 | mm | Wall thickness for hollow polygon |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| sketchCircle(r, options) | Creates circular sketch at position/plane |
| sketchEllipse(rx,ry, options) | Creates elliptical sketch |
| sketchRectangle(w,h) | Creates rectangular sketch |
| sketchRoundedRectangle(w,h,r) | Creates rectangle with rounded corners |
| sketchPolysides(r,n,sagitta) | Creates n-sided polygon with optional sagitta |
| polysideInnerRadius(r,n,s) | Computes inner radius of polygon |
| new Plane([x,y,z]) | Creates a plane object at given position |
| .extrude(h) | Extrudes sketch to height h |
| .fuse(other) | Boolean union |
| .cut(other) | Boolean subtract |

## Code
```javascript
function main({ sketchCircle, sketchEllipse, sketchRectangle, sketchRoundedRectangle,
                sketchPolysides, polysideInnerRadius, Plane }) {
  let height=10, radius=20, fillet=2, sides=6, sagitta=-1, thickness=1;
  let circle = sketchCircle(radius, new Plane([0,0,height])).extrude(height);
  let ellipse = sketchEllipse(1.5*radius, radius/2, {plane:"YZ",origin:[0,0,height/2]}).extrude(height);
  let box = sketchRectangle(radius, radius*2).extrude(30);
  let box2 = sketchRoundedRectangle(radius*2, radius*3, fillet).extrude(2);
  let innerRadius = polysideInnerRadius(radius, sides, sagitta);
  let poly = sketchPolysides(radius, sides, sagitta).extrude(height*5);
  let hole = sketchCircle(innerRadius-thickness, {plane:"XY",origin:[0,0,4*height]}).extrude(20);
  circle = circle.fuse(ellipse).fuse(box).fuse(box2).fuse(poly).cut(hole);
  return circle;
}
```
