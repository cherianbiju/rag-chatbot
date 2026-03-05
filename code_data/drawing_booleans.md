---
source_file: drawing_booleans.js
category: replicad_example
type: annotated_code
use_case: 2D boolean operations reference, cut fuse intersect on flat sketches
related: boolean.md, arc_ellipse.md
---

# Drawing Booleans

## Description
Minimal demonstration of 2D boolean operations in replicad using a D-shaped profile (horizontal lines + halfEllipse) and a circle. The circle is cut from the D-shape. Commented lines show the same shape can be fused or intersected instead. Ideal quick reference for 2D sketch booleans.

## Keywords
draw, drawCircle, cut, fuse, intersect, 2D boolean, halfEllipse, hLine, translate, flat sketch, 2D profile, cutout, boolean operations, reference

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| hLine length | 25 | mm | Width of flat base of D-shape |
| halfEllipse dy | 40 | mm | Height of ellipse end of D-shape |
| halfEllipse sagitta | 5 | mm | Bulge amount of ellipse |
| circle radius | 8 | mm | Radius of cutout circle |
| circle offset | [20,10] | mm | Translation of circle from origin |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| draw() | Starts a 2D freeform drawing at origin |
| .hLine(d) | Draws horizontal line of length d |
| .halfEllipse(dx, dy, sagitta) | Draws half-ellipse arc |
| .close() | Closes the 2D path |
| drawCircle(r) | Creates a circular 2D sketch |
| .translate([x,y]) | Moves 2D sketch to given position |
| .cut(other2D) | 2D boolean subtract |
| .fuse(other2D) | 2D boolean union (commented out) |
| .intersect(other2D) | 2D boolean intersection (commented out) |

## Code
```javascript
const { draw, drawCircle } = replicad;
const main = () => {
  let drawing1 = draw().hLine(25).halfEllipse(0,40,5).hLine(-25).close();
  let drawing2 = drawCircle(8).translate([20,10]);
  drawing1 = drawing1.cut(drawing2);
  // drawing1 = drawing1.fuse(drawing2)
  // drawing1 = drawing1.intersect(drawing2)
  return drawing1;
};
```
