---
source_file: boolean.js
category: replicad_example
type: annotated_code
use_case: boolean operations demo, fuse cut intersect, learning tool
related: addthickness.md, birdhouse.md
---

# Boolean Operations

## Description
Demonstration file showing the three core boolean operations in replicad — fuse (union), cut (subtract), and intersect — using two overlapping spheres. Also shows makeCompound for grouping shapes without merging them. Useful as a reference for understanding boolean logic in CAD.

## Keywords
boolean, fuse, cut, intersect, makeCompound, makeSphere, union, subtract, compound, boolean operations, replicad basics, overlap

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| sphere radius | 20 | mm | Radius of both spheres |
| translate X | 45 | mm | X offset of sphere1 from origin |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| makeSphere(r) | Creates a solid sphere of radius r |
| .translate([x,y,z]) | Moves shape by given vector |
| makeCompound([s1,s2]) | Groups shapes together without merging |
| .fuse(other) | Boolean union — combines two shapes |
| .cut(other) | Boolean subtract — removes overlapping volume |
| .intersect(other) | Boolean intersect — keeps only overlapping volume |

## Code
```javascript
const main = ({ makeSphere, makeCompound }, {}) => {
  let sphere1 = makeSphere(20).translate([45,0,0]);
  let sphere2 = makeSphere(20);
  let compound = makeCompound([sphere1, sphere2]);
  // return sphere1.fuse(sphere2)    // union
  // return sphere1.cut(sphere2)     // subtract
  // return sphere1.intersect(sphere2) // intersection
  return compound;
};
```
