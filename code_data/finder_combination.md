---
source_file: finder_combination.js
category: replicad_example
type: annotated_code
use_case: multi-radius fillet, combineFinderFilters, EdgeFinder reference
related: edgefinder.md, edges-inlist.md
---

# Finder Combination

## Description
Demonstrates combineFinderFilters to apply two different fillet radii in a single fillet call — 10mm on vertical (Z-direction) edges and 9.99999mm on horizontal (XY-parallel) edges. The slightly different radii avoid kernel conflicts. Clean example of combining multiple EdgeFinder rules.

## Keywords
combineFinderFilters, EdgeFinder, fillet, inDirection, parallelTo, drawRoundedRectangle, sketchOnPlane, extrude, multi-radius fillet, combined filters, reference, Z edges, XY parallel

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| rectangle width | 30 | mm | Base rectangle width |
| rectangle height | 50 | mm | Base rectangle height |
| extrude height | 20 | mm | Extrusion height |
| vertical fillet | 10 | mm | Fillet on Z-direction edges |
| horizontal fillet | 9.99999 | mm | Fillet on XY-parallel edges (slightly less to avoid conflict) |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| combineFinderFilters([...]) | Takes array of {filter, radius} objects and returns combined filter |
| new EdgeFinder().inDirection("Z") | Selects vertical edges |
| new EdgeFinder().parallelTo("XY") | Selects edges parallel to XY plane |
| drawRoundedRectangle(w,h) | Creates rounded rectangle sketch |
| .sketchOnPlane() | Places on default plane |
| .extrude(h) | Extrudes to 3D |
| .fillet(combinedFilters) | Applies all filters with their respective radii |

## Code
```javascript
const { drawRoundedRectangle, EdgeFinder, combineFinderFilters } = replicad;
const main = () => {
  const [filters] = combineFinderFilters([
    { filter: new EdgeFinder().inDirection("Z"), radius: 10 },
    { filter: new EdgeFinder().parallelTo("XY"), radius: 9.99999 }
  ]);
  return drawRoundedRectangle(30,50).sketchOnPlane().extrude(20).fillet(filters);
};
```
