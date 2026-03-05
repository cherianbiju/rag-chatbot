---
source_file: icosahedron-replicad.js
category: geometry
type: annotated_code
use_case: procedural icosahedron and geodesic sphere generation for display or Boolean operations
related: inlist-test.md
---
# Icosahedron in Replicad

## Description
Generates a regular icosahedron by computing the 12 vertices from four overlapping golden-ratio rectangles, projecting them onto a sphere surface, and assembling 20 triangular polygon faces into a closed solid. Also demonstrates combining it with a box and sphere for Boolean difference operations.

## Keywords
icosahedron, geodesic, sphere-projection, golden-ratio, makePolygon, makeSolid, makeSphere, triangular-faces, procedural-geometry, replicad, 3d-printing, mathematical-solid, boolean, scale

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| radius (makeIcosahedron) | 2.0 | — | Base radius before scaling |
| scale | 50 | — | Scale factor applied to icosahedron (final radius ~100) |
| box size | 200×200×200 | mm | Box used for Boolean demonstration |
| sphere radius | 100 | mm | Sphere used for Boolean demonstration |
| sphere translate | [50,30,20] | mm | Sphere position offset |

## Code
```javascript
function projectOnSphere(radius, vertex) {
  let x = vertex[0];
  let y = vertex[1];
  let z = vertex[2];
  let currentRadius = Math.sqrt(
    Math.pow(x, 2) + Math.pow(y, 2) + Math.pow(z, 2)
  );
  let scale = radius / currentRadius;
  let scaledVertex = [scale * x, scale * y, scale * z];
  return scaledVertex;
}

const icosahedronFaces = (radius) => {
  let golden = (1 + Math.sqrt(5)) / 2;

  let v = [
    projectOnSphere(radius, [-1, golden, 0]),
    projectOnSphere(radius, [1, golden, 0]),
    projectOnSphere(radius, [-1, -golden, 0]),
    projectOnSphere(radius, [1, -golden, 0]),
    projectOnSphere(radius, [0, -1, golden]),
    projectOnSphere(radius, [0, 1, golden]),
    projectOnSphere(radius, [0, -1, -golden]),
    projectOnSphere(radius, [0, 1, -golden]),
    projectOnSphere(radius, [golden, 0, -1]),
    projectOnSphere(radius, [golden, 0, 1]),
    projectOnSphere(radius, [-golden, 0, -1]),
    projectOnSphere(radius, [-golden, 0, 1]),
  ];

  return [
    [v[0], v[11], v[5]],  [v[0], v[5], v[1]],
    [v[0], v[10], v[11]], [v[0], v[7], v[10]],
    [v[5], v[11], v[4]],  [v[4], v[9], v[5]],
    [v[3], v[9], v[4]],   [v[3], v[8], v[9]],
    [v[3], v[6], v[8]],   [v[3], v[2], v[6]],
    [v[6], v[2], v[10]],  [v[10], v[7], v[6]],
    [v[8], v[6], v[7]],   [v[0], v[1], v[7]],
    [v[1], v[5], v[9]],   [v[11], v[10], v[2]],
    [v[7], v[1], v[8]],   [v[3], v[4], v[2]],
    [v[2], v[4], v[11]],  [v[9], v[8], v[1]],
  ];
};

const main = (
  { makeSolid, sketchRoundedRectangle, makeSphere, makePolygon },
  {}
) => {
  function makeIcosahedron(radius) {
    const faces = icosahedronFaces(radius).map((f) => makePolygon(f));
    return makeSolid(faces);
  }

  const icosahedron = makeIcosahedron(2.0).scale(50);
  const box = sketchRoundedRectangle(200, 200)
    .extrude(200)
    .translate([100, 100, 0]);
  const sphere = makeSphere(100).translate([50, 30, 20]);

  let shapes = [
  {shape: icosahedron, name: "icosehadron", color: "steelblue"},
  {shape: box,         name: "box",         color: "yellow"},
  {shape: sphere,      name: "sphere",      color: "grey"}
  ]

  return shapes;
};
```
