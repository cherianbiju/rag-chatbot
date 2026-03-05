---
source_file: m8_bolt_threaded.js
category: mechanical
type: annotated_code
use_case: M8 metric bolt with realistic helical thread profile, hex head with chamfer, and unthreaded shank
related: keyway.md, mm2001-v2.md
---
# M8 Bolt with Threaded Section

## Description
Constructs a dimensionally accurate M8 metric bolt consisting of three parts: a hexagonal head with chamfered top and bottom edges, an unthreaded shank, and a threaded section created by revolving a zig-zag profile around the Z axis. Thread geometry follows ISO metric thread proportions with pitch-derived tooth depth.

## Keywords
M8, bolt, metric, thread, revolve, zig-zag, hex-head, drawPolysides, chamfer, drawCircle, fuse, inPlane, pitch, threadDepth, replicad, fastener, 3d-printing, ISO-metric

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| majorDiameter | 10 | mm | Thread major (outer) diameter |
| pitch | 1.25 | mm | Thread pitch (distance between teeth) |
| totalLength | 40 | mm | Total bolt length |
| threadLength | 30 | mm | Length of threaded section |
| headWrenchSize | 13 | mm | Hex head across-flats dimension |
| headHeight | 5.3 | mm | Height of hex head |
| headChamfer | 1 | mm | Chamfer on top and bottom edges of head |
| threadDepth | 0.6134 × pitch ≈ 0.767 | mm | ISO metric thread tooth depth |
| minorRadius | majorRadius − threadDepth | mm | Thread minor (root) radius |
| unthreadedLength | totalLength − threadLength = 10 | mm | Length of plain shank |
| tipChamfer | 0.8 | mm | Chamfer on bolt tip face |

## Code
```javascript
// FILE: m8_bolt_threaded.js
// DESCRIPTION: M8 metric bolt with realistic threaded section, hex head, chamfers, and unthreaded shank

const main = (replicad) => {
  const { draw, drawPolysides, drawCircle } = replicad;

  const majorDiameter = 10;
  const pitch = 1.25;
  const totalLength = 40;
  const threadLength = 30;
  const headWrenchSize = 13;
  const headHeight = 5.3;
  const headChamfer = 1;

  const majorRadius = majorDiameter / 2;
  const threadDepth = 0.6134 * pitch;
  const minorRadius = majorRadius - threadDepth;
  const unthreadedLength = totalLength - threadLength;
  const headRadius = (headWrenchSize / 2) / Math.cos(Math.PI / 6);

  // Hex Head
  const headShape = drawPolysides(headRadius, 6)
    .sketchOnPlane("XY").extrude(headHeight).translateZ(-headHeight);
  const chamferedHead = headShape.chamfer(headChamfer, (e) => e.inPlane("XY", 0));

  // Unthreaded Shank
  let shankPart = null;
  if (unthreadedLength > 0.1) {
    shankPart = drawCircle(majorRadius).sketchOnPlane("XY").extrude(unthreadedLength);
  }

  // Threaded Section (Revolve of zig-zag profile)
  let threadProfile = draw().movePointerTo([0, 0]).lineTo([minorRadius, 0]);
  const numTeeth = Math.floor(threadLength / pitch);
  for (let i = 0; i < numTeeth; i++) {
    threadProfile = threadProfile.line(majorRadius - minorRadius, pitch / 2);
    threadProfile = threadProfile.line(-(majorRadius - minorRadius), pitch / 2);
  }
  const totalProfileHeight = numTeeth * pitch;
  threadProfile = threadProfile.lineTo([0, totalProfileHeight]).close();
  const threadedPart = threadProfile.sketchOnPlane("XZ").revolve().translateZ(unthreadedLength);

  // Combine
  let bolt = chamferedHead;
  if (shankPart) bolt = bolt.fuse(shankPart);
  bolt = bolt.fuse(threadedPart);

  const tipZ = unthreadedLength + totalProfileHeight;
  return bolt.chamfer(0.8, (e) => e.inPlane("XY", tipZ));
};
```
