---
source_file: three_arm_rc.js
category: mechanical, hardware
type: annotated_code
use_case: servo arm, RC linkage, three-way bracket
related: lever_design.md, shaft_design.md
---

# Three-Arm RC Servo Lever

## Description
Designs a symmetric three-arm servo lever (horn) for RC applications by creating a single two-circle lever profile via tangent-line geometry, rotating it 120° to produce three arms, then applying a revolve-cut for the side profile and adding a central bore plus countersunk mounting holes. Demonstrates reusable inner functions, rotational symmetry, and multi-step boolean operations.

## Keywords
servo arm, RC lever, three-arm, fuse, rotate, bore, counterbore, fillet, revolve cut, tangent lines, circular arc, sketchCircle, makeCylinder, boolean cut, symmetry, mechanical linkage

## Parameters
| Variable | Value | Unit | Meaning                                            |
|----------|-------|------|----------------------------------------------------|
| r1       | 12.5  | mm   | Radius of inner hub circle of the lever            |
| r2       | 6     | mm   | Radius of outer tip circle of the lever            |
| d        | 35    | mm   | Distance between the two lever circle centers      |
| t        | 3     | mm   | Wall thickness for the leverHoles function         |
| h        | 22    | mm   | Lever extrusion height (thickness in Z)            |
| fl       | 22    | mm   | Fillet length applied to top/bottom Z edges        |

## Code
```javascript
// FILE: three_arm_rc.js
// Three-arm RC servo lever built from a tangent-line lever profile,
// rotated for symmetry, with bores and countersinks.

function main({ Sketcher, sketchCircle, makeCylinder }) {

  // --- Parameters ---
  let r1 = 12.5;  // inner hub radius (mm)
  let r2 = 6;     // tip circle radius (mm)
  let d  = 35;    // distance between circle centers (mm)
  let t  = 3;     // wall thickness for holes
  let h  = 22;    // lever height / extrusion depth (mm)
  let fl = 22;    // fillet radius on Z-direction edges

  // --- Lever function: solid two-circle body connected by tangent lines ---
  // Creates a "dog bone" / dumbbell shape between two circles.
  // The outline is computed using the sine of the angle between centers.
  function Lever(radius1, radius2, distance, leverHeight) {
    const sinus_angle = (radius1 - radius2) / distance;
    const angle = Math.asin(sinus_angle);

    const p1 = [radius1 * Math.sin(angle),  radius1 * Math.cos(angle)];
    const p2 = [distance + radius2 * Math.sin(angle),  radius2 * Math.cos(angle)];
    const p3 = [distance + radius2, 0];    // tip of outer arc
    const p4 = [distance + radius2 * Math.sin(angle), -radius2 * Math.cos(angle)];
    const p5 = [radius1 * Math.sin(angle), -radius1 * Math.cos(angle)];
    const p6 = [-radius1, 0];              // back of inner arc

    let sketchLever = new Sketcher("XY").movePointerTo(p1)
      .lineTo(p2)
      .threePointsArcTo(p4, p3)  // outer arc
      .lineTo(p5)
      .threePointsArcTo(p1, p6)  // inner arc
      .close();

    return sketchLever.extrude(leverHeight);
  }

  // --- leverHoles: Lever with bore holes at both ends ---
  function leverHoles(radius1, radius2, distance, leverHeight, wallThickness) {
    let leverBody = Lever(radius1, radius2, distance, leverHeight);
    let orig_hole = sketchCircle(radius1 - wallThickness).extrude(leverHeight + 10);
    let dist_hole = sketchCircle(radius2 - wallThickness).extrude(leverHeight + 10)
      .translate([distance, 0, 0]);
    return leverBody.cut(orig_hole).cut(dist_hole);
  }

  // --- Build three arms at 0°, 120°, 240° ---
  let arm1 = Lever(r1, r2, d, h);
  let arm2 = Lever(r1, r2, d, h).rotate(120, [0,0,0], [0,0,1]);
  let arm3 = Lever(r1, r2, d, h).rotate(240, [0,0,0], [0,0,1]);
  let threeArm = arm1.fuse(arm2).fuse(arm3)
    .fillet(fl, (e) => e.inDirection("Z"));  // fillet all vertical edges

  // --- Side profile revolve cut: shapes the top/bottom face ---
  let side = new Sketcher("XZ").movePointerTo([41, 6])
    .lineTo([50, 6]).lineTo([50, 30]).lineTo([0, 30])
    .lineTo([0, 22]).lineTo([11, 22])
    .lineTo([11, 6 + (30 * Math.sin(22 * Math.PI / 180))])
    .close();

  // Rotate cutter 60° to avoid intersecting the first arm
  let sideCutter = side.revolve().rotate(60, [0,0,0], [0,0,1]);
  threeArm = threeArm.cut(sideCutter, false, false);

  // Fillet the step edges created by the side cut
  threeArm = threeArm.fillet(1, (e) => e.inBox([50,50,2], [-50,-50,20]));

  // --- Central shaft bore ---
  let bigBore = sketchCircle(8).extrude(40).translate([0, 0, -10]);
  threeArm = threeArm.cut(bigBore);

  // --- Mounting holes at r=35mm, spaced 120° apart ---
  // Small bore (M4.5 clearance, through hole)
  let smallBore1 = makeCylinder(4.5/2, 22, [0,0,0], [0,0,1]).translate([35,0,-5]);
  let smallBore2 = smallBore1.clone().rotate(120, [0,0,0], [0,0,1]);
  let smallBore3 = smallBore1.clone().rotate(240, [0,0,0], [0,0,1]);
  threeArm = threeArm.cut(smallBore1).cut(smallBore2).cut(smallBore3);

  // Counterbore (M6 head recess, from top face)
  let counterBore1 = makeCylinder(6/2, 22, [0,0,0], [0,0,1]).translate([35,0,4]);
  let counterBore2 = counterBore1.clone().rotate(120, [0,0,0], [0,0,1]);
  let counterBore3 = counterBore1.clone().rotate(240, [0,0,0], [0,0,1]);
  threeArm = threeArm.cut(counterBore1).cut(counterBore2).cut(counterBore3);

  return [{ shape: threeArm, color: "steelblue" }];
}
```
