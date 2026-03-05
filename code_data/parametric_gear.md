---
source_file: parametric_gear.js
category: mechanical
type: annotated_code
use_case: parametric involute-approximated spur gear with configurable tooth count, diameter, bore, and pressure angle
related: keyway.md, lever.md, m8_bolt_threaded.md
---
# Parametric Spur Gear

## Description
Generates a spur gear by computing involute-approximated tooth geometry analytically from standard gear parameters (module, pitch diameter, root diameter, pressure angle). Each tooth is drawn as four straight lines between root and tip radii at computed angular offsets, producing a realistic gear profile loop that is extruded and optionally bored. Suitable for functional 3D-printed gears or visual assemblies.

## Keywords
gear, spur-gear, involute, parametric, module, pitchDiameter, rootDiameter, pressureAngle, teethCount, drawCircle, extrude, cut, bore, mechanical, power-transmission, replicad, 3d-printing, analytic-geometry

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| teethCount | 87 | — | Number of gear teeth |
| outerDiameter | 70 | mm | Gear tip circle diameter |
| gearThickness | 7 | mm | Extrusion thickness of the gear |
| boreDiameter | 8 | mm | Centre bore diameter (0 = no bore) |
| pressureAngle | 20 | ° | Standard involute pressure angle |
| module | outerDiameter / (teethCount + 2) | mm | Gear module (tooth size unit) |
| pitchDiameter | module × teethCount | mm | Pitch circle diameter |
| rootDiameter | pitchDiameter − 2.5 × module | mm | Root circle diameter |
| angleStep | 360 / teethCount | ° | Angular pitch between teeth |
| angleOffsetTip | computed from pressureAngle | ° | Angular offset at tip due to pressure angle |
| angleOffsetRoot | computed from pressureAngle | ° | Angular offset at root due to pressure angle |

## Code
```javascript
// FILE: parametric_gear.js
// DESCRIPTION: Parametric spur gear with configurable teeth, diameter, bore, and pressure angle

const main = (replicad) => {
  const { draw, drawCircle } = replicad;

  const teethCount = 87;
  const outerDiameter = 70;
  const gearThickness = 7;
  const boreDiameter = 8;
  const pressureAngle = 20;
  
  const module = outerDiameter / (teethCount + 2);
  const pitchDiameter = module * teethCount;
  const rootDiameter = pitchDiameter - (2.5 * module);
  
  const radiusOuter = outerDiameter / 2;
  const radiusRoot = rootDiameter / 2;
  const radiusPitch = pitchDiameter / 2;
  
  const toRad = Math.PI / 180;
  const tanPressure = Math.tan(pressureAngle * toRad);
  
  const angleStep = 360 / teethCount;
  const angleToothHalfAtPitch = angleStep / 4;
  
  const angleOffsetTip = ((radiusOuter - radiusPitch) * tanPressure) / radiusOuter / toRad;
  const angleOffsetRoot = ((radiusPitch - radiusRoot) * tanPressure) / radiusRoot / toRad;

  const angleTipHalf = Math.max(0.1, angleToothHalfAtPitch - angleOffsetTip);
  const angleRootHalf = angleToothHalfAtPitch + angleOffsetRoot;

  let gearPen = draw();
  
  for (let i = 0; i < teethCount; i++) {
    const currentAngle = i * angleStep;
    const a1 = currentAngle - angleRootHalf;
    const a2 = currentAngle - angleTipHalf;
    const a3 = currentAngle + angleTipHalf;
    const a4 = currentAngle + angleRootHalf;
    const r1 = a1 * toRad; const r2 = a2 * toRad;
    const r3 = a3 * toRad; const r4 = a4 * toRad;
    const p1 = [radiusRoot * Math.cos(r1), radiusRoot * Math.sin(r1)];
    const p2 = [radiusOuter * Math.cos(r2), radiusOuter * Math.sin(r2)];
    const p3 = [radiusOuter * Math.cos(r3), radiusOuter * Math.sin(r3)];
    const p4 = [radiusRoot * Math.cos(r4), radiusRoot * Math.sin(r4)];
    if (i === 0) { gearPen.movePointerTo(p1); } else { gearPen.lineTo(p1); }
    gearPen.lineTo(p2); gearPen.lineTo(p3); gearPen.lineTo(p4);
  }
  
  const gearProfile = gearPen.close();
  let gearShape = gearProfile.sketchOnPlane().extrude(gearThickness);
  
  if (boreDiameter > 0) {
    const hole = drawCircle(boreDiameter / 2).sketchOnPlane().extrude(gearThickness);
    gearShape = gearShape.cut(hole);
  }
  
  return gearShape;
};
```
