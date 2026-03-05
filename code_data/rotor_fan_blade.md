---
source_file: rotor_fan_blade.js
category: aerospace, fluid_mechanics
type: annotated_code
use_case: propeller design, fan blade modeling, UAV components
related: shaft_design.md, airfoil_theory.md
---

# Parametric Propeller / Fan Blade

## Description
Generates a parametric propeller or fan blade with configurable blade count, chord length, twist angle, and airfoil profile using multi-section lofting. The blade geometry varies from root to tip using interpolated airfoil cross-sections, making it suitable for UAV propellers, cooling fans, or wind turbine prototyping.

## Keywords
propeller, fan blade, airfoil, chord length, blade twist, pitch angle, loft, hub, shaft bore, rotor, NACA profile, camber, blade count, parametric, extrude

## Parameters
| Variable           | Value | Unit  | Meaning                                      |
|--------------------|-------|-------|----------------------------------------------|
| propDiameterInch   | 15    | inch  | Total propeller diameter                     |
| hubDiameter        | 30    | mm    | Outer diameter of the central hub            |
| hubHeight          | 20    | mm    | Axial height of the hub cylinder             |
| shaftDiameter      | 8     | mm    | Bore diameter for the drive shaft            |
| bladeCount         | 2     | —     | Number of blades (equally spaced)            |
| rootChord          | 45    | mm    | Blade chord width at root (hub side)         |
| tipChord           | 15    | mm    | Blade chord width at tip (outer edge)        |
| rootTwist          | 35    | deg   | Blade pitch angle at root                    |
| tipTwist           | 10    | deg   | Blade pitch angle at tip                     |
| maxBladeThickness  | 6     | mm    | Maximum airfoil thickness at root            |

## Code
```javascript
// FILE: rotor_fan_blade.js
// Parametric propeller/fan blade with configurable blade count, chord, twist, and hub

const main = (replicad) => {
  const { drawCircle, drawPointsInterpolation } = replicad;

  // --- Configuration ---
  const propDiameterInch = 15;       // total prop diameter in inches
  const hubDiameter = 30;            // hub outer diameter (mm)
  const hubHeight = 20;              // hub axial length (mm)
  const shaftDiameter = 8;           // shaft bore diameter (mm)
  const bladeCount = 2;              // number of blades
  const rootChord = 45;              // chord at root (mm)
  const tipChord = 15;               // chord at tip (mm)
  const rootTwist = 35;              // pitch angle at root (degrees)
  const tipTwist = 10;               // pitch angle at tip (degrees)
  const maxBladeThickness = 6;       // max airfoil thickness at root (mm)

  // --- Derived geometry ---
  const propRadius = (propDiameterInch * 25.4) / 2; // convert inches → mm
  const hubRadius = hubDiameter / 2;
  const shaftRadius = shaftDiameter / 2;
  const bladeStart = hubRadius - 2;  // blade starts just inside hub edge
  const bladeEnd = propRadius;
  const bladeLength = bladeEnd - bladeStart;

  // --- Airfoil cross-section generator ---
  // Returns scaled upper + lower surface points for a simplified cambered airfoil.
  // x is normalized 0→1 along chord, y is normalized thickness offset.
  const getAirfoilPoints = (chord, thickness) => {
    const upper = [[0,0],[0.05,0.4],[0.1,0.6],[0.2,0.8],[0.3,0.9],[0.4,0.85],[0.6,0.6],[0.8,0.3],[1,0]];
    const lower = [[1,0],[0.8,-0.1],[0.6,-0.2],[0.4,-0.25],[0.3,-0.25],[0.2,-0.2],[0.1,-0.15],[0.05,-0.1],[0,0]];
    // Scale to actual chord and thickness; offset so quarter-chord is at origin
    const scalePoint = ([x, y]) => [x * chord - (chord * 0.25), y * thickness];
    return [...upper.map(scalePoint), ...lower.map(scalePoint)];
  };

  // --- Build one blade section at a given radial fraction (0=root, 1=tip) ---
  const createSection = (fraction) => {
    const radialPosition = bladeStart + (fraction * bladeLength);
    const currentChord = rootChord - (fraction * (rootChord - tipChord));         // linear taper
    const currentTwist = rootTwist - (fraction * (rootTwist - tipTwist));         // linear twist
    const currentThickness = maxBladeThickness * (1 - (fraction * 0.5));           // thinning toward tip
    const points = getAirfoilPoints(currentChord, currentThickness);
    return drawPointsInterpolation(points, { closed: true })
      .rotate(currentTwist)                          // apply twist angle
      .sketchOnPlane("XY", radialPosition);          // position along span axis
  };

  // --- Loft through radial stations to form one blade ---
  const stations = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const sketches = stations.map(createSection);
  let singleBlade = sketches[0].loftWith(sketches.slice(1));
  singleBlade = singleBlade.rotate(90, [0,0,0], [0,1,0]); // orient blade radially

  // --- Build hub cylinder ---
  const hub = drawCircle(hubRadius)
    .sketchOnPlane()
    .extrude(hubHeight)
    .translateZ(-hubHeight / 2);

  // --- Attach all blades to hub, evenly spaced ---
  let propeller = hub;
  for (let i = 0; i < bladeCount; i++) {
    const angle = (360 / bladeCount) * i;
    const rotatedBlade = singleBlade.clone().rotate(angle, [0,0,0], [0,0,1]);
    propeller = propeller.fuse(rotatedBlade);
  }

  // --- Cut shaft bore through hub center ---
  const shaftHole = drawCircle(shaftRadius)
    .sketchOnPlane()
    .extrude(hubHeight * 2)
    .translateZ(-hubHeight);

  return propeller.cut(shaftHole);
};
```
