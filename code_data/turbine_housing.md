---
source_file: turbine_housing.js
category: turbocharger
type: annotated_code
use_case: directs exhaust gas through turbine wheel to extract energy and drive compressor
related: compressor_wheel.md, turbo_shaft.md
---
# Turbocharger Turbine Housing

## Description
A sand-cast iron turbine housing with a volute scroll inlet, axial turbine wheel bore, V-band outlet flange, and oil drain port. The scroll accelerates exhaust gas tangentially into the turbine wheel.

## Keywords
turbine housing, volute, scroll, turbocharger, exhaust, turbine wheel, V-band, wastegate, cast iron, revolve, draw, fuse, cut, cylinder, sweep, exhaust housing

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| SCROLL_OUTER_R | 90 | mm | scroll outer radius |
| SCROLL_INNER_R | 45 | mm | turbine wheel chamber radius |
| HOUSING_HEIGHT | 70 | mm | axial housing height |
| INLET_R | 28 | mm | exhaust inlet bore radius |
| OUTLET_R | 32 | mm | outlet bore radius |
| VBAND_R | 40 | mm | V-band flange radius |
| VBAND_THICK | 12 | mm | V-band flange thickness |
| WALL_THICK | 8 | mm | housing wall thickness |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    makeCylinder,
    drawCircle,
  } = replicad;

  const SCROLL_OUTER_R = 90;
  const SCROLL_INNER_R = 45;
  const HOUSING_HEIGHT = 70;
  const INLET_R        = 28;
  const OUTLET_R       = 32;
  const VBAND_R        = 40;
  const VBAND_THICK    = 12;
  const WALL_THICK     = 8;

  // Main scroll body — annular ring
  const scrollProfile = draw([SCROLL_INNER_R, 0])
    .hLine(SCROLL_OUTER_R - SCROLL_INNER_R)
    .vLine(HOUSING_HEIGHT)
    .hLine(-(SCROLL_OUTER_R - SCROLL_INNER_R))
    .close();
  let housing = scrollProfile.sketchOnPlane("XZ").revolve();

  // Turbine wheel chamber bore
  const chamberBore = makeCylinder(SCROLL_INNER_R - WALL_THICK, HOUSING_HEIGHT + 2, [0, 0, -1], [0, 0, 1]);
  housing = housing.cut(chamberBore);

  // Exhaust inlet port tangential to scroll
  const inlet = makeCylinder(INLET_R, SCROLL_OUTER_R - SCROLL_INNER_R + 20, [SCROLL_OUTER_R - 10, 0, HOUSING_HEIGHT * 0.6], [1, 0, 0]);
  housing = housing.fuse(inlet);
  const inletBore = makeCylinder(INLET_R - WALL_THICK, SCROLL_OUTER_R + 10, [SCROLL_INNER_R - 5, 0, HOUSING_HEIGHT * 0.6], [1, 0, 0]);
  housing = housing.cut(inletBore);

  // Outlet V-band flange on bottom
  const vbandProfile = draw([OUTLET_R, -VBAND_THICK])
    .hLine(VBAND_R - OUTLET_R)
    .vLine(VBAND_THICK)
    .hLine(-(VBAND_R - OUTLET_R))
    .close();
  const vband = vbandProfile.sketchOnPlane("XZ").revolve();
  housing = housing.fuse(vband);

  // Outlet bore
  const outletBore = makeCylinder(OUTLET_R - WALL_THICK, VBAND_THICK + 4, [0, 0, -VBAND_THICK - 2], [0, 0, 1]);
  housing = housing.cut(outletBore);

  return { shape: housing, name: "Turbine Housing", color: "dimgrey" };
};
```
