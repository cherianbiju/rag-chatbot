---
source_file: flange_assembly_telescoping.js
category: structural
type: annotated_code
use_case: telescoping assembly, rack and pinion drive, flanged tubes, industrial mechanism
related: flange_with_lobes.md, pipe_fittings.md, shaft_design.md
---

# Flange Assembly Telescoping

## Description
Three-stage telescoping tube assembly with flanged bottom and top ends, rack and pinion drive mechanism, crank arm, and crank handle. Each stage is a hollow cylinder with collars. Stage 2 and 3 are lifted by extensionFactor×collapsedLength. The rack is a rectangular extrusion attached alongside each tube; the pinion is a polygon gear (drawPolysides) on the YZ plane.

## Keywords
telescoping, flange, rack and pinion, hollow tube, drawCircle, drawRectangle, drawPolysides, cut, fuse, translate, extrude, bolt holes, collar, industrial, mechanism, assembly, 3-stage

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| collapsedLength | 1000 | mm | Length of each tube stage |
| extensionFactor | 0.35 | - | Each stage lifts 35% of collapsedLength |
| t1OuterDia / t1InnerDia | 180 / 160 | mm | Stage 1 outer/inner diameter |
| t2OuterDia / t2InnerDia | 140 / 120 | mm | Stage 2 outer/inner diameter |
| t3OuterDia / t3InnerDia | 100 / 80 | mm | Stage 3 outer/inner diameter |
| t1FlangeDia / t1FlangeThk | 260 / 20 | mm | Stage 1 flange diameter and thickness |
| t3FlangeDia / t3FlangeThk | 180 / 15 | mm | Stage 3 top flange |
| boltCount | 8 | - | Number of bolt holes per flange |
| boltHoleRadius | 8 | mm | Bolt hole radius |
| pinionRadius | 30 | mm | Rack-and-pinion gear radius |
| gearToothCount | 12 | - | Number of teeth on pinion (polygon sides) |
| rackWidth / rackThickness | 20 / 15 | mm | Rack cross-section |

## Replicad Functions Used
| Function | What it does |
|----------|-------------|
| drawCircle(r) | Creates circular sketch |
| .cut(drawCircle(r)) | 2D boolean subtract to make ring |
| .sketchOnPlane("XY", z) | Places sketch at height z |
| .sketchOnPlane("YZ") | Places pinion on YZ plane |
| .extrude(h) | Extrudes to 3D |
| drawRectangle(w,h) | Creates rack cross-section |
| drawPolysides(r,n) | Creates n-sided polygon for pinion gear |
| .translate(x,y,z) | Positions components |
| .fuse(other) | Boolean union for assembly |
| Math.cos/sin | Calculates bolt hole positions in circular pattern |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawRectangle, drawPolysides } = replicad;
  const collapsedLength=1000, extensionFactor=0.35;
  const t1OuterDia=180, t1InnerDia=160, t1FlangeDia=260, t1FlangeThk=20, t1CollarHeight=50;
  const t2OuterDia=140, t2InnerDia=120, t2CollarHeight=40;
  const t3OuterDia=100, t3InnerDia=80, t3FlangeDia=180, t3FlangeThk=15;
  const rackWidth=20, rackThickness=15, pinionRadius=30, pinionWidth=25;
  const boltHoleRadius=8, boltCount=8, gearToothCount=12;
  const createTube = (od,id,length) =>
    drawCircle(od/2).cut(drawCircle(id/2)).sketchOnPlane().extrude(length);
  const createFlange = (od,id,thickness,boltCircleDia) => {
    let sketch=drawCircle(od/2).cut(drawCircle(id/2));
    const r=boltCircleDia/2;
    for(let i=0;i<boltCount;i++){
      const a=(i*360/boltCount)*Math.PI/180;
      sketch=sketch.cut(drawCircle(boltHoleRadius).translate(r*Math.cos(a),r*Math.sin(a)));
    }
    return sketch.sketchOnPlane().extrude(thickness);
  };
  const createPinion = (radius,width) =>
    drawPolysides(radius,gearToothCount).sketchOnPlane("YZ").extrude(width).translate(-width/2,0,0);
  let stage1 = createTube(t1OuterDia,t1InnerDia,collapsedLength)
    .fuse(createFlange(t1FlangeDia,t1InnerDia,t1FlangeThk,t1FlangeDia-40)).fuse(baseCollar);
  let stage2 = createTube(t2OuterDia,t2InnerDia,collapsedLength).fuse(middleCollar).fuse(rack2)
    .translate(0,0,collapsedLength*extensionFactor);
  let stage3 = createTube(t3OuterDia,t3InnerDia,collapsedLength).fuse(topFlange).fuse(rack3)
    .translate(0,0,collapsedLength*extensionFactor*2);
  const mechanism = driveHousing.fuse(gear).fuse(crankArm).fuse(crankHandle);
  return stage1.fuse(stage2).fuse(stage3).fuse(mechanism);
};
```
