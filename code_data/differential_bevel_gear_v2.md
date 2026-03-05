---
source_file: differential_bevel_gear_v2.md
category: assembly
type: annotated_code
use_case: open differential torque split, alloy steel ring and pinion 3.9:1 ratio with splined axle outputs
related: manual_transmission_gearset_v2.md, wheel_hub_bearing_v2.md
---

# Open Differential Bevel Gear Assembly — 3.9:1 Alloy Steel

## Description
An open differential with a 3.9:1 ring-and-pinion gear set (carburized and ground), differential carrier with 8×M10 housing bolt flanges, two spider bevel pinion gears on a cross pin, and two side bevel gears with internal splines for axle shaft drive. Shim sets establish bearing preload and tooth contact pattern. Oil grooves machined into spider gear faces for lubrication flow.

## Keywords
open differential, ring gear, pinion gear, bevel gear, 3.9 ratio, spider gear, side gear, splined axle, differential carrier, cross pin, backlash 0.08mm, carburized gear, preload shim, M10 carrier bolt, tooth contact pattern, lubrication groove, alloy steel, drive axle, torque split, cornering

## Parameters
| Variable           | Value  | Unit | Meaning                               |
|--------------------|--------|------|---------------------------------------|
| carrierRadius      | 95.0   | mm   | Carrier outer radius                  |
| carrierHeight      | 85.0   | mm   | Carrier height                        |
| carrierWallThick   | 10.0   | mm   | Carrier wall thickness                |
| ringGearRadius     | 92.0   | mm   | Ring gear outer pitch radius          |
| ringGearWidth      | 24.0   | mm   | Ring gear face width                  |
| ringTeeth          | 39     | -    | Ring gear tooth count (ratio 3.9:1)   |
| pinionTeeth        | 10     | -    | Drive pinion tooth count              |
| pinionRadius       | 28.0   | mm   | Drive pinion pitch radius             |
| pinionLength       | 65.0   | mm   | Drive pinion total length             |
| sideGearRadius     | 42.0   | mm   | Side bevel gear pitch radius          |
| sideGearHeight     | 28.0   | mm   | Side bevel gear height                |
| sideTeeth          | 16     | -    | Side gear tooth count                 |
| spiderRadius       | 30.0   | mm   | Spider pinion radius                  |
| spiderHeight       | 24.0   | mm   | Spider pinion height                  |
| spiderTeeth        | 10     | -    | Spider pinion tooth count             |
| crossPinRadius     | 12.0   | mm   | Cross pin radius                      |
| crossPinLength     | 90.0   | mm   | Cross pin length                      |
| axleRadius         | 20.0   | mm   | Output axle stub radius               |
| axleSplineRadius   | 16.0   | mm   | Axle spline root radius               |
| axleLength         | 130.0  | mm   | Axle stub length                      |
| carrierBoltRadius  | 5.5    | mm   | M10 carrier bolt hole radius          |
| shimThickness      | 1.5    | mm   | Preload shim thickness                |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides } = replicad;

  const carrierRadius     = 95.0;
  const carrierHeight     = 85.0;
  const carrierWallThick  = 10.0;
  const ringGearRadius    = 92.0;
  const ringGearWidth     = 24.0;
  const ringTeeth         = 39;
  const pinionTeeth       = 10;
  const pinionRadius      = 28.0;
  const pinionLength      = 65.0;
  const sideGearRadius    = 42.0;
  const sideGearHeight    = 28.0;
  const sideTeeth         = 16;
  const spiderRadius      = 30.0;
  const spiderHeight      = 24.0;
  const spiderTeeth       = 10;
  const crossPinRadius    = 12.0;
  const crossPinLength    = 90.0;
  const axleRadius        = 20.0;
  const axleSplineRadius  = 16.0;
  const axleLength        = 130.0;
  const carrierBoltRadius = 5.5;
  const shimThickness     = 1.5;

  // ── DIFFERENTIAL CARRIER ──────────────────────────────────
  const carrierOuter = drawCircle(carrierRadius)
    .sketchOnPlane("XY", 0)
    .extrude(carrierHeight);
  const carrierInner = drawCircle(carrierRadius - carrierWallThick)
    .sketchOnPlane("XY", carrierWallThick)
    .extrude(carrierHeight - carrierWallThick * 2);
  const axleBore = drawCircle(axleRadius + 3)
    .sketchOnPlane("XY", 0)
    .extrude(carrierHeight);

  // 8× M10 flange bolt holes
  let carrier = carrierOuter.cut(carrierInner).cut(axleBore);
  for (let i = 0; i < 8; i++) {
    const angle = i * 45;
    const rad = angle * Math.PI / 180;
    const bx = (carrierRadius - 12) * Math.cos(rad);
    const by = (carrierRadius - 12) * Math.sin(rad);
    const bolt = drawCircle(carrierBoltRadius)
      .sketchOnPlane("XY", 0)
      .extrude(carrierWallThick)
      .translateX(bx).translateY(by);
    carrier = carrier.cut(bolt);
  }

  // ── RING GEAR ─────────────────────────────────────────────
  const ringTeethShape = drawPolysides(ringGearRadius + 3, ringTeeth)
    .sketchOnPlane("XY", carrierHeight)
    .extrude(ringGearWidth);
  const ringBody = drawCircle(ringGearRadius)
    .sketchOnPlane("XY", carrierHeight)
    .extrude(ringGearWidth);
  const ringBore = drawCircle(carrierRadius - carrierWallThick - 2)
    .sketchOnPlane("XY", carrierHeight)
    .extrude(ringGearWidth);
  const ringGear = ringTeethShape.intersect(ringBody).cut(ringBore);

  // ── DRIVE PINION ──────────────────────────────────────────
  const pinionTeethShape = drawPolysides(pinionRadius + 2.5, pinionTeeth)
    .sketchOnPlane("XY", 0)
    .extrude(ringGearWidth);
  const pinionBody = drawCircle(pinionRadius)
    .sketchOnPlane("XY", 0)
    .extrude(ringGearWidth);
  const pinionShaft = drawCircle(pinionRadius * 0.6)
    .sketchOnPlane("XY", ringGearWidth)
    .extrude(pinionLength);
  const drivePinion = pinionTeethShape.intersect(pinionBody)
    .fuse(pinionShaft)
    .translateX(ringGearRadius + pinionRadius + 5)
    .translateZ(carrierHeight + ringGearWidth / 2 - ringGearWidth / 2);

  // ── SIDE BEVEL GEARS ─────────────────────────────────────
  const sideTeethShape = drawPolysides(sideGearRadius + 2.5, sideTeeth)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);
  const sideGearBody = drawCircle(sideGearRadius)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);
  const sideGearBore = drawCircle(axleSplineRadius)
    .sketchOnPlane("XY", 0)
    .extrude(sideGearHeight);
  const sideGearL = sideTeethShape.intersect(sideGearBody).cut(sideGearBore)
    .translateZ(carrierHeight * 0.12);
  const sideGearR = sideGearL.clone().translateZ(carrierHeight * 0.5);

  // ── SPIDER PINION GEARS ───────────────────────────────────
  const spiderTeethShape = drawPolysides(spiderRadius + 2, spiderTeeth)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderHeight);
  const spiderBody = drawCircle(spiderRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderHeight);
  const spiderBore = drawCircle(crossPinRadius)
    .sketchOnPlane("XZ", 0)
    .extrude(spiderHeight);
  const spider1 = spiderTeethShape.intersect(spiderBody).cut(spiderBore)
    .translateY(-spiderHeight / 2)
    .translateZ(carrierHeight / 2);
  const spider2 = spider1.clone().rotate(90, [0, 0, carrierHeight / 2], [0, 0, 1]);

  // ── CROSS PIN ─────────────────────────────────────────────
  const crossPin = drawCircle(crossPinRadius - 1)
    .sketchOnPlane("XZ", 0)
    .extrude(crossPinLength)
    .translateY(-crossPinLength / 2)
    .translateZ(carrierHeight / 2);

  // ── OUTPUT AXLE STUBS ─────────────────────────────────────
  const axleLeft = drawCircle(axleRadius)
    .sketchOnPlane("XY", -axleLength)
    .extrude(axleLength);
  const axleRight = drawCircle(axleRadius)
    .sketchOnPlane("XY", carrierHeight)
    .extrude(axleLength);

  // ── PRELOAD SHIMS ─────────────────────────────────────────
  const shim1 = drawCircle(axleRadius + 8)
    .sketchOnPlane("XY", 0)
    .extrude(shimThickness)
    .cut(drawCircle(axleRadius + 1)
      .sketchOnPlane("XY", 0)
      .extrude(shimThickness));
  const shim2 = shim1.clone().translateZ(carrierHeight - shimThickness);

  return [
    { shape: carrier,    name: "Differential Carrier",  color: "#5A6A78" },
    { shape: ringGear,   name: "Ring Gear (39T)",        color: "#B8860B" },
    { shape: drivePinion,name: "Drive Pinion (10T)",     color: "#8B6914" },
    { shape: sideGearL,  name: "Side Gear Left",         color: "#CD853F" },
    { shape: sideGearR,  name: "Side Gear Right",        color: "#CD853F" },
    { shape: spider1,    name: "Spider Pinion 1",        color: "#A0522D" },
    { shape: spider2,    name: "Spider Pinion 2",        color: "#A0522D" },
    { shape: crossPin,   name: "Cross Pin",              color: "#909090" },
    { shape: axleLeft,   name: "Axle Stub Left",         color: "#778899" },
    { shape: axleRight,  name: "Axle Stub Right",        color: "#778899" },
    { shape: shim1,      name: "Preload Shim 1",         color: "#D4AF37" },
    { shape: shim2,      name: "Preload Shim 2",         color: "#D4AF37" },
  ];
};
```
