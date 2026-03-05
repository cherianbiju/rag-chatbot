---
source_file: piston_crank_assembly_v2.md
category: assembly
type: annotated_code
use_case: engine reciprocating mechanism, forged steel crankshaft with aluminum pistons and steel connecting rods
related: engine_block.md, camshaft_assembly_v2.md, main_bearing_cap.md
---

# Piston-Crank Assembly — Forged Steel / Aluminum Alloy

## Description
A precision engine assembly with a forged steel crankshaft (ground to H7 journal tolerance), aluminum alloy pistons (bore 86 mm, stroke 86 mm) with three ring grooves and steel wrist pins, and steel connecting rods with bushed small-end bores. Crankshaft includes balance counterweights, oil galleries through crankpin, and 4-bolt main bearing cap flanges for block mounting.

## Keywords
forged crankshaft, aluminum piston, connecting rod, wrist pin, oil gallery, balance counterweight, main bearing cap, journal H7 tolerance, ring groove, bore 86mm, stroke 86mm, piston clearance 0.04mm, big end bearing, small end bushing, crankpin, 4-bolt cap, engine assembly, reciprocating mass, forged steel, CNC piston

## Parameters
| Variable              | Value  | Unit | Meaning                                      |
|-----------------------|--------|------|----------------------------------------------|
| pistonRadius          | 43.0   | mm   | Piston radius (bore 86mm / 2)                |
| pistonHeight          | 72.0   | mm   | Total piston height                          |
| pistonWallThickness   | 5.0    | mm   | Piston wall/skirt thickness                  |
| ringGrooveWidth       | 1.5    | mm   | Compression ring groove width                |
| ringGrooveDepth       | 3.5    | mm   | Ring groove radial depth                     |
| oilRingGrooveWidth    | 3.0    | mm   | Oil control ring groove width                |
| wristPinRadius        | 11.0   | mm   | Wrist pin radius                             |
| wristPinLength        | 72.0   | mm   | Wrist pin full length                        |
| rodLength             | 144.0  | mm   | Con rod centre-to-centre                     |
| rodBigEndRadius       | 26.0   | mm   | Big-end bore radius (H7)                     |
| rodSmallEndRadius     | 12.5   | mm   | Small-end bushing bore radius                |
| rodSectionWidth       | 20.0   | mm   | Rod beam width                               |
| rodSectionThick       | 14.0   | mm   | Rod beam thickness                           |
| crankMainRadius       | 30.0   | mm   | Main journal radius (H7)                     |
| crankMainLength       | 24.0   | mm   | Main journal width                           |
| crankPinRadius        | 24.0   | mm   | Crankpin radius (H7)                         |
| crankPinWidth         | 22.0   | mm   | Crankpin width                               |
| crankWebThick         | 18.0   | mm   | Crank web thickness                          |
| crankWebWidth         | 52.0   | mm   | Crank web width                              |
| crankThrow            | 43.0   | mm   | Crank throw = stroke/2                       |
| balanceWeightRadius   | 48.0   | mm   | Counterweight outer radius                   |
| oilGalleryRadius      | 3.5    | mm   | Oil gallery bore radius                      |
| bearingCapWidth       | 52.0   | mm   | Main bearing cap width                       |
| bearingCapHeight      | 22.0   | mm   | Main bearing cap height                      |
| capBoltRadius         | 5.0    | mm   | M10 cap bolt hole radius                     |

## Code
```javascript
const main = (replicad) => {
  const { draw, drawCircle, drawRectangle } = replicad;

  const pistonRadius        = 43.0;
  const pistonHeight        = 72.0;
  const pistonWallThickness = 5.0;
  const ringGrooveWidth     = 1.5;
  const ringGrooveDepth     = 3.5;
  const oilRingGrooveWidth  = 3.0;
  const wristPinRadius      = 11.0;
  const wristPinLength      = 72.0;
  const rodLength           = 144.0;
  const rodBigEndRadius     = 26.0;
  const rodSmallEndRadius   = 12.5;
  const rodSectionWidth     = 20.0;
  const rodSectionThick     = 14.0;
  const crankMainRadius     = 30.0;
  const crankMainLength     = 24.0;
  const crankPinRadius      = 24.0;
  const crankPinWidth       = 22.0;
  const crankWebThick       = 18.0;
  const crankWebWidth       = 52.0;
  const crankThrow          = 43.0;
  const balanceWeightRadius = 48.0;
  const oilGalleryRadius    = 3.5;
  const bearingCapWidth     = 52.0;
  const bearingCapHeight    = 22.0;
  const capBoltRadius       = 5.0;

  // ── PISTON ────────────────────────────────────────────────
  const pistonOuter = drawCircle(pistonRadius)
    .sketchOnPlane("XY", 0)
    .extrude(pistonHeight);

  const pistonInner = drawCircle(pistonRadius - pistonWallThickness)
    .sketchOnPlane("XY", 0)
    .extrude(pistonHeight - 8);

  const groove1 = drawCircle(pistonRadius + 0.5)
    .sketchOnPlane("XY", pistonHeight * 0.82)
    .extrude(ringGrooveWidth)
    .cut(drawCircle(pistonRadius - ringGrooveDepth)
      .sketchOnPlane("XY", pistonHeight * 0.82)
      .extrude(ringGrooveWidth));

  const groove2 = drawCircle(pistonRadius + 0.5)
    .sketchOnPlane("XY", pistonHeight * 0.71)
    .extrude(ringGrooveWidth)
    .cut(drawCircle(pistonRadius - ringGrooveDepth)
      .sketchOnPlane("XY", pistonHeight * 0.71)
      .extrude(ringGrooveWidth));

  const groove3 = drawCircle(pistonRadius + 0.5)
    .sketchOnPlane("XY", pistonHeight * 0.58)
    .extrude(oilRingGrooveWidth)
    .cut(drawCircle(pistonRadius - ringGrooveDepth)
      .sketchOnPlane("XY", pistonHeight * 0.58)
      .extrude(oilRingGrooveWidth));

  const wristPinBore = drawCircle(wristPinRadius)
    .sketchOnPlane("XZ", pistonHeight * 0.3)
    .extrude(wristPinLength)
    .translateX(-wristPinLength / 2);

  const piston = pistonOuter
    .cut(pistonInner)
    .cut(groove1)
    .cut(groove2)
    .cut(groove3)
    .cut(wristPinBore);

  // ── WRIST PIN ─────────────────────────────────────────────
  const wristPin = drawCircle(wristPinRadius - 2.5)
    .sketchOnPlane("XZ", pistonHeight * 0.3)
    .extrude(wristPinLength)
    .translateX(-wristPinLength / 2);

  // ── CONNECTING ROD ────────────────────────────────────────
  const rodBody = drawRectangle(rodSectionWidth, rodLength - rodBigEndRadius - rodSmallEndRadius)
    .sketchOnPlane("XY", rodBigEndRadius)
    .extrude(rodSectionThick)
    .translateX(-rodSectionWidth / 2);

  const bigEndOuter = drawCircle(rodBigEndRadius + 8)
    .sketchOnPlane("XY", 0)
    .extrude(rodSectionThick);
  const bigEndBore = drawCircle(rodBigEndRadius)
    .sketchOnPlane("XY", 0)
    .extrude(rodSectionThick);

  const smallEndOuter = drawCircle(rodSmallEndRadius + 7)
    .sketchOnPlane("XY", rodLength)
    .extrude(rodSectionThick);
  const smallEndBore = drawCircle(rodSmallEndRadius)
    .sketchOnPlane("XY", rodLength)
    .extrude(rodSectionThick);

  const rodOilHole = drawCircle(oilGalleryRadius)
    .sketchOnPlane("XY", rodLength * 0.5)
    .extrude(rodSectionThick);

  const connectingRod = rodBody
    .fuse(bigEndOuter.cut(bigEndBore))
    .fuse(smallEndOuter.cut(smallEndBore))
    .cut(rodOilHole)
    .translateY(-(rodBigEndRadius + 8))
    .translateZ(pistonHeight * 0.3 - rodSectionThick / 2);

  // ── CRANKSHAFT ────────────────────────────────────────────
  const totalCrankZ = crankMainLength + crankWebThick + crankPinWidth + crankWebThick + crankMainLength;

  const journal1 = drawCircle(crankMainRadius)
    .sketchOnPlane("XY", 0)
    .extrude(crankMainLength);

  const web1Profile = draw([-crankWebWidth / 2, -crankMainRadius])
    .lineTo([crankWebWidth / 2, -crankMainRadius])
    .lineTo([crankWebWidth / 2, crankThrow + crankPinRadius])
    .lineTo([-crankWebWidth / 2, crankThrow + crankPinRadius])
    .close();
  const web1 = web1Profile
    .sketchOnPlane("XY", crankMainLength)
    .extrude(crankWebThick);

  const crankPin = drawCircle(crankPinRadius)
    .sketchOnPlane("XY", crankMainLength + crankWebThick)
    .extrude(crankPinWidth)
    .translateY(crankThrow);

  const crankPinOilGallery = drawCircle(oilGalleryRadius)
    .sketchOnPlane("XY", crankMainLength + crankWebThick)
    .extrude(crankPinWidth)
    .translateY(crankThrow);

  const web2 = web1Profile
    .sketchOnPlane("XY", crankMainLength + crankWebThick + crankPinWidth)
    .extrude(crankWebThick);

  const journal2 = drawCircle(crankMainRadius)
    .sketchOnPlane("XY", crankMainLength + crankWebThick * 2 + crankPinWidth)
    .extrude(crankMainLength);

  const balanceWeight = drawCircle(balanceWeightRadius)
    .sketchOnPlane("XY", crankMainLength)
    .extrude(crankWebThick + crankPinWidth + crankWebThick)
    .cut(drawCircle(crankMainRadius + 5)
      .sketchOnPlane("XY", crankMainLength)
      .extrude(crankWebThick + crankPinWidth + crankWebThick))
    .translateY(-balanceWeightRadius * 0.55);

  const crankshaft = journal1
    .fuse(web1)
    .fuse(crankPin.cut(crankPinOilGallery))
    .fuse(web2)
    .fuse(journal2)
    .fuse(balanceWeight)
    .translateZ(-totalCrankZ / 2);

  // ── MAIN BEARING CAP (4-bolt) ─────────────────────────────
  const capZ = -crankMainLength / 2;
  const capBody = drawRectangle(bearingCapWidth, bearingCapHeight)
    .sketchOnPlane("XY", capZ)
    .extrude(crankMainLength)
    .translateX(-bearingCapWidth / 2)
    .translateY(-crankMainRadius - bearingCapHeight);

  const capSemiBore = drawCircle(crankMainRadius + 1.5)
    .sketchOnPlane("XY", capZ)
    .extrude(crankMainLength)
    .translateY(-crankMainRadius * 0.5);

  const bolt1 = drawCircle(capBoltRadius).sketchOnPlane("XY", capZ).extrude(crankMainLength)
    .translateX(-bearingCapWidth * 0.35).translateY(-crankMainRadius - bearingCapHeight * 0.4);
  const bolt2 = bolt1.clone().translateX(bearingCapWidth * 0.7);
  const bolt3 = drawCircle(capBoltRadius).sketchOnPlane("XY", capZ).extrude(crankMainLength)
    .translateX(-bearingCapWidth * 0.35).translateY(-crankMainRadius - bearingCapHeight * 0.85);
  const bolt4 = bolt3.clone().translateX(bearingCapWidth * 0.7);

  const bearingCap = capBody
    .cut(capSemiBore)
    .cut(bolt1).cut(bolt2).cut(bolt3).cut(bolt4);

  return [
    { shape: piston,        name: "Aluminum Alloy Piston",  color: "#C8C8C8" },
    { shape: wristPin,      name: "Steel Wrist Pin",        color: "#A0A0A0" },
    { shape: connectingRod, name: "Steel Connecting Rod",   color: "#8C7B6E" },
    { shape: crankshaft,    name: "Forged Crankshaft",      color: "#607080" },
    { shape: bearingCap,    name: "Main Bearing Cap",       color: "#505A60" },
  ];
};
```
