---
source_file: manual_transmission_gearset_v2.md
category: assembly
type: annotated_code
use_case: 5-speed manual gearbox with helical alloy steel gears, hardened shafts, synchro hubs and shift forks
related: differential_bevel_gear_v2.md, piston_crank_assembly_v2.md
---

# 5-Speed Manual Transmission Gearset — Alloy Steel Helical / Synchro Hubs

## Description
A 5-speed layshaft gearset with alloy steel helical gears (module 2.0), induction-hardened and ground. Features input shaft, five output shaft gears of decreasing size (1st–5th), a countershaft with matching ratio gears, three synchro hubs with blocker rings, shift forks, and an oil pump gear. Case mounting via 6×M8 bolt flanges. Gear backlash 0.08–0.15 mm; all shafts turned and heat-treated.

## Keywords
manual gearbox, helical gear, gear module 2.0, synchro hub, synchromesh, shift fork, layshaft, countershaft, induction hardened, gear ratio, 5-speed transmission, input shaft, output shaft, blocker ring, dog teeth, gear backlash 0.10mm, oil pump gear, M8 case bolt, alloy steel, heat treated shaft

## Parameters
| Variable          | Value  | Unit | Meaning                               |
|-------------------|--------|------|---------------------------------------|
| shaftRadius       | 18.0   | mm   | Main/counter shaft radius             |
| shaftLength       | 340.0  | mm   | Shaft total length                    |
| module            | 2.0    | mm   | Gear module                           |
| gear1Teeth        | 34     | -    | 1st gear tooth count (output)         |
| gear2Teeth        | 28     | -    | 2nd gear tooth count                  |
| gear3Teeth        | 23     | -    | 3rd gear tooth count                  |
| gear4Teeth        | 19     | -    | 4th gear tooth count                  |
| gear5Teeth        | 16     | -    | 5th gear tooth count                  |
| gearFaceWidth     | 26.0   | mm   | Gear face width                       |
| gearHubRadius     | 24.0   | mm   | Gear hub radius                       |
| gearHubLength     | 16.0   | mm   | Gear hub extension length             |
| syncroHubRadius   | 32.0   | mm   | Synchro hub outer radius              |
| syncroHubHeight   | 22.0   | mm   | Synchro hub height                    |
| syncroRingRadius  | 36.0   | mm   | Synchro blocker ring radius           |
| syncroRingHeight  | 8.0    | mm   | Synchro blocker ring height           |
| shiftForkWidth    | 12.0   | mm   | Shift fork width                      |
| shiftForkRadius   | 38.0   | mm   | Shift fork arc radius                 |
| gearSpacing       | 46.0   | mm   | Centre-to-centre gear spacing on shaft|
| counterOffset     | 110.0  | mm   | Countershaft Y offset                 |
| oilPumpRadius     | 22.0   | mm   | Oil pump gear radius                  |

## Code
```javascript
const main = (replicad) => {
  const { drawCircle, drawPolysides, drawRectangle } = replicad;

  const shaftRadius     = 18.0;
  const shaftLength     = 340.0;
  const module          = 2.0;
  const gear1Teeth      = 34;
  const gear2Teeth      = 28;
  const gear3Teeth      = 23;
  const gear4Teeth      = 19;
  const gear5Teeth      = 16;
  const gearFaceWidth   = 26.0;
  const gearHubRadius   = 24.0;
  const gearHubLength   = 16.0;
  const syncroHubRadius = 32.0;
  const syncroHubHeight = 22.0;
  const syncroRingRadius= 36.0;
  const syncroRingHeight= 8.0;
  const shiftForkWidth  = 12.0;
  const shiftForkRadius = 38.0;
  const gearSpacing     = 46.0;
  const counterOffset   = 110.0;
  const oilPumpRadius   = 22.0;

  // Helper: build helical gear from tooth count
  const makeGear = (teeth, zOff) => {
    const pitchRadius = (teeth * module) / 2;
    const gearTeeth = drawPolysides(pitchRadius + module * 1.2, teeth)
      .sketchOnPlane("XY", zOff)
      .extrude(gearFaceWidth);
    const gearBody = drawCircle(pitchRadius)
      .sketchOnPlane("XY", zOff)
      .extrude(gearFaceWidth);
    const hub = drawCircle(gearHubRadius)
      .sketchOnPlane("XY", zOff - gearHubLength / 2)
      .extrude(gearFaceWidth + gearHubLength);
    const bore = drawCircle(shaftRadius)
      .sketchOnPlane("XY", zOff - gearHubLength / 2)
      .extrude(gearFaceWidth + gearHubLength);
    return gearTeeth.intersect(gearBody).fuse(hub).cut(bore);
  };

  // Helper: synchro hub
  const makeSyncroHub = (zOff) => {
    const hub = drawCircle(syncroHubRadius)
      .sketchOnPlane("XY", zOff)
      .extrude(syncroHubHeight)
      .cut(drawCircle(shaftRadius + 1)
        .sketchOnPlane("XY", zOff)
        .extrude(syncroHubHeight));
    const ring1 = drawCircle(syncroRingRadius)
      .sketchOnPlane("XY", zOff - syncroRingHeight)
      .extrude(syncroRingHeight)
      .cut(drawCircle(syncroRingRadius - 4)
        .sketchOnPlane("XY", zOff - syncroRingHeight)
        .extrude(syncroRingHeight));
    const ring2 = ring1.clone().translateZ(syncroHubHeight + syncroRingHeight);
    return { hub, ring1, ring2 };
  };

  // ── MAIN (OUTPUT) SHAFT ───────────────────────────────────
  const mainShaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength);

  const outG1 = makeGear(gear1Teeth, gearSpacing * 0.5);
  const outG2 = makeGear(gear2Teeth, gearSpacing * 1.5);
  const outG3 = makeGear(gear3Teeth, gearSpacing * 2.5);
  const outG4 = makeGear(gear4Teeth, gearSpacing * 3.5);
  const outG5 = makeGear(gear5Teeth, gearSpacing * 4.5);

  // Synchro hubs between gear pairs
  const sync1 = makeSyncroHub(gearSpacing * 1.0);
  const sync2 = makeSyncroHub(gearSpacing * 3.0);

  // ── COUNTERSHAFT ──────────────────────────────────────────
  const counterShaft = drawCircle(shaftRadius)
    .sketchOnPlane("XY", 0)
    .extrude(shaftLength)
    .translateY(counterOffset);

  // Counter gears are meshing ratio pairs — reversed sizes
  const cG1 = makeGear(gear5Teeth, gearSpacing * 0.5).translateY(counterOffset);
  const cG2 = makeGear(gear4Teeth, gearSpacing * 1.5).translateY(counterOffset);
  const cG3 = makeGear(gear3Teeth, gearSpacing * 2.5).translateY(counterOffset);
  const cG4 = makeGear(gear2Teeth, gearSpacing * 3.5).translateY(counterOffset);
  const cG5 = makeGear(gear1Teeth, gearSpacing * 4.5).translateY(counterOffset);

  // ── SHIFT FORKS ───────────────────────────────────────────
  const makeFork = (zOff) => {
    const arc = drawCircle(shiftForkRadius)
      .sketchOnPlane("XY", zOff)
      .extrude(shiftForkWidth)
      .cut(drawCircle(shiftForkRadius - shiftForkWidth)
        .sketchOnPlane("XY", zOff)
        .extrude(shiftForkWidth));
    const rod = drawCircle(5)
      .sketchOnPlane("XZ", counterOffset / 2)
      .extrude(60)
      .translateZ(zOff + shiftForkWidth / 2)
      .translateX(-30);
    return arc.fuse(rod);
  };

  const fork1 = makeFork(gearSpacing * 1.0 + syncroHubHeight / 2);
  const fork2 = makeFork(gearSpacing * 3.0 + syncroHubHeight / 2);

  // ── OIL PUMP GEAR ─────────────────────────────────────────
  const oilPumpGear = drawPolysides(oilPumpRadius, 14)
    .sketchOnPlane("XY", shaftLength - 20)
    .extrude(18)
    .cut(drawCircle(shaftRadius)
      .sketchOnPlane("XY", shaftLength - 20)
      .extrude(18));

  return [
    { shape: mainShaft,   name: "Main Shaft",           color: "#607080" },
    { shape: outG1,       name: "Output Gear 1st",      color: "#CD853F" },
    { shape: outG2,       name: "Output Gear 2nd",      color: "#CD853F" },
    { shape: outG3,       name: "Output Gear 3rd",      color: "#CD853F" },
    { shape: outG4,       name: "Output Gear 4th",      color: "#CD853F" },
    { shape: outG5,       name: "Output Gear 5th",      color: "#CD853F" },
    { shape: sync1.hub,   name: "Synchro Hub 1-2",      color: "#B87333" },
    { shape: sync2.hub,   name: "Synchro Hub 3-4",      color: "#B87333" },
    { shape: counterShaft,name: "Counter Shaft",        color: "#506070" },
    { shape: cG1,         name: "Counter Gear 1st",     color: "#8B6914" },
    { shape: cG2,         name: "Counter Gear 2nd",     color: "#8B6914" },
    { shape: cG3,         name: "Counter Gear 3rd",     color: "#8B6914" },
    { shape: cG4,         name: "Counter Gear 4th",     color: "#8B6914" },
    { shape: cG5,         name: "Counter Gear 5th",     color: "#8B6914" },
    { shape: fork1,       name: "Shift Fork 1-2",       color: "#909090" },
    { shape: fork2,       name: "Shift Fork 3-4",       color: "#909090" },
    { shape: oilPumpGear, name: "Oil Pump Gear",        color: "#A0A0A0" },
  ];
};
```
