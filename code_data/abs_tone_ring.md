---
source_file: abs_tone_ring.js
category: wheel_hub
type: annotated_code
use_case: provides speed signal pulses to ABS sensor via toothed ring rotating with hub
related: hub_flange.md, bearing_race.md, wheel_stud.md
---
# ABS Tone Ring

## Description
A stamped steel ring with 48 equally-spaced teeth pressed onto the hub flange. As the hub rotates, the teeth pass an inductive sensor, generating speed pulses for ABS and traction control systems.

## Keywords
ABS tone ring, reluctor ring, speed sensor ring, toothed ring, wheel speed, anti-lock brakes, traction control, teeth, press fit, extrude, fuse, cut, stamped steel, sensor ring

## Parameters
| Variable | Value | Unit | Meaning |
|----------|-------|------|---------|
| RING_OUTER_R | 68 | mm | outer radius of ring |
| RING_INNER_R | 62 | mm | inner press-fit radius |
| RING_WIDTH | 10 | mm | axial width of ring |
| TOOTH_H | 4 | mm | tooth height |
| TOOTH_W | 3 | mm | tooth width |
| NUM_TEETH | 48 | — | number of ABS teeth |

## Code
```javascript
const main = (replicad) => {
  const {
    draw,
    drawCircle,
  } = replicad;

  const RING_OUTER_R = 68;
  const RING_INNER_R = 62;
  const RING_WIDTH   = 10;
  const TOOTH_H      = 4;
  const TOOTH_W      = 3;
  const NUM_TEETH    = 48;

  // Base ring body
  const ringProfile = draw([RING_INNER_R, 0])
    .hLine(RING_OUTER_R - RING_INNER_R)
    .vLine(RING_WIDTH)
    .hLine(-(RING_OUTER_R - RING_INNER_R))
    .close();
  let ring = ringProfile.sketchOnPlane("XZ").revolve();

  // Teeth around outer circumference
  const TOOTH_PITCH_ANGLE = 360 / NUM_TEETH;
  const TOOTH_ARC_W = (2 * Math.PI * RING_OUTER_R) / NUM_TEETH * 0.5;

  for (let i = 0; i < NUM_TEETH; i++) {
    const angle = i * TOOTH_PITCH_ANGLE;
    const tooth = draw([-TOOTH_ARC_W / 2, RING_OUTER_R])
      .hLine(TOOTH_ARC_W)
      .vLine(TOOTH_H)
      .hLine(-TOOTH_ARC_W)
      .close()
      .sketchOnPlane("XY")
      .extrude(RING_WIDTH)
      .rotate(angle, [0, 0, 0], [0, 0, 1]);
    ring = ring.fuse(tooth);
  }

  return { shape: ring, name: "ABS Tone Ring", color: "dimgrey" };
};
```
