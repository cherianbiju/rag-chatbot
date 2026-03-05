---
source_file: shoppingCardCoin3.js
category: consumer_product, hardware
type: annotated_code
use_case: 3D printable shopping cart token, keyring attachment
related: keyring_designs.md, printable_coins.md
---

# Shopping Cart Coin (Euro) with Keyring Attachment

## Description
Models a 3D-printable shopping cart token sized to match standard euro coins (1€, 0.50€, or 2€), with a rounded handle extension and a keyring hole for everyday carry. The coin profile is built from boolean operations on 2D shapes before extrusion, demonstrating fuse and cut operations on drawings.

## Keywords
shopping cart coin, keyring, euro coin, boolean union, boolean cut, fuse, drawCircle, drawRoundedRectangle, drawPolysides, hexagon cutout, extrude, fillet, parametric, consumer product, 3D print

## Parameters
| Variable       | Value  | Unit | Meaning                                         |
|----------------|--------|------|-------------------------------------------------|
| coinDiameter   | 23.25  | mm   | Coin diameter (1 euro; change for 0.50 / 2 euro)|
| coinThickness  | 2.38   | mm   | Coin thickness (1 euro)                         |
| handleLength   | 22     | mm   | Length of the keyring handle extension          |
| handleWidth    | 8      | mm   | Width of the handle                             |
| handleHoleDia  | 4      | mm   | Keyring hole diameter (handleWidth − 4)         |

## Code
```javascript
// FILE: shoppingCardCoin3.js
// 3D-printable shopping cart token sized to a euro coin
// with a handle and keyring hole.

const { draw,
        drawCircle,
        drawRoundedRectangle,
        drawPolysides,
      } = replicad;

const main = () => {

  // --- Coin dimensions (euro standard) ---
  // Uncomment the appropriate pair for the coin size you want:
  const coinDiameter = 23.25;   // 1 euro
  // const coinDiameter = 24.25; // 0.50 euro
  // const coinDiameter = 25.75; // 2 euro

  const coinThickness = 2.38;   // 1 euro
  // const coinThickness = 2.33; // 0.50 euro
  // const coinThickness = 2.20; // 2 euro

  // --- Handle dimensions ---
  const handleLength = 22;                        // mm, extension beyond coin
  const handleWidth = 8;                          // mm
  const handleHoleDia = handleWidth - 4;          // mm, keyring hole diameter

  // --- 2D profile construction ---
  let coinContour   = drawCircle(coinDiameter / 2);

  let handleContour = drawRoundedRectangle(handleLength, handleWidth, handleWidth / 2)
    .translate(handleLength / 2);                 // shift handle to sit beside coin

  let handleHole    = drawCircle(handleHoleDia / 2)
    .translate(handleLength - (handleWidth / 2)); // center hole at far end of handle

  let hexagon       = drawPolysides(6.5, 6, 0);  // hex cutout in coin face (decorative)

  // --- Boolean operations on 2D drawings ---
  let totalContour = coinContour
    .fuse(handleContour)   // add handle to coin disc
    .cut(handleHole)       // punch keyring hole
    .cut(hexagon);         // punch hex recess in coin face

  // --- Extrude and fillet ---
  let shoppingCartCoin = totalContour
    .sketchOnPlane("XY")
    .extrude(coinThickness)
    .fillet(0.5);           // soften all edges for printability

  return shoppingCartCoin;
};
```
