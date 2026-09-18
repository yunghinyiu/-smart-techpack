# Segmentation models (Slice 2)

## Chosen model: cloth-seg U²-Net (`u2net_cloth_seg.onnx`)

- **What:** U²-Net trained for cloth segmentation. Input `1x3x768x768`
  (ImageNet-normalized), output `1x4x768x768` logits over
  background / upper body / lower body / full body. We take argmax and treat
  any non-background class as garment foreground.
- **Source:** `levindabhi/cloth-segmentation` (MIT license).
- **Why this one:** the Slice 2 spike compared it against generic salient
  detection (`u2net.onnx`, rembg release) on five CC0 garment photos
  (two dresses, jacket on mannequin, sheer barong flat-lay, trousers).
  Both were excellent on four; on the jacket-on-mannequin shot the salient
  model kept the whole ensemble (jacket + skirt + mannequin neck) while
  cloth-seg kept jacket + skirt but dropped the mannequin neck. (Both
  models keep the skirt — argmax treats the lower-body channel as
  foreground, so separating co-worn pieces is Slice 5+ work, not Slice 2.)
  Full numbers and the decision are recorded in issue #3. Timings (Apple
  Silicon CPU, ort): ~2.5–2.9 s inference + ~1 s pre/post per photo.
- **Bonus for later slices:** the upper/lower/full-body channels give us
  part labels for callout proposal (Slice 5) for free.

## Installing weights (required to run the server)

Weights are **never committed** (176 MB, and the repo must stay lean).
Fetch once into the gitignored runtime dir:

```sh
mkdir -p data/weights
curl -L -o data/weights/cloth-seg-u2net.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx
```

Verify integrity (MD5 `2434d1f3cb744e0e49386c906e5a08bb`):

```sh
md5 data/weights/cloth-seg-u2net.onnx
# macOS: md5; Linux: md5sum
```

Without this file the server still starts, but uploads return
`503` with a pointer back here. The app never downloads weights at
runtime: `ClothSeg::load` reads a local path and the CPU execution
provider performs no network I/O (pinned by the offline unit test in
`crates/vision`).

## Rejected alternative

- **Generic salient U²-Net** (`u2net.onnx`, rembg release): same backbone,
  1-channel saliency output. Faster (~0.5 s inference) but keeps
  non-garment foreground (mannequin necks, display stands). Note:
  several mirrors relabel this file as cloth segmentation — check the
  output shape (`1x4x768x768` = real cloth-seg, `1x1x320x320` = salient).
