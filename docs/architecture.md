# Architecture and reconstruction decisions

## The controlled difference

Both alternatives have identical trainable parameter shapes and a common state
dict. The only difference is where the same convolutional stem runs:

1. **image_conv:** RGB image → two convolutions over the whole image → partition
   feature map into spatial blocks → flatten/project each block → patch tokens.
2. **patch_conv:** RGB image → nonoverlapping patches → run the same two
   convolutions independently on every patch → flatten/project → patch tokens.

The stem is Conv2d(3,16,3,stride=2,padding=1), per-location channel LayerNorm,
ReLU, then Conv2d(16,32,3,stride=2,padding=1), channel LayerNorm, ReLU. No
normalization aggregates across spatial positions. In the patch-first path the
weights are shared across every patch; there is no separate CNN per patch.

For 224×224 images and 16×16 patches:

- Full-image stem: `[B,3,224,224] → [B,32,56,56]`; 4×4 feature blocks give
  `[B,196,512]` before the shared linear projection to 400 dimensions.
- Patch-first stem: `[B,196,3,16,16] → [B*196,32,4,4]`; reshape gives the same
  `[B,196,512]`, projected to `[B,196,400]`.
- Raster token order and channel/spatial flatten order agree in both variants.
- Full-image convolution uses neighboring pixels across future patch boundaries.
  Patch convolution pads those boundaries instead. This is a testable distinction,
  not two equivalent implementations of a kernel=stride patch embedding.

The test suite changes a pixel beside a patch boundary and checks that an adjacent
token changes only in the full-image version. It also checks matching parameter
counts and common-weight loading. Equal parameters do not imply equal execution
time: reshape/unfold, convolution kernels and memory traffic can differ.

## Shared downstream model

- Learned CLS token and 197 learned visual positional embeddings.
- Pre-normalization vision transformer; CLS output only is retained.
- Linear(vision_dim,text_dim) + GELU projects the pooled image to one text-width
  visual prefix token.
- Learned text embedding; prepend the visual prefix; learned positional embedding;
  causal pre-normalization decoder; final LayerNorm and untied vocabulary head.
- Loss is next-token cross-entropy on real text and EOS only. PAD is ignored.
  BOS predicts the first text token; the output at the image position is discarded.
- Right padding cannot influence earlier real tokens under causal attention.
  No padding attention mask is required for the implemented right-padded decoder.
- All weights are trained from scratch. No pretrained vision/language backbone.

Mini uses 1 vision block, 400 visual dimensions, 4 decoder blocks, 96 text
dimensions, 8 decoder heads (12 dimensions per head), dropout 0.1, and learning
rate 0.001, following the paper's listed values. Encoder heads are separately
specified as 8; the paper does not clearly distinguish encoder attention settings.

## Explicit choices not recoverable from the paper

The stem channels, kernels, strides, padding, exact LayerNorm axes, MLP ratio 4,
learned position embeddings, vocabulary method/size, untied output weights,
AdamW settings, batch size, context budget, precision, and epoch budget are
reconstruction decisions. They must not be presented as author-specified values.

Images are EXIF-oriented, converted to RGB, resized directly to a square with
bicubic interpolation, and normalized from [0,1] to [-1,1]. No random augmentation
is used, making paired input and restart behavior easier to verify. Square resize
can distort aspect ratio; it is a declared preprocessing choice.

Text uses a lowercase word/punctuation tokenizer, retaining contractions, fitted
only on the selected training descriptions, with PAD/BOS/EOS/UNK and up to 4,096
tokens. Unknown rates are reported. Long examples fail with their image ID rather
than being silently truncated. Vocabulary size affects parameter count.

At exactly 4,096 vocabulary entries, the mini reconstruction has:

- Visual encoder: 2,216,384 parameters.
- Projector: 38,496 parameters.
- Decoder: 1,249,440 parameters.
- Total: 3,504,320 parameters in either variant.

This differs from the paper's 5M total and module percentages. In particular a
single 400→96 projection is about 38K parameters, not 14% of a 5M model. The paper
also lists base module percentages summing to 102%. We do not invent undocumented
layers to force a numerical match. Author clarification/code would be needed to
claim the exact original architecture. This project instead isolates a plausible,
explicit pair of interpretations within the permitted smaller pilot scope.

The configuration supports other dimensions. For the paper's base use vision
layers=3, vision_dim=512, text_layers=8, text_dim=128, text_heads=8; for large use
vision_layers=5, vision_dim=512, text_layers=10, text_dim=192, text_heads=16.
Recompute exact counts with `inspect`; these also remain reconstructions.

## Experimental interpretation

The initial 4,500/500 subset and one seed are a pilot. Keep descriptions, subset
IDs, vocabulary, seed, image processing, batch order, optimizer, training budget,
decoder and evaluation settings fixed between encoder runs. Compare loss,
image-ablation gaps, generation, elapsed time and peak GPU memory. Follow up with
paired repeated seeds before interpreting small differences.

The paper performs image-conditioned partial-text completion, not unrestricted
instruction following. Fluent continuation alone does not establish visual
grounding. Evaluation here uses no paid judge and does not reproduce the paper's
five GPT-4o scores. At the initial scale, failure to learn is a diagnostic result,
not proof that either architecture cannot work at a larger data scale.
