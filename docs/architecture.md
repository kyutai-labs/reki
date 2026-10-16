# Architecture

REKI is a vision-language-action (VLA) model with 2.4B parameters. For each observation, a
Qwen3.5-2B vision-language model (VLM) reads the camera images and the instruction. A smaller
action expert (0.2B), a diffusion transformer, then predicts the next second of actions with flow
matching. The model stores a compressed representation of the past in 32 learned register tokens
per frame, whose keys and values are cached in the action expert.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture_dark.png">
  <img src="assets/architecture.png" alt="Left: the VLM encodes views and instruction; the action expert cross-attends to it; state and flow time enter through adaptive RMSNorm; past registers come from a KV cache. Right: one expert block, self-attention over past registers, current registers and actions, then cross-attention, then MLP." width="80%">
</picture>
</p>

| Part | Value |
|---|---|
| Parameters | 2.4B: VLM 2.2B, action expert 0.2B |
| VLM | Qwen3.5-2B, one observation at a time, images up to 448×448 |
| Action expert | 10 layers, width 768, 12 heads, MLP width 3072 |
| Memory | the last 3 frames, 1 s apart, 32 register tokens per frame |
| Robot state | an MLP for each state format (e.g. single-arm end-effector, bimanual joints); conditions the action tokens through adaptive RMSNorm |
| Loss | flow matching on the actions; pretraining also trains the VLM's text output |
| Action chunk | 1 s of actions, 10 denoising steps |
| Prompt | embodiment, instruction, speed, control frequency, control mode |

The model code is in `reki/vlm/qwen3_5/` (VLM and action expert,
[file map](../reki/vlm/qwen3_5/README.md)) and `reki/policy/modeling.py` (flow matching and state
conditioning).

## Model

**Action expert.** The expert has 10 transformer layers. Each layer applies self-attention over the
expert's tokens, cross-attention to the VLM's last-layer outputs for the current observation, and an
MLP. The flow time and the robot state condition the action tokens through adaptive RMSNorm. Actions
are zero-padded to 32 dimensions, and the loss ignores the padding.

**Robot state.** Robots report their state in different formats (e.g. end-effector pose, joint
angles, one or two arms). Each format has its own MLP, whose output joins the flow time in the
adaptive RMSNorm of the action tokens. A sample without state uses a learned null state. The VLM
and the register tokens never see the state.

**Register tokens.** The VLM sees one observation at a time, so the memory lives in the expert. Each
frame adds 32 register tokens, which attend to the registers of the current and past frames. The
action tokens of a frame attend to these registers and to their own chunk. The registers see neither
the actions nor the state, so inference computes them once per frame and caches them.

## Training

**Samples.** A training sample is a window of 3 frames from one episode, 1 s apart. Each frame has
its own action chunk, and the loss averages the flow-matching loss over the 3 chunks. A chunk reads
the registers of its frame and of the earlier frames of the window, so the registers of a frame get
gradient from the losses of all later frames.

**Packing.** The VLM encodes all the observations of a batch in one pass. Their tokens are packed
into one sequence, and attention boundaries keep each observation separate. A GPU samples 30 windows
per forward pass (`batch_size`) and drops windows until the sequence fits in 30,000 tokens
(`max_text_tokens`, image tokens included).

**Noise draws.** The flow loss is averaged over 8 independent draws of the noise and the flow time
per frame. The 8 draws share the VLM pass and the registers of their frame, so only the expert runs
8 times. This lowers the variance of the loss at little cost.

## Inference

A prediction runs the VLM on the new observation, computes its register tokens, then takes 10
denoising steps. The policy keeps the registers of the last 2 frames. Call `RekiPolicy.reset()` at
the start of each episode.

- **Predict cadence.** Predict once per `history_stride_s` (1 s), the spacing of the frames in
  training, and execute 1 s of actions in between. The evaluation, the LeRobot plugin and the
  real-robot server follow this cadence.
- **Compilation.** The model compiles on the first prediction (15–30 s).
- **Cost.** On one H100, one chunk takes 21 ms end to end with compilation (16.5 ms of GPU
  time), and 139 ms without.
