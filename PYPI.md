# REKI: Registering Knowledge in VLAs

**A 2.4B vision-language-action model pretrained on public data only, with a memory of its recent past at negligible cost.**

[Paper](https://arxiv.org/abs/2610.22178) · [Project page](https://kyutai.org/reki/) ·
[Code](https://github.com/kyutai-labs/reki) ·
[Models](https://huggingface.co/collections/kyutai/reki-6ac371366b3da8cdece73ce2)

## Installation

Requires Linux x86-64, Python 3.12 and an NVIDIA GPU of the Ampere generation or newer with
driver 570 or newer (CUDA 12.8). Triton compiles kernels at run time: the system also needs a C
compiler (gcc or clang) and the Python 3.12 headers.

```bash
pip install reki-vla    # or: uv add reki-vla
```

## Quickstart

```python
from reki import RekiPolicy

policy = RekiPolicy.from_pretrained("kyutai/reki-2b-libero", device="cuda")
policy.reset()                    # at the start of every episode: clears the register memory
actions = policy.predict(
    images={"camera_top": img0, "camera_wrist_left": img1},  # HxWx3 uint8, RGB
    state=proprio,                # training dataset's observation.state, float [policy.spec.state_dim]
    task="put the bowl on the plate",
)                                 # float32 np.ndarray [policy.n_action_steps, action_dim], unnormalized
```

- Call `predict` once every `policy.n_action_steps` environment steps (1 s for all released
  models), and execute the returned actions in between.
- `policy.spec.camera_keys` and `policy.spec.state_layout` give the expected cameras and the
  state order. The images must have the orientation of the training data (for LIBERO, rotate the
  env images by 180°).

Evaluation, training and real-robot deployment use the repository: see the
[README](https://github.com/kyutai-labs/reki#readme).

## License

- Code: MIT. Two files and parts of one file come from LeRobot, and parts of three files from
  Transformers, under Apache-2.0 (`NOTICE` and `LICENSE-APACHE` in the package).
- Weights: [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The weights
  derive from Qwen3.5-2B and third-party datasets. Each keeps its own license.
