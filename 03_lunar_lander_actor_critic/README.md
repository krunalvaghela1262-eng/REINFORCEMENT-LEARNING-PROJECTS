# Lunar Lander: actor-critic

[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) has discrete actions and no wind. An episode counts as solved at a reward of 200 or more.

| File | What it does |
|---|---|
| `simple_AC.py` | one-step actor-critic with **hand-written eligibility traces**: the policy gradients are accumulated and applied manually |
| `optimized.py` | batched A2C: 16 synchronous environments on the GPU, 32-step batches, Adam, gradient clipping. It stops once the recent average reaches 200 |
| `key_control.py` | play the lander yourself with the arrow keys (adapted from Gymnasium's example) |
| `checkpoints/vectorized_checkpoint_20250701-193850.pt` | the trained policy and value networks, plus the reward of every training episode |

```bash
python optimized.py                          # train
python ../docs/make_figures.py lander        # fly the saved checkpoint and redraw the animation
```

`figures/` holds two training runs. One stopped when the last 100 episodes averaged 200. The other, which is the saved checkpoint, stopped when the last 500 episodes did.
