# ViZDoom: defend the center

| File | What it does |
|---|---|
| `getting_familiar.ipynb` | first steps with ViZDoom: random play in the *basic* scenario and a first Gymnasium wrapper |
| `defend_centre.ipynb` | the `VizDoomEnv` wrapper for *defend the center*, my reward shaping, PPO training with checkpoints, and a `play()` function that watches a trained model |

**The wrapper:**
- observations are 100 × 160 greyscale frames;
- 3 actions: turn left, turn right, shoot;
- each action is repeated for 3 frames.

**The shaped reward:** the game reward, +0.05 per step alive, and 0.05 × the change in health. A penalty per shot is optional (0 in the final run).

**Training:** PPO with a CNN policy (Stable-Baselines3) on 4 parallel environments, with 2048-step rollouts, batch size 256 and learning rate 1e-5.

**Result:** after 1 M steps (32 minutes), the mean episode length rose from about 120 to 410 steps. See `figures/`.

ViZDoom installs with `pip install vizdoom` and ships the scenario files. In the notebooks, point `config=` at `defend_the_center.cfg` from the installed package.
