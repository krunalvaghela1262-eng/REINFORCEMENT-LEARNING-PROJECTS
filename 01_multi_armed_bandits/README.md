# Multi-armed bandits

Four exploration strategies, each written as an `Agent` subclass with `give_pull()` and `reinforce()`:

| File | Agent |
|---|---|
| `epsilon_greedy.py` | ε-greedy (explores a random arm with probability ε) |
| `ucb.py` | UCB1: estimate + √(2 ln t / n) |
| `klucb.py` | KL-UCB: largest q with n·KL(p̂, q) ≤ ln t + c·ln ln t, found by a grid search |
| `thompson.py` | Thompson sampling with a Beta(successes + 1, failures + 1) belief per arm |
| `base.py` | the Bernoulli bandit and plotting helpers (course template, see `LICENSE_course_template`) |
| `compare_agents.py` | **S1:** all four agents on arms (0.23, 0.55, 0.76, 0.44) for 30 000 pulls · **S2:** two arms p and p + 0.1 for 17 values of p |

```bash
python compare_agents.py      # S2 by default; uncomment run_agents_and_plot() for S1
python ucb.py                 # any agent file runs on its own and plots its pulls and regret
```

`figures/` holds the plots from my original single runs:
- Figure 1: regret, S1
- Figure 2: reward, S1
- Figure 3: average reward, S1
- Figure 4: optimal-arm pulls, S1
- Figure 5: final regret vs p, S2

Here regret is measured on realised rewards, so it can go negative in a lucky run. The animations in the main README use expected regret, averaged over many runs.
