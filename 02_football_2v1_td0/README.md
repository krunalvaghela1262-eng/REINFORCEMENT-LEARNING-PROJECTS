# 2-vs-1 football shoot-outs

The brief is in [`problem_description.html`](problem_description.html). Open it in a browser, next to the `imgs/` folder. It came with no starter code. Everything here is my own design:

| File | What it does |
|---|---|
| `definitions.py` | `State` (B1, B2, R squares and the ball holder), the transition and reward function `take_action()`, and a TD(0) episode runner |
| `opponent.py` | three defenders: random, greedy (chases the ball) and defensive (guards the goal zone) |
| `algorithms/finalattempt.py` | the final learner. It uses TD(0) for V(s), simulator look-ahead to choose actions, and an ε that decays exponentially |
| `main.py` | asks for a start state and an opponent, then sweeps p and q and plots the win probability |

```bash
python main.py        # enter d for the default start [5, 9, 8, 1], then r / g / d for the opponent
```

`figures/` holds the win probability of the start state [5, 9, 8, 1] against p (with q = 0.7) and against q (with p = 0.3). It is computed as (1 + V(start)) / 21, since a goal is worth +20 and a loss −1.
