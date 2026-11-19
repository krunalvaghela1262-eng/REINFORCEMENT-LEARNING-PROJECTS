"""Animations and figures for the README.     python docs/make_figures.py [banner|bandits|football|lander|doom]

  banner.png          title banner
  bandit_beliefs.gif  what UCB and Thompson sampling believe about each arm as pulls accumulate
  bandit_race.gif     the four agents side by side: pulls per arm and regret over time
  football.gif        one 2-vs-1 episode played out under the game's rules (02_football_2v1_td0/definitions.py)
  lunar_lander.gif    the trained actor-critic checkpoint flying LunarLander-v3, with its action probabilities
  lander_training.png reward per training episode, read from the same checkpoint
  doom_pipeline.gif   ViZDoom "defend the center": the game frame and the 100x160 greyscale image the agent sees

The bandit animations import the agents in 01_multi_armed_bandits unchanged; the lander animation loads
03_lunar_lander_actor_critic/checkpoints/vectorized_checkpoint_20250701-193850.pt.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BG, PANEL, FG, MUTED, GRID = "#0b1020", "#111827", "#f3f4f6", "#9ca3af", "#1f2937"
AGENT_COL = {"ε-greedy": "#60a5fa", "UCB": "#f59e0b", "KL-UCB": "#34d399", "Thompson": "#f472b6"}
ARMS = np.array([0.23, 0.55, 0.76, 0.44])          # the 4-armed bandit used in compare_agents.py
warnings.filterwarnings("ignore")


def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def grab(fig):
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    return img


def save_gif(frames, name, dur, colors=128):
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    if isinstance(dur, (int, float)):
        dur = [dur] * len(frames)
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)


# ===================================================================================== bandits
def run_agents(T, seed):
    sys.path.insert(0, str(ROOT / "01_multi_armed_bandits"))
    from base import MultiArmedBandit
    from epsilon_greedy import EpsilonGreedyAgent
    from klucb import KLUCBAgent
    from thompson import ThompsonSamplingAgent
    from ucb import UCBAgent
    np.random.seed(seed)
    agents = {
        "ε-greedy": EpsilonGreedyAgent(T, MultiArmedBandit(ARMS), epsilon=0.01),
        "UCB": UCBAgent(T, MultiArmedBandit(ARMS)),
        "KL-UCB": KLUCBAgent(T, MultiArmedBandit(ARMS), c=3),
        "Thompson": ThompsonSamplingAgent(T, MultiArmedBandit(ARMS)),
    }
    gap = ARMS.max() - ARMS
    hist = {k: dict(counts=[], means=[], s=[], f=[], regret=[0.0]) for k in agents}
    for t in range(T):
        for k, a in agents.items():
            before = a.count_memory.copy()
            a.give_pull()
            arm = int(np.argmax(a.count_memory - before))
            h = hist[k]
            h["regret"].append(h["regret"][-1] + gap[arm])          # expected regret of the arm it pulled
            h["counts"].append(a.count_memory.copy())
            with np.errstate(all="ignore"):
                h["means"].append(np.where(a.count_memory > 0, a.reward_memory / np.maximum(a.count_memory, 1), np.nan))
            if k == "Thompson":
                h["s"].append(a.success_mem.copy()); h["f"].append(a.fail_mem.copy())
    return hist


def bandit_gifs(T=3000, runs=12):
    print("running the four agents ...")
    hists = [run_agents(T, seed) for seed in range(runs)]
    h0 = hists[0]
    reg = {k: np.mean([h[k]["regret"] for h in hists], axis=0) for k in AGENT_COL}
    ticks = np.unique(np.round(np.geomspace(4, T, 70)).astype(int))

    # ---------------------------------------------------------------- beliefs: UCB bounds and Thompson posteriors
    from scipy.stats import beta
    xs = np.linspace(0, 1, 400)
    arm_col = ["#94a3b8", "#60a5fa", "#22c55e", "#f59e0b"]
    frames = []
    for t in ticks:
        fig = plt.figure(figsize=(12, 4.6), dpi=90, facecolor=BG)
        fig.text(0.5, 0.94, f"What the agent believes after {t:,} pulls", color=FG, ha="center", fontsize=15, weight="bold")
        ax = fig.add_axes([0.05, 0.13, 0.42, 0.68]); style(ax)
        n = h0["UCB"]["counts"][t - 1]; m = np.nan_to_num(h0["UCB"]["means"][t - 1])
        bonus = np.sqrt(2 * np.log(max(t, 1)) / np.maximum(n, 1))
        for a in range(4):
            lo, hi = max(m[a] - bonus[a], 0), min(m[a] + bonus[a], 1.05)
            ax.add_patch(Rectangle((a - 0.18, lo), 0.36, hi - lo, fc=arm_col[a], alpha=0.35, ec=arm_col[a]))
            ax.plot([a - 0.18, a + 0.18], [m[a]] * 2, color=arm_col[a], lw=3)
            ax.plot([a - 0.3, a + 0.3], [ARMS[a]] * 2, color=FG, lw=1, ls=(0, (3, 2)))
            ax.text(a, 1.08, f"{int(n[a])} pulls", color=MUTED, ha="center", fontsize=9)
        best = int(np.argmax(m + bonus))
        ax.annotate("next pull", xy=(best, min(m[best] + bonus[best], 1.05)), xytext=(best, 1.2), color=FG, ha="center",
                    fontsize=9, arrowprops=dict(arrowstyle="-|>", color=FG))
        ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 1.3); ax.set_xticks(range(4), [f"arm {a + 1}\np={ARMS[a]}" for a in range(4)])
        ax.set_title("UCB: estimate  ±  confidence bonus", fontsize=12)
        ax.text(3.55, 0.03, "dashed = true payout rate", color=MUTED, ha="right", fontsize=8)
        ax2 = fig.add_axes([0.55, 0.13, 0.42, 0.68]); style(ax2)
        s, f = h0["Thompson"]["s"][t - 1], h0["Thompson"]["f"][t - 1]
        ymax = 0
        for a in range(4):
            d = beta.pdf(xs, s[a] + 1, f[a] + 1); ymax = max(ymax, d.max())
            ax2.fill_between(xs, d, color=arm_col[a], alpha=0.3); ax2.plot(xs, d, color=arm_col[a], lw=2, label=f"arm {a + 1}")
            ax2.axvline(ARMS[a], color=arm_col[a], lw=1, ls=(0, (3, 2)))
        ax2.set_ylim(0, ymax * 1.1); ax2.set_xlim(0, 1); ax2.set_yticks([])
        ax2.set_xlabel("payout rate"); ax2.set_title("Thompson sampling: Beta belief per arm", fontsize=12)
        ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper left")
        frames.append(grab(fig)); plt.close(fig)
    save_gif(frames, "bandit_beliefs.gif", [120] * (len(frames) - 1) + [2500])

    # ---------------------------------------------------------------- race: pulls per arm + regret curves
    frames = []
    for t in ticks:
        fig = plt.figure(figsize=(12, 5.4), dpi=90, facecolor=BG)
        fig.text(0.5, 0.94, f"Four agents, same 4-armed bandit  ·  pull {t:,} / {T:,}", color=FG, ha="center",
                 fontsize=15, weight="bold")
        for i, (k, c) in enumerate(AGENT_COL.items()):
            ax = fig.add_axes([0.05 + i * 0.118, 0.14, 0.1, 0.66]); style(ax)
            cnt = h0[k]["counts"][t - 1]
            ax.bar(range(4), cnt / t, color=[c if a == 2 else "#374151" for a in range(4)])
            ax.set_ylim(0, 1); ax.set_xticks(range(4), ["1", "2", "3", "4"]); ax.set_title(k, fontsize=11, color=c)
            if i == 0:
                ax.set_ylabel("share of pulls per arm")
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_xlabel("arm (3 is best)")
        ax = fig.add_axes([0.56, 0.14, 0.41, 0.66]); style(ax)
        for k, c in AGENT_COL.items():
            ax.plot(np.arange(t + 1), reg[k][:t + 1], color=c, lw=2.2, label=k)
            ax.plot(t, reg[k][t], "o", color=c, ms=6)
        ax.set_xlim(0, T); ax.set_ylim(0, max(r[-1] for r in reg.values()) * 1.08)
        ax.set_xlabel("pulls"); ax.set_ylabel("regret (reward lost vs. always best arm)")
        ax.set_title(f"regret, averaged over {runs} runs", fontsize=11)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper left")
        frames.append(grab(fig)); plt.close(fig)
    save_gif(frames, "bandit_race.gif", [110] * (len(frames) - 1) + [3000])
    print({k: round(float(r[-1]), 1) for k, r in reg.items()})


# ===================================================================================== football
def football_gif():
    """One episode from the start state [B1=5, B2=9, R=8, ball=B1], drawn with the probabilities the
    environment in definitions.py uses (p = 0.1, q = 0.7)."""
    p, q = 0.1, 0.7
    xy = lambda sq: ((sq - 1) % 4, (sq - 1) // 4)
    steps = [  # (B1, B2, R, ball, caption, kind, success probability of this step)
        (5, 9, 8, 1, "Start: B1 has the ball, R guards the goal.\nR moves every step too.", None, 1),
        (6, 9, 12, 1, "B1 dribbles right: keeps the ball with\nprob 1 − 2p = 0.8", "move", 1 - 2 * p),
        (6, 9, 11, 2, "Pass B1 → B2: q − 0.1·distance = 0.6\n(R is not on the passing line)", "pass", q - 0.1),
        (7, 9, 15, 2, "B1 runs into space without the ball:\n1 − p = 0.9", "move", 1 - p),
        (7, 9, 14, 1, "Long pass back to B1:\nq − 0.1·2 = 0.5", "pass", q - 0.2),
        (8, 9, 10, 1, "B1 dribbles into the last column: 0.8", "move", 1 - 2 * p),
        (8, 9, 10, 1, "Shoot from the last column:\nq − 0.2·(3 − 3) = 0.7", "shoot", q),
        (8, 9, 10, 1, "GOAL  +20", "goal", 1),
    ]
    frames, durs, chance = [], [], 1.0
    for i, (b1, b2, r, ball, cap, kind, ps) in enumerate(steps):
        chance *= ps
        for sub in range(6 if kind in ("pass", "shoot") else 1):
            fig = plt.figure(figsize=(10, 5.2), dpi=90, facecolor=BG)
            ax = fig.add_axes([0.03, 0.08, 0.5, 0.84]); ax.set_facecolor(BG); ax.axis("off")
            for x in range(4):
                for y in range(4):
                    ax.add_patch(Rectangle((x, y), 1, 1, fc="#14532d" if (x + y) % 2 else "#166534", ec="#22c55e", lw=0.6))
                    ax.text(x + 0.08, y + 0.14, str(1 + x + 4 * y), color="#86efac", fontsize=8, alpha=0.7)
            ax.add_patch(Rectangle((4, 1), 0.12, 2, fc="white"))
            for sq in (8, 12):
                gx, gy = xy(sq); ax.add_patch(Rectangle((gx, gy), 1, 1, fc="none", ec="#fbbf24", lw=1.2, ls="--"))
            pos = {"B1": xy(b1), "B2": xy(b2), "R": xy(r)}
            for name, (x, y) in pos.items():
                col = "#ef4444" if name == "R" else "#3b82f6"
                ax.add_patch(Circle((x + 0.5, y + 0.5), 0.3, fc=col, ec="white", lw=1.5))
                ax.text(x + 0.5, y + 0.5, name, color="white", ha="center", va="center", fontsize=12, weight="bold")
            holder = pos["B1" if ball == 1 else "B2"]
            bx, by = holder[0] + 0.5, holder[1] + 0.5
            if kind == "pass" and sub < 5:
                prev = steps[i - 1][3]
                src = pos["B1" if prev == 1 else "B2"]
                f = sub / 5
                bx = src[0] + 0.5 + f * (holder[0] - src[0]); by = src[1] + 0.5 + f * (holder[1] - src[1])
                ax.plot([src[0] + 0.5, holder[0] + 0.5], [src[1] + 0.5, holder[1] + 0.5], color="white", ls=":", lw=1.5)
            if kind == "shoot":
                f = sub / 5
                bx = bx + f * (4.05 - bx); by = by + f * (2.0 - by)
                ax.plot([holder[0] + 0.5, 4.05], [holder[1] + 0.5, 2.0], color="#fbbf24", ls=":", lw=1.5)
            if kind == "goal":
                bx, by = 4.05, 2.0
                ax.text(2, -0.35, "GOAL!", color="#fbbf24", ha="center", fontsize=22, weight="bold")
            ax.add_patch(Circle((bx + 0.22 if kind not in ("shoot", "goal") else bx, by + 0.22 if kind not in ("shoot", "goal") else by),
                                0.11, fc="white", ec="black", lw=1, zorder=5))
            ax.set_xlim(-0.1, 4.3); ax.set_ylim(4.1, -0.6); ax.set_aspect("equal")
            t = fig.add_axes([0.56, 0.08, 0.42, 0.84]); t.axis("off")
            t.text(0, 0.95, "2 vs 1 football shoot-out", color=FG, fontsize=16, weight="bold", transform=t.transAxes)
            t.text(0, 0.86, f"step {i}", color=MUTED, fontsize=11, transform=t.transAxes)
            t.text(0, 0.78, cap, color="#fde68a" if kind == "goal" else FG, fontsize=12, va="top", transform=t.transAxes)
            t.text(0, 0.58, f"chance this whole sequence works so far: {chance:.0%}", color="#fbbf24", fontsize=10.5,
                   transform=t.transAxes)
            rules = ("state   (B1, B2, R squares, ball holder)\n"
                     "actions move B1 / B2 (4 each), pass, shoot\n\n"
                     "move with ball  1 − 2p    (0.5 − p into R)\n"
                     "pass            q − 0.1·distance\n"
                     "                (halved if R on the line)\n"
                     "shoot           q − 0.2·(3 − x)\n"
                     "                (halved if R guards goal)\n\n"
                     "reward  +20 goal, −1 ball lost or missed\n"
                     "learn   V(s) ← V(s) + α[r + V(s′) − V(s)]")
            t.text(0, 0.02, rules, color=MUTED, fontsize=9.5, family="monospace", transform=t.transAxes, va="bottom")
            frames.append(grab(fig)); plt.close(fig)
            durs.append(90 if (kind in ("pass", "shoot") and sub < 5) else (2600 if kind == "goal" else 1400))
    save_gif(frames, "football.gif", durs, colors=96)


# ===================================================================================== lunar lander
def lander_policy():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MyNN(nn.Module):                          # same network as in optimized.py
        def __init__(self, i, o):
            super().__init__()
            self.fc1, self.fc2, self.fc3 = nn.Linear(i, 128), nn.Linear(128, 128), nn.Linear(128, o)

        def forward(self, x):
            return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    ck = torch.load(ROOT / "03_lunar_lander_actor_critic/checkpoints/vectorized_checkpoint_20250701-193850.pt",
                    map_location="cpu", weights_only=False)
    pol = MyNN(8, 4); pol.load_state_dict(ck["policy_state_dict"]); pol.eval()
    return pol, ck


def lander_gif(seeds=(24, 21), eval_episodes=100):
    import gymnasium as gym
    import torch
    pol, ck = lander_policy()
    probs_of = lambda s: torch.softmax(pol(torch.tensor(s, dtype=torch.float32)), -1).detach().numpy()
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, render_mode="rgb_array")

    # evaluation over fresh start positions (greedy = most likely action)
    scores = []
    for seed in range(1000, 1000 + eval_episodes):
        s, _ = env.reset(seed=seed); R = 0
        while True:
            s, r, te, tr, _ = env.step(int(np.argmax(probs_of(s)))); R += r
            if te or tr:
                break
        scores.append(R)
    scores = np.array(scores)
    print(f"evaluation over {eval_episodes} episodes: mean {scores.mean():.0f}, >=200 in {(scores >= 200).mean():.0%}")

    # training curve from the checkpoint
    perf = np.array(ck["performance"]); k = 50
    sm = np.convolve(perf, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=100, facecolor=BG); style(ax)
    ax.plot(perf, color="#f59e0b", alpha=0.18, lw=0.6, label="episode reward")
    ax.plot(np.arange(k - 1, len(perf)), sm, color="#f59e0b", lw=2, label=f"{k}-episode average")
    ax.axhline(200, color="#22c55e", ls="--", lw=1.2); ax.text(len(perf) * 0.01, 212, "solved (200)", color="#22c55e", fontsize=9)
    ax.set_xlabel("training episode (16 environments in parallel)"); ax.set_ylabel("reward")
    ax.set_xlim(0, len(perf)); ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "lander_training.png", facecolor=BG); plt.close(fig)

    names = ["do nothing", "left engine", "main engine", "right engine"]
    cols = ["#e5e7eb", "#60a5fa", "#f59e0b", "#f472b6"]
    frames = []
    for ep, seed in enumerate(seeds):
        s, _ = env.reset(seed=seed); R, step = 0.0, 0
        while True:
            pr = probs_of(s); a = int(np.argmax(pr))
            if step % 3 == 0:
                img = env.render()
                fig = plt.figure(figsize=(10.5, 4.6), dpi=80, facecolor=BG)
                ax = fig.add_axes([0.01, 0.03, 0.6, 0.86]); ax.imshow(img); ax.axis("off")
                fig.text(0.31, 0.93, f"trained policy  ·  landing {ep + 1} of {len(seeds)}", color=FG, ha="center",
                         fontsize=13, weight="bold")
                b = fig.add_axes([0.66, 0.3, 0.31, 0.55]); style(b)
                b.barh(range(4), pr, color=cols, alpha=None)
                for i, bar in enumerate(b.patches):
                    bar.set_alpha(1.0 if i == a else 0.35)
                b.set_yticks(range(4), names); b.invert_yaxis(); b.set_xlim(0, 1)
                b.set_title("policy network: action probabilities", fontsize=10)
                fig.text(0.66, 0.15, f"reward so far  {R:7.1f}", color=FG, fontsize=13, family="monospace")
                fig.text(0.66, 0.07, f"step {step}", color=MUTED, fontsize=10, family="monospace")
                frames.append(grab(fig)); plt.close(fig)
            s, r, te, tr, _ = env.step(a); R += r; step += 1
            if te or tr:
                break
        for _ in range(12):
            frames.append(frames[-1])
        print(f"seed {seed}: reward {R:.0f}, {step} steps")
    save_gif(frames, "lunar_lander.gif", 70, colors=96)
    return scores


# ===================================================================================== vizdoom
def doom_gif(steps=150, tics=3):
    """The observation pipeline of 04_vizdoom_defend_the_center: game frame -> greyscale -> 100x160 image, and the
    shaped reward from VizDoomEnv.step(). The actions here come from a simple scripted turn-and-shoot loop
    (the trained PPO weights were not kept), so this shows what the agent sees and how it is scored."""
    import cv2
    import vizdoom as vzd
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "defend_the_center.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.set_seed(7)
    game.init(); game.new_episode()
    acts = np.identity(3, dtype=np.uint8)            # TURN_LEFT, TURN_RIGHT, ATTACK (as in the notebook)
    names = ["turn left", "turn right", "shoot"]
    frames, total, kills, step = [], 0.0, 0, 0
    while not game.is_episode_finished() and step < steps:
        st = game.get_state()
        rgb = np.moveaxis(st.screen_buffer, 0, -1) if st.screen_buffer.shape[0] == 3 else st.screen_buffer
        grey = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (160, 100), interpolation=cv2.INTER_CUBIC)
        a = 2 if step % 3 == 0 else 0
        ammo, hp = st.game_variables[0], st.game_variables[1]
        r_game = game.make_action(acts[a].tolist(), tics)
        if game.is_episode_finished():
            new_hp = hp
        else:
            new_hp = game.get_state().game_variables[1]
        r_health = (new_hp - hp) * 0.05
        r_alive = 0.05 if not game.is_episode_finished() else 0.0
        kills += int(r_game > 0)
        total += r_game + r_health + r_alive
        fig = plt.figure(figsize=(11, 4.4), dpi=66, facecolor=BG)
        ax = fig.add_axes([0.01, 0.2, 0.47, 0.7]); ax.imshow(rgb); ax.axis("off")
        ax.set_title("game frame (640×480 colour)", color=MUTED, fontsize=10)
        ax2 = fig.add_axes([0.52, 0.2, 0.3, 0.7]); ax2.imshow(grey, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("what the CNN policy sees (100×160 grey)", color=MUTED, fontsize=10)
        for sp in ax2.spines.values(): sp.set_color(GRID)
        fig.text(0.5, 0.94, "ViZDoom · defend the center", color=FG, ha="center", fontsize=14, weight="bold")
        led = (f"action      {names[a]}\n"
               f"game        {r_game:+.2f}\n"
               f"health Δ    {r_health:+.2f}\n"
               f"alive       {r_alive:+.2f}\n"
               f"──────────────────\n"
               f"total       {total:+.1f}\n"
               f"kills {kills}   hp {int(new_hp)}")
        fig.text(0.835, 0.3, led, color=FG, fontsize=9.5, family="monospace")
        fig.text(0.835, 0.72, "shaped reward", color="#fbbf24", fontsize=11, weight="bold")
        fig.text(0.5, 0.06, "Scripted turn-and-shoot actions, to show the observation and reward pipeline.",
                 color=MUTED, ha="center", fontsize=9)
        if step % 2 == 0:
            frames.append(grab(fig))
        plt.close(fig); step += 1
    game.close()
    save_gif(frames, "doom_pipeline.gif", 150, colors=64)
    print("doom frames", len(frames), "kills", kills)


# ===================================================================================== banner
def banner():
    fig = plt.figure(figsize=(16, 4), dpi=100, facecolor=BG)
    t = fig.add_axes([0, 0, 0.5, 1]); t.axis("off")
    t.text(0.06, 0.62, "Reinforcement learning", color="white", fontsize=32, weight="bold", transform=t.transAxes)
    t.text(0.06, 0.45, "from slot machines to Doom, one idea at a time", color="#e5e7eb", fontsize=16, transform=t.transAxes)
    t.text(0.06, 0.3, "UCB · KL-UCB · Thompson  ·  TD(0)  ·  actor-critic / A2C  ·  PPO", color=MUTED, fontsize=12,
           transform=t.transAxes)
    stops = [("bandits", "#60a5fa"), ("football MDP", "#22c55e"), ("Lunar Lander", "#f59e0b"), ("ViZDoom", "#ef4444")]
    ax = fig.add_axes([0.5, 0.15, 0.48, 0.7]); ax.axis("off"); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1, 1)
    ax.plot([0, 3], [0, 0], color=GRID, lw=4, zorder=0)
    for i, (n, c) in enumerate(stops):
        ax.add_patch(Circle((i, 0), 0.32, fc=c, ec="white", lw=2))
        ax.text(i, 0, str(i + 1), color="white", ha="center", va="center", fontsize=18, weight="bold")
        ax.text(i, -0.62, n, color=FG, ha="center", fontsize=13)
    ax.set_aspect("equal")
    fig.savefig(OUT / "banner.png", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["banner", "bandits", "football", "lander"]
    if "banner" in todo: banner()
    if "football" in todo: football_gif()
    if "lander" in todo: lander_gif()
    if "doom" in todo: doom_gif()
    if "bandits" in todo: bandit_gifs()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:24s} {f.stat().st_size / 1e6:6.2f} MB")
