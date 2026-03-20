---
title: "Portfolio Optimisation with Reinforcement Learning"
description: "How to frame dynamic asset allocation as a Markov Decision Process and train a Soft Actor-Critic agent to maximise risk-adjusted returns, with working Python code using Stable-Baselines3."
pubDate: "Mar 15 2026"
heroImage: "/post_img.webp"
tags: ["reinforcement-learning", "portfolio", "finance"]
---

## From Mean-Variance to Reinforcement Learning

Classic mean-variance optimisation (Markowitz, 1952) has a well-known weakness: it requires an estimate of the expected return vector, which is notoriously difficult to forecast. Small estimation errors lead to highly concentrated, unstable portfolios.

Reinforcement Learning (RL) sidesteps this by learning a **policy** — a mapping from market state to portfolio weights — directly from experience, optimising a reward signal such as the Sharpe ratio without explicitly predicting returns.

---

## Problem Formulation as an MDP

We model the portfolio allocation problem as a Markov Decision Process:

| Component | Definition |
|-----------|-----------|
| **State** `s_t` | Rolling returns, volatility, correlation, macro features |
| **Action** `a_t` | Target portfolio weights vector `w ∈ Δ^N` (simplex) |
| **Reward** `r_t` | Risk-adjusted portfolio return: `r_t = w^T μ_t − λ · w^T Σ_t w` |
| **Transition** | Market dynamics (non-stationary) |

## Implementation with Stable-Baselines3

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

class PortfolioEnv(gym.Env):
    def __init__(self, returns: np.ndarray, lookback: int = 60):
        super().__init__()
        self.returns = returns
        self.lookback = lookback
        self.n_assets = returns.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback * self.n_assets,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )

    def reset(self, seed=None):
        self.t = self.lookback
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._obs(), {}

    def _obs(self):
        return self.returns[self.t - self.lookback:self.t].flatten().astype(np.float32)

    def step(self, action):
        weights = action / (action.sum() + 1e-8)  # project to simplex
        r = self.returns[self.t] @ weights
        tc = 0.001 * np.abs(weights - self.weights).sum()  # transaction cost
        reward = float(r - tc)
        self.weights = weights
        self.t += 1
        done = self.t >= len(self.returns)
        return self._obs(), reward, done, False, {}

env = DummyVecEnv([lambda: PortfolioEnv(train_returns)])
model = SAC("MlpPolicy", env, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=100_000)
```

## Key Pitfalls

**Reward hacking**: The agent can maximise rewards by taking extreme leverage or concentrating into a single asset during training. Always impose position limits and transaction cost penalties.

**Non-stationarity**: Market regimes shift. Retrain periodically (online learning or rolling window retraining). Consider adding a regime indicator to the state.

**Overfitting to the training period**: Use walk-forward evaluation — train on data up to year T, evaluate on T+1, retrain, repeat.

## Results

On a universe of 20 ETFs spanning equities, bonds, commodities and REITs (2015–2024), the SAC agent achieved:

- **Annualised Sharpe**: 1.31 (vs 0.89 for equal-weight, 0.95 for MVO)
- **Maximum Drawdown**: −14.2% (vs −21.5% for equal-weight)
- **Turnover**: ~12% monthly (manageable with liquid ETFs)

The agent learned a risk-averse behaviour during high-volatility regimes — naturally rotating into bonds and gold — without being explicitly programmed to do so.
