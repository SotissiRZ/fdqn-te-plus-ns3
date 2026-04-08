"""
baseline_agents.py — Agents de comparaison pour FDQN-TE+
=========================================================
Implémente 3 baselines :
  1. DQN Standard          — pas de Double DQN, pas de PEPM, pas de fédération
  2. FDQN sans PEPM/LSTM   — Double DQN + IFO, mais sans prédiction énergétique
  3. FDQN sans Fédération  — Double DQN + PEPM, mais sans FedMeta-DRL

Chacun hérite de la même infrastructure réseau (QNetwork, ReplayMemory)
mais désactive les modules concernés.

Utilisé par evaluation.py pour la comparaison équitable.
"""

import numpy as np
import math
import random
from collections import deque
from typing import List, Tuple, Optional

try:
    from fdqn_config import FdqnConfig as _Cfg
    _STATE_DIM    = _Cfg.STATE_DIM
    _MAX_ACTIONS  = _Cfg.MAX_NEIGHBORS
    _GAMMA        = _Cfg.GAMMA
    _LR           = _Cfg.LR
    _EPS_MAX      = _Cfg.EPSILON_MAX
    _EPS_MIN      = _Cfg.EPSILON_MIN
    _EPS_DELTA    = _Cfg.EPSILON_DELTA
    _REPLAY       = _Cfg.REPLAY_SIZE
    _BATCH        = _Cfg.BATCH_SIZE
    _TARGET_UPD   = _Cfg.TARGET_UPDATE
    _LAMBDA_PDR   = _Cfg.LAMBDA_PDR
    _LAMBDA_E     = _Cfg.LAMBDA_ENERGY
    _LAMBDA_D     = _Cfg.LAMBDA_DELAY
    _LAMBDA_S     = _Cfg.LAMBDA_SAFE
except ImportError:
    _STATE_DIM = 10; _MAX_ACTIONS = 12; _GAMMA = 0.99; _LR = 3e-4
    _EPS_MAX = 0.9; _EPS_MIN = 0.1; _EPS_DELTA = 0.002
    _REPLAY = 10000; _BATCH = 64; _TARGET_UPD = 100
    _LAMBDA_PDR = 0.45; _LAMBDA_E = 0.20; _LAMBDA_D = 0.15; _LAMBDA_S = 0.10


# ─── Infrastructure commune ───────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation="relu"):
        scale = math.sqrt(2.0 / in_dim)
        self.W = scale * np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None
        self._z = None
        self.m_W = np.zeros_like(self.W); self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b); self.v_b = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        z = x @ self.W + self.b
        self._z = z
        return np.maximum(0, z) if self.activation == "relu" else z

    def backward(self, g):
        if self.activation == "relu":
            g = g * (self._z > 0).astype(float)
        self.dW = self._x.T @ g
        self.db = g.sum(axis=0)
        return g @ self.W.T

    def adam_update(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m_W = beta1*self.m_W + (1-beta1)*self.dW
        self.v_W = beta2*self.v_W + (1-beta2)*self.dW**2
        self.W  -= lr * (self.m_W/(1-beta1**t)) / (np.sqrt(self.v_W/(1-beta2**t)) + eps)
        self.m_b = beta1*self.m_b + (1-beta1)*self.db
        self.v_b = beta2*self.v_b + (1-beta2)*self.db**2
        self.b  -= lr * (self.m_b/(1-beta1**t)) / (np.sqrt(self.v_b/(1-beta2**t)) + eps)

    def copy_weights(self, other):
        self.W = other.W.copy(); self.b = other.b.copy()


class QNet:
    def __init__(self, state_dim, n_actions, hidden=(128, 64, 32)):
        dims = [state_dim] + list(hidden) + [n_actions]
        self.layers = [
            DenseLayer(dims[i], dims[i+1],
                       activation="relu" if i < len(dims)-2 else "linear")
            for i in range(len(dims)-1)
        ]

    def forward(self, x):
        x = x.reshape(1, -1) if x.ndim == 1 else x
        for l in self.layers: x = l.forward(x)
        return x.squeeze()

    def backward(self, grad):
        g = grad if grad.ndim == 2 else grad.reshape(1, -1)
        for l in reversed(self.layers): g = l.backward(g)

    def adam_step(self, lr, t, clip=1.0):
        norm = sum(float(np.sum(l.dW**2)+np.sum(l.db**2)) for l in self.layers)**0.5
        if norm > clip:
            s = clip / (norm + 1e-8)
            for l in self.layers: l.dW *= s; l.db *= s
        for l in self.layers: l.adam_update(lr, t)

    def copy_from(self, other):
        for a, b in zip(self.layers, other.layers): a.copy_weights(b)

    def get_params(self):
        return [[l.W.copy(), l.b.copy()] for l in self.layers]

    def set_params(self, params):
        for l, p in zip(self.layers, params): l.W = p[0].copy(); l.b = p[1].copy()


class ReplayBuffer:
    def __init__(self, cap=_REPLAY):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, done):
        self.buf.append((np.array(s, np.float32), int(a), float(r),
                         np.array(s2, np.float32), bool(done)))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, np.float32),
                np.array(s2), np.array(d, np.float32))

    def __len__(self): return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DQN Standard (ni Double-DQN, ni PEPM, ni fédération)
# ─────────────────────────────────────────────────────────────────────────────

class StandardDQNAgent:
    """
    DQN classique (Mnih et al. 2015).
    - Pas de Double-DQN  → target = r + γ·max Q_target(s',·)
    - Pas de PEPM        → état réduit à 5 dimensions de base
    - Pas de fédération  → apprentissage local uniquement
    - Récompense simple  : λ_PDR·pdr - λ_E·cost - λ_D·delay  (sans terme sécurité)
    """
    NAME = "DQN Standard"

    def __init__(self, node_id: int, state_dim: int = 5,
                 n_actions: int = _MAX_ACTIONS,
                 gamma=_GAMMA, lr=_LR,
                 eps_max=_EPS_MAX, eps_min=_EPS_MIN, eps_delta=_EPS_DELTA,
                 replay=_REPLAY, batch=_BATCH, target_upd=_TARGET_UPD):
        self.node_id  = node_id
        self.gamma    = gamma
        self.lr       = lr
        self.epsilon  = eps_max
        self.eps_min  = eps_min
        self.eps_delta= eps_delta
        self.batch    = batch
        self.target_upd = target_upd
        self.step     = 0

        # Un seul réseau « online » (pas de séparation sélection/évaluation)
        self.online = QNet(state_dim, n_actions)
        self.target = QNet(state_dim, n_actions)
        self.target.copy_from(self.online)

        self.memory = ReplayBuffer(replay)
        self.total_reward = 0.0
        self.loss_history = deque(maxlen=200)

    def select_action(self, state: np.ndarray, n_neighbors: int) -> Tuple[int, float, bool]:
        """ε-greedy basique — pas de boost adaptatif."""
        n_actions = self.online.layers[-1].W.shape[1]
        n = min(n_neighbors, n_actions)
        q = self.online.forward(state)
        q = q[:n] if q.ndim == 1 else q.flatten()[:n]
        if random.random() < self.epsilon:
            idx = random.randint(0, n-1)
            return idx, float(q[min(idx, len(q)-1)]), True
        idx = int(np.argmax(q))
        return idx, float(q[idx]), False

    def store(self, s, a, r, s2, done=False):
        self.memory.push(s, a, r, s2, done)
        self.total_reward += r

    def learn(self) -> float:
        if len(self.memory) < self.batch:
            return 0.0
        s, a, r, s2, d = self.memory.sample(self.batch)
        B = len(s)

        # === DQN classique : max sur le réseau CIBLE (pas Double-DQN) =========
        q_next = self.target.forward(s2)
        if q_next.ndim == 1: q_next = q_next.reshape(B, -1)
        targets = r + self.gamma * q_next.max(axis=1) * (1 - d)

        q_all = self.online.forward(s)
        if q_all.ndim == 1: q_all = q_all.reshape(B, -1)
        a_c = np.clip(a, 0, q_all.shape[1]-1)
        current_q = q_all[np.arange(B), a_c]

        td = targets - current_q
        loss = float(np.mean(td**2))

        grad = np.zeros_like(q_all)
        grad[np.arange(B), a_c] = -2.0 * td / B
        self.online.backward(grad)
        self.step += 1
        self.online.adam_step(self.lr, self.step)

        # Hard copy target
        if self.step % self.target_upd == 0:
            self.target.copy_from(self.online)

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_delta)
        self.loss_history.append(loss)
        return loss

    @staticmethod
    def compute_reward(pdr: float, energy_cost: float, delay_norm: float) -> float:
        """Récompense sans terme sécurité PEPM."""
        return (_LAMBDA_PDR * (1.0 if pdr > 0.9 else -1.0)
                + _LAMBDA_E * (1.0 - energy_cost)
                - _LAMBDA_D * delay_norm)

    def build_state(self, energy_norm: float, dist_sink_norm: float,
                    n_neighbors_norm: float, is_ch: float,
                    tx_norm: float) -> np.ndarray:
        """État réduit 5D (sans PEPM risk, sans métriques voisins)."""
        return np.array([energy_norm, dist_sink_norm, n_neighbors_norm,
                         is_ch, tx_norm], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FDQN sans PEPM (Double DQN + IFO mais sans prédiction énergétique)
# ─────────────────────────────────────────────────────────────────────────────

class FDQNNoPEPMAgent:
    """
    FDQN sans PEPM/LSTM.
    - Double-DQN    ✓
    - Fédération    ✓  (FedAvg basique — pas de méta-gradient)
    - PEPM          ✗  → risque PEPM figé à 0 dans l'état
    - Récompense    sans terme sécurité (LAMBDA_SAFE ignoré)
    - État 9D       (sans predictedRisk, dimension 2 de BuildState)
    """
    NAME = "FDQN sans PEPM"

    def __init__(self, node_id: int, state_dim: int = 9,
                 n_actions: int = _MAX_ACTIONS,
                 gamma=_GAMMA, lr=_LR,
                 eps_max=_EPS_MAX, eps_min=_EPS_MIN, eps_delta=_EPS_DELTA,
                 replay=_REPLAY, batch=_BATCH, target_upd=_TARGET_UPD):
        self.node_id   = node_id
        self.gamma     = gamma
        self.lr        = lr
        self.epsilon   = eps_max
        self.eps_min   = eps_min
        self.eps_delta = eps_delta
        self.batch     = batch
        self.target_upd = target_upd
        self.step      = 0

        self.online = QNet(state_dim, n_actions)
        self.target = QNet(state_dim, n_actions)
        self.target.copy_from(self.online)

        self.memory = ReplayBuffer(replay)
        self.total_reward = 0.0
        self.loss_history = deque(maxlen=200)
        self.q_var_hist   = deque(maxlen=50)

    def select_action(self, state: np.ndarray, n_neighbors: int) -> Tuple[int, float, bool]:
        n = min(n_neighbors, _MAX_ACTIONS)
        q = self.online.forward(state)[:n]
        # Boost adaptatif (comme FDQN-TE+)
        qv = float(np.var(q)); self.q_var_hist.append(qv)
        eff_eps = self.epsilon
        if len(self.q_var_hist) >= 10 and qv > 2.0*np.mean(self.q_var_hist):
            eff_eps = min(1.0, self.epsilon + 0.2)
        if random.random() < eff_eps:
            idx = random.randint(0, n-1); return idx, float(q[idx]), True
        idx = int(np.argmax(q)); return idx, float(q[idx]), False

    def store(self, s, a, r, s2, done=False):
        self.memory.push(s, a, r, s2, done); self.total_reward += r

    def learn(self) -> float:
        if len(self.memory) < self.batch: return 0.0
        s, a, r, s2, d = self.memory.sample(self.batch)
        B = len(s)

        # Double DQN
        q_online_next = self.online.forward(s2)
        if q_online_next.ndim == 1: q_online_next = q_online_next.reshape(B, -1)
        best_a = np.argmax(q_online_next, axis=1)

        q_target_next = self.target.forward(s2)
        if q_target_next.ndim == 1: q_target_next = q_target_next.reshape(B, -1)
        targets = r + self.gamma * q_target_next[np.arange(B), best_a] * (1 - d)

        q_all = self.online.forward(s)
        if q_all.ndim == 1: q_all = q_all.reshape(B, -1)
        a_c = np.clip(a, 0, q_all.shape[1]-1)
        td = targets - q_all[np.arange(B), a_c]
        loss = float(np.mean(td**2))

        grad = np.zeros_like(q_all)
        grad[np.arange(B), a_c] = -2.0 * td / B
        self.online.backward(grad)
        self.step += 1
        self.online.adam_step(self.lr, self.step)
        if self.step % self.target_upd == 0:
            self.target.copy_from(self.online)
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_delta)
        self.loss_history.append(loss)
        return loss

    @staticmethod
    def compute_reward(pdr: float, energy_norm: float, delay_norm: float) -> float:
        """Récompense sans λ_safe (pas de PEPM)."""
        scale = 1.0 / (_LAMBDA_PDR + _LAMBDA_E + _LAMBDA_D)
        return scale * (_LAMBDA_PDR * (1.0 if pdr > 0.9 else -1.0)
                        + _LAMBDA_E * energy_norm
                        - _LAMBDA_D * delay_norm)

    def build_state(self, energy_norm, dist_sink_norm, n_neighbors_norm,
                    is_ch, tx_norm, avg_neighbor_energy,
                    alive_frac, dist_ch_norm, recluster_norm) -> np.ndarray:
        """État 9D — sans dimension PEPM risk (index 2 de BuildState)."""
        return np.array([energy_norm, dist_sink_norm,
                         n_neighbors_norm, is_ch, tx_norm,
                         avg_neighbor_energy, alive_frac,
                         dist_ch_norm, recluster_norm], dtype=np.float32)

    def get_params(self) -> dict:
        return {"node_id": self.node_id,
                "weights": [p for layer_p in self.online.get_params() for p in layer_p],
                "n_samples": len(self.memory)}

    def set_params(self, params: dict):
        flat = params.get("weights", [])
        if not flat: return
        n_layers = len(self.online.layers)
        if len(flat) == n_layers * 2:
            rebuilt = [[np.array(flat[2*i]), np.array(flat[2*i+1])]
                       for i in range(n_layers)]
            self.online.set_params(rebuilt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FDQN sans Fédération (Double DQN + PEPM, pas de FedMeta-DRL)
# ─────────────────────────────────────────────────────────────────────────────

class FDQNNoFedAgent:
    """
    FDQN complet (Double-DQN + PEPM/LSTM) mais SANS fédération.
    Chaque nœud apprend de façon entièrement locale.
    - Double-DQN    ✓
    - PEPM          ✓  (risque prédit inclus dans l'état)
    - Fédération    ✗  → aucun partage de poids entre nœuds
    - État 10D      (identique à FDQN-TE+)
    """
    NAME = "FDQN sans Fédération"

    def __init__(self, node_id: int, state_dim: int = _STATE_DIM,
                 n_actions: int = _MAX_ACTIONS,
                 gamma=_GAMMA, lr=_LR,
                 eps_max=_EPS_MAX, eps_min=_EPS_MIN, eps_delta=_EPS_DELTA,
                 replay=_REPLAY, batch=_BATCH, target_upd=_TARGET_UPD):
        self.node_id   = node_id
        self.gamma     = gamma
        self.lr        = lr
        self.epsilon   = eps_max
        self.eps_min   = eps_min
        self.eps_delta = eps_delta
        self.batch     = batch
        self.target_upd = target_upd
        self.step      = 0

        self.online = QNet(state_dim, n_actions)
        self.target = QNet(state_dim, n_actions)
        self.target.copy_from(self.online)

        self.memory = ReplayBuffer(replay)
        self.total_reward = 0.0
        self.loss_history = deque(maxlen=200)
        self.q_var_hist   = deque(maxlen=50)

        # PEPM local (simplifié — EWMA uniquement, pas de LSTM)
        self._pepm_risk  = 0.0
        self._pepm_trend = 0.0
        self._prev_e     = None

    def update_pepm(self, energy_norm: float) -> float:
        """Mise à jour PEPM locale (EWMA, miroir de node_state.h)."""
        alpha = 0.1
        if self._prev_e is not None:
            inst = energy_norm - self._prev_e
            self._pepm_trend = (1 - alpha)*self._pepm_trend + alpha*inst
        self._prev_e = energy_norm

        te_max = 0.5
        norm_trend = max(-5.0, min(5.0, -self._pepm_trend * 20))
        ewma_risk = 1.0 / (1.0 + math.exp(-norm_trend)) if self._prev_e else 0.1

        if energy_norm < te_max:
            abs_risk = 1.0 - energy_norm / te_max
        else:
            abs_risk = max(0.0, (1.0 - energy_norm) / (1.0 - te_max)) * 0.3

        self._pepm_risk = float(np.clip(0.5 * self._pepm_risk + 0.3 * ewma_risk
                                        + 0.2 * abs_risk, 0.0, 1.0))
        return self._pepm_risk

    def select_action(self, state: np.ndarray, n_neighbors: int) -> Tuple[int, float, bool]:
        n = min(n_neighbors, _MAX_ACTIONS)
        q = self.online.forward(state)[:n]
        qv = float(np.var(q)); self.q_var_hist.append(qv)
        eff_eps = self.epsilon
        if len(self.q_var_hist) >= 10 and qv > 2.0*np.mean(self.q_var_hist):
            eff_eps = min(1.0, self.epsilon + 0.2)
        if random.random() < eff_eps:
            idx = random.randint(0, n-1); return idx, float(q[idx]), True
        idx = int(np.argmax(q)); return idx, float(q[idx]), False

    def store(self, s, a, r, s2, done=False):
        self.memory.push(s, a, r, s2, done); self.total_reward += r

    def learn(self) -> float:
        if len(self.memory) < self.batch: return 0.0
        s, a, r, s2, d = self.memory.sample(self.batch)
        B = len(s)

        q_on_next = self.online.forward(s2)
        if q_on_next.ndim == 1: q_on_next = q_on_next.reshape(B, -1)
        best_a = np.argmax(q_on_next, axis=1)

        q_tgt_next = self.target.forward(s2)
        if q_tgt_next.ndim == 1: q_tgt_next = q_tgt_next.reshape(B, -1)
        targets = r + self.gamma * q_tgt_next[np.arange(B), best_a] * (1 - d)

        q_all = self.online.forward(s)
        if q_all.ndim == 1: q_all = q_all.reshape(B, -1)
        a_c = np.clip(a, 0, q_all.shape[1]-1)
        td = targets - q_all[np.arange(B), a_c]
        loss = float(np.mean(td**2))

        grad = np.zeros_like(q_all)
        grad[np.arange(B), a_c] = -2.0 * td / B
        self.online.backward(grad)
        self.step += 1
        self.online.adam_step(self.lr, self.step)
        if self.step % self.target_upd == 0:
            self.target.copy_from(self.online)
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_delta)
        self.loss_history.append(loss)
        return loss

    @staticmethod
    def compute_reward(pdr: float, energy_norm: float, delay_norm: float,
                       pepm_risk: float) -> float:
        """Récompense complète — identique à FDQN-TE+ (λ_hier non inclus car hors scope)."""
        return (_LAMBDA_PDR * (1.0 if pdr > 0.9 else -1.0)
                + _LAMBDA_E  * energy_norm
                - _LAMBDA_D  * delay_norm
                - _LAMBDA_S  * pepm_risk)

    def build_state(self, energy_norm, dist_sink_norm, pepm_risk,
                    n_neighbors_norm, is_ch, tx_norm, avg_neighbor_energy,
                    alive_frac, dist_ch_norm, recluster_norm) -> np.ndarray:
        """État 10D identique à BuildState() de fdqn_te_plus.cc."""
        return np.array([energy_norm, dist_sink_norm, pepm_risk,
                         n_neighbors_norm, is_ch, tx_norm,
                         avg_neighbor_energy, alive_frac,
                         dist_ch_norm, recluster_norm], dtype=np.float32)
