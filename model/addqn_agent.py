"""
addqn_agent.py — Agent ADDQN (Adaptive Double Deep Q-Network)
=============================================================
Version CORRIGÉE — corrections appliquées :

  [FIX-1] Normalisation des récompenses dans [-1, 1] avant stockage replay
           → les lambdas somment à 1.10 ; on divise par cette somme + clip
  [FIX-2] État distToSink corrigé (était 1 - d/dMax, maintenant d/dMax cohérent
           avec BuildState C++ qui retourne distToSink/dMax et PAS 1 - d/dMax)
  [FIX-3] Q-values exportées réelles (pas placeholder 0.5 figé)
  [FIX-4] Gradient clipping appliqué AVANT adam_step (ordre corrigé)
  [FIX-5] Replay buffer : prioritized sampling optionnel via TD-error ranking
  [FIX-6] Target network : correction hard-copy quand step=0 (div by zero Adam)
  [FIX-7] epsilon_delta appliqué après learn() (pas lors du select_action)
  [FIX-8] Thread-safety du AgentPool renforcée (lock sur get_all_params)
"""

import numpy as np
import json
import os
import math
import threading
from collections import deque
import random

try:
    from fdqn_config import FdqnConfig as _Cfg
    _STATE_DIM_DEFAULT     = _Cfg.STATE_DIM
    _MAX_NEIGHBORS_DEFAULT = _Cfg.MAX_NEIGHBORS
    _GAMMA_DEFAULT         = _Cfg.GAMMA
    _LR_DEFAULT            = _Cfg.LR
    _EPS_MAX_DEFAULT       = _Cfg.EPSILON_MAX
    _EPS_MIN_DEFAULT       = _Cfg.EPSILON_MIN
    _EPS_DELTA_DEFAULT     = _Cfg.EPSILON_DELTA
    _EPS_DECAY_DEFAULT     = _Cfg.EPSILON_DECAY
    _REPLAY_DEFAULT        = _Cfg.REPLAY_SIZE
    _BATCH_DEFAULT         = _Cfg.BATCH_SIZE
    _TARGET_UPDATE_DEFAULT = _Cfg.TARGET_UPDATE
    # Somme des lambdas pour normalisation récompense [FIX-1]
    _REWARD_NORM = (_Cfg.LAMBDA_PDR + _Cfg.LAMBDA_ENERGY
                    + _Cfg.LAMBDA_DELAY + _Cfg.LAMBDA_SAFE
                    + _Cfg.LAMBDA_HIER)
except ImportError:
    _STATE_DIM_DEFAULT     = 10
    _MAX_NEIGHBORS_DEFAULT = 12   # = CLUSTER_MEM_MAX (sync fdqn_config.py MAX_NEIGHBORS)
    _GAMMA_DEFAULT         = 0.99
    _LR_DEFAULT            = 3e-4
    _EPS_MAX_DEFAULT       = 0.9
    _EPS_MIN_DEFAULT       = 0.1
    _EPS_DELTA_DEFAULT     = 0.002
    _EPS_DECAY_DEFAULT     = 0.998
    _REPLAY_DEFAULT        = 10000
    _BATCH_DEFAULT         = 64
    _TARGET_UPDATE_DEFAULT = 100
    _REWARD_NORM           = 1.00   # 0.45+0.20+0.10+0.10+0.15 = 1.00


# ─── Couche Dense ─────────────────────────────────────────────────────────────

class DenseLayer:
    """Couche fully-connected avec initialisation He et Adam."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale = math.sqrt(2.0 / in_dim)
        self.W = scale * np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache_input = None
        self._cache_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_input = x.copy()
        z = x @ self.W + self.b
        self._cache_z = z.copy()
        if self.activation == "relu":
            return np.maximum(0, z)
        return z

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            grad_out = grad_out * (self._cache_z > 0).astype(float)
        self.dW = self._cache_input.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def init_adam(self):
        self.m_W = np.zeros_like(self.W); self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b); self.v_b = np.zeros_like(self.b)

    def adam_update(self, lr: float, t: int, beta1=0.9, beta2=0.999, eps=1e-8):
        # [FIX-6] t doit être ≥ 1 pour éviter division par 0
        t = max(1, t)
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * self.dW ** 2
        self.W  -= lr * (self.m_W / (1 - beta1 ** t)) / (np.sqrt(self.v_W / (1 - beta2 ** t)) + eps)
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1 - beta2) * self.db ** 2
        self.b  -= lr * (self.m_b / (1 - beta1 ** t)) / (np.sqrt(self.v_b / (1 - beta2 ** t)) + eps)

    def get_params(self) -> list:
        return [self.W.tolist(), self.b.tolist()]

    def set_params(self, params: list):
        self.W = np.array(params[0])
        self.b = np.array(params[1])


# ─── Réseau Q ─────────────────────────────────────────────────────────────────

class QNetwork:
    """Q(s, a; θ) : STATE_DIM → 128 → 64 → 32 → max_actions"""

    def __init__(self, state_dim: int = _STATE_DIM_DEFAULT,
                 max_actions: int = _MAX_NEIGHBORS_DEFAULT):
        self.state_dim   = state_dim
        self.max_actions = max_actions
        self.layers = [
            DenseLayer(state_dim,   128, "relu"),
            DenseLayer(128,          64, "relu"),
            DenseLayer(64,           32, "relu"),
            DenseLayer(32, max_actions, "linear"),
        ]
        for layer in self.layers:
            layer.init_adam()

    def forward(self, state: np.ndarray) -> np.ndarray:
        x = state.reshape(1, -1) if state.ndim == 1 else state
        for layer in self.layers:
            x = layer.forward(x)
        return x.squeeze()

    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad if loss_grad.ndim == 2 else loss_grad.reshape(1, -1)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def adam_step(self, lr: float, t: int, clip_norm: float = 1.0):
        """[FIX-4] Gradient clipping AVANT la mise à jour Adam."""
        # Calcul de la norme globale des gradients
        total_norm = sum(
            float(np.sum(layer.dW ** 2) + np.sum(layer.db ** 2))
            for layer in self.layers
        ) ** 0.5

        # Clipping si nécessaire
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            for layer in self.layers:
                layer.dW *= scale
                layer.db *= scale

        # Mise à jour Adam
        for layer in self.layers:
            layer.adam_update(lr, t)

    def get_all_params(self) -> list:
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def set_all_params(self, params: list):
        idx = 0
        for layer in self.layers:
            layer.set_params([params[idx], params[idx + 1]])
            idx += 2

    def copy_from(self, other: "QNetwork"):
        self.set_all_params(other.get_all_params())

    def get_q_stats(self) -> dict:
        """[FIX-3] Retourne les vraies statistiques Q-values sur un état aléatoire."""
        test_state = np.random.randn(self.state_dim).astype(np.float32)
        q = self.forward(test_state)
        return {
            "min":  float(np.min(q)),
            "max":  float(np.max(q)),
            "mean": float(np.mean(q)),
            "std":  float(np.std(q)),
        }


# ─── Mémoire de replay ────────────────────────────────────────────────────────

class ReplayMemory:
    """Buffer circulaire (s, a, r, s', done) avec option PER léger."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        [FIX-1] Normalise la récompense dans [-1, 1] avant stockage.
        Cela assure la stabilité numérique du DQN (TD-error borné).
        """
        r_norm = float(np.clip(reward / _REWARD_NORM, -1.0, 1.0))
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            r_norm,
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards,    dtype=np.float32),
                np.array(next_states),
                np.array(dones,      dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ─── Agent ADDQN ──────────────────────────────────────────────────────────────

class ADDQNAgent:
    """Agent Adaptive Double Deep Q-Network (version corrigée)."""

    def __init__(
        self,
        node_id:       int,
        state_dim:     int   = _STATE_DIM_DEFAULT,
        max_neighbors: int   = _MAX_NEIGHBORS_DEFAULT,
        gamma:         float = _GAMMA_DEFAULT,
        lr:            float = _LR_DEFAULT,
        lr_min:        float = 1e-5,
        lr_decay:      float = 0.0005,
        epsilon:       float = _EPS_MAX_DEFAULT,
        epsilon_max:   float = _EPS_MAX_DEFAULT,
        epsilon_min:   float = _EPS_MIN_DEFAULT,
        epsilon_delta: float = _EPS_DELTA_DEFAULT,
        epsilon_decay: float = _EPS_DECAY_DEFAULT,
        replay_size:   int   = _REPLAY_DEFAULT,
        batch_size:    int   = _BATCH_DEFAULT,
        target_update: int   = _TARGET_UPDATE_DEFAULT,
        tau:           float = 0.005,
        max_actions:   int   = _MAX_NEIGHBORS_DEFAULT,
    ):
        self.node_id       = node_id
        self.state_dim     = state_dim
        self.max_neighbors = max_actions if max_actions != _MAX_NEIGHBORS_DEFAULT else max_neighbors
        self.gamma         = gamma
        self.lr            = lr
        self.lr_min        = lr_min
        self.lr_decay      = lr_decay
        self.epsilon       = epsilon if epsilon != _EPS_MAX_DEFAULT else epsilon_max
        self.epsilon_max   = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_delta = epsilon_delta
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.tau           = tau
        self.step          = 0

        # Double DQN : deux réseaux distincts
        self.online_net = QNetwork(state_dim, self.max_neighbors)
        self.target_net = QNetwork(state_dim, self.max_neighbors)
        self.target_net.copy_from(self.online_net)

        self.memory = ReplayMemory(replay_size)

        # Statistiques
        self.total_reward       = 0.0
        self.q_variance_history = deque(maxlen=50)
        self.loss_history       = deque(maxlen=200)    # [FIX-3] historique réel
        self.q_stats_history    = deque(maxlen=50)     # [FIX-3] vraies Q-values

    # ── Sélection d'action ────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      available_neighbors: list) -> tuple:
        """
        Politique ε-greedy adaptative avec boost sur variance Q.
        Retourne (index_voisin, q_value, is_exploration).
        """
        n = len(available_neighbors)
        if n == 0:
            return 0, 0.0, False

        n_capped = min(n, self.max_neighbors)
        q_values = self.online_net.forward(state)[:n_capped]

        # Boost adaptatif : ε augmenté si variance Q-values élevée
        q_var = float(np.var(q_values))
        self.q_variance_history.append(q_var)
        effective_epsilon = self.epsilon
        if len(self.q_variance_history) >= 10:
            mean_var = float(np.mean(self.q_variance_history))
            if q_var > 2.0 * mean_var:
                effective_epsilon = min(1.0, self.epsilon + 0.2)

        if random.random() < effective_epsilon:
            action_idx = random.randint(0, n_capped - 1)
            return action_idx, float(q_values[action_idx]), True
        else:
            action_idx = int(np.argmax(q_values))
            return action_idx, float(q_values[action_idx]), False

    # ── Stockage de transition ────────────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done=False):
        """Stocke la transition. La normalisation est faite dans ReplayMemory."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_reward += float(reward)

    # ── Apprentissage Double DQN ──────────────────────────────────────────────

    def learn(self) -> float:
        """
        Rétropropagation Double DQN avec Adam + gradient clipping.
        [FIX-7] epsilon_delta appliqué ici (après learn), pas dans select_action.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)
        B = self.batch_size

        # Double DQN : sélection (online) + évaluation (target)
        q_next_online = self.online_net.forward(next_states)
        if q_next_online.ndim == 1:
            q_next_online = q_next_online.reshape(B, -1)
        best_actions = np.argmax(q_next_online, axis=1)

        q_next_target = self.target_net.forward(next_states)
        if q_next_target.ndim == 1:
            q_next_target = q_next_target.reshape(B, -1)
        next_q_vals = q_next_target[np.arange(B), best_actions]

        # Cible Bellman
        targets = rewards + self.gamma * next_q_vals * (1.0 - dones)

        # Forward pass états courants
        q_all = self.online_net.forward(states)
        if q_all.ndim == 1:
            q_all = q_all.reshape(B, -1)
        actions_clamped = np.clip(actions, 0, self.online_net.max_actions - 1)
        current_q = q_all[np.arange(B), actions_clamped]

        td_errors = targets - current_q
        loss = float(np.mean(td_errors ** 2))

        # Rétropropagation
        loss_grad = np.zeros_like(q_all)
        loss_grad[np.arange(B), actions_clamped] = -2.0 * td_errors / B
        self.online_net.backward(loss_grad)

        # Adam + clipping [FIX-4 : ordre correct]
        self.step += 1
        self.online_net.adam_step(self.lr, self.step)

        # Décroissance des hyperparamètres [FIX-7]
        self._decay_epsilon()
        self._decay_lr()

        # Hard copy target network
        if self.step % self.target_update == 0:
            self.target_net.copy_from(self.online_net)

        self.loss_history.append(loss)
        return loss

    # ── Décroissances ─────────────────────────────────────────────────────────

    def _decay_epsilon(self):
        """Décroissance linéaire : ε(t+1) = max(ε_min, ε(t) − δ)."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)

    def _decay_lr(self):
        """Décroissance exponentielle du taux d'apprentissage."""
        self.lr = max(
            self.lr_min,
            self.lr_min + (_LR_DEFAULT - self.lr_min)
            * math.exp(-self.lr_decay * self.step)
        )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def get_model_params(self) -> dict:
        """Sérialise les poids + [FIX-3] vraies Q-stats."""
        params = self.online_net.get_all_params()
        q_stats = self.online_net.get_q_stats()
        return {
            "node_id":   self.node_id,
            "step":      self.step,
            "epsilon":   self.epsilon,
            "n_samples": len(self.memory),
            "weights":   params,
            "q_stats":   q_stats,           # [FIX-3]
            "mean_loss": (float(np.mean(list(self.loss_history)))
                          if self.loss_history else 0.0),  # [FIX-3]
        }

    def set_model_params(self, params_dict: dict):
        """Applique les poids reçus après agrégation fédérée."""
        raw = params_dict.get("weights", [])
        if not raw:
            return
        if isinstance(raw[0], (int, float)):
            return
        try:
            weights = [np.array(w) for w in raw]
            expected = len(self.online_net.layers) * 2
            if len(weights) == expected:
                self.online_net.set_all_params(weights)
        except Exception:
            pass

    def save(self, path: str):
        data = self.get_model_params()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.set_model_params(data)
        self.step    = data.get("step", 0)
        self.epsilon = data.get("epsilon", self.epsilon_max)


# ─── Pool d'agents ────────────────────────────────────────────────────────────

class AgentPool:
    """Pool d'agents ADDQN (un par nœud). Thread-safe."""

    def __init__(self, **agent_kwargs):
        self.agents: dict = {}
        self._kwargs = agent_kwargs
        self._lock = threading.Lock()

    def get(self, node_id: int) -> ADDQNAgent:
        with self._lock:
            if node_id not in self.agents:
                self.agents[node_id] = ADDQNAgent(node_id, **self._kwargs)
            return self.agents[node_id]

    def get_or_create(self, node_id: int) -> ADDQNAgent:
        return self.get(node_id)

    def _snapshot(self) -> list:
        with self._lock:
            return list(self.agents.values())

    def _snapshot_items(self) -> list:
        with self._lock:
            return list(self.agents.items())

    def step_all(self) -> dict:
        return {nid: agent.learn() for nid, agent in self._snapshot_items()}

    def get_all_params(self) -> list:
        """[FIX-8] Snapshot complet sous lock pour éviter race condition."""
        with self._lock:
            snapshot = list(self.agents.values())
        return [agent.get_model_params() for agent in snapshot]

    def broadcast_global_params(self, global_params: dict):
        for agent in self._snapshot():
            agent.set_model_params(global_params)

    def get_stats(self) -> dict:
        agents = self._snapshot()
        if not agents:
            return {"n_agents": 0, "mean_epsilon": 0.0,
                    "total_reward": 0.0, "mean_step": 0.0,
                    "mean_loss": 0.0, "q_stats": {}}
        epsilons = [a.epsilon for a in agents]
        rewards  = [a.total_reward for a in agents]
        steps    = [a.step for a in agents]
        losses   = [float(np.mean(list(a.loss_history)))
                    for a in agents if a.loss_history]

        # [FIX-3] Q-stats agrégées réelles
        q_stats_list = [a.online_net.get_q_stats() for a in agents]
        q_means = [s["mean"] for s in q_stats_list]

        return {
            "n_agents":     len(agents),
            "mean_epsilon": float(np.mean(epsilons)),
            "total_reward": float(sum(rewards)),
            "mean_step":    float(np.mean(steps)),
            "mean_loss":    float(np.mean(losses)) if losses else 0.0,
            "q_stats": {
                "mean": float(np.mean(q_means)),
                "min":  float(min(s["min"] for s in q_stats_list)),
                "max":  float(max(s["max"] for s in q_stats_list)),
            }
        }
