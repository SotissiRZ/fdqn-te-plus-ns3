"""
pepm_lstm.py — PEPM Proactif (Predictive Energy Proactive Mechanism)

Architecture proactive :
  - LSTM PyTorch entraîné en ligne sur l'historique énergétique de chaque nœud
  - Prédit le Time-To-Death (TTD) : combien de steps avant épuisement
  - Score de risque = exp(-TTD / horizon) → décision AVANT que l'énergie soit critique
  - Contrairement à l'approche réactive (sigmoïde sur énergie courante), le TTD
    permet d'alerter même un nœud encore à 70% d'énergie si son taux de drain
    est anormalement élevé (ex: cluster head surchargé)

Entrées LSTM (3 features par step) :
  - e_norm  : énergie normalisée ∈ [0,1]
  - t_norm  : tendance de drain lissée (tanh) ∈ [-1,1]
  - l_norm  : charge relative (load normalisée) ∈ [0,1)

Sortie LSTM : TTD prédit (steps avant mort)
Risque final : score = exp(-TTD_prédit / HORIZON) ∈ [0,1]
"""

import numpy as np
import math
import json
import os
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from fdqn_config import FdqnConfig
except ImportError:
    class FdqnConfig:
        PEPM_WINDOW         = 10
        PEPM_HIDDEN         = 32
        PEPM_RISK_THRESHOLD = 0.5
        PEPM_TE_MAX         = 0.5
        PEPM_ALPHA          = 0.1
        E_INIT              = 1.2
        N_NODES             = 300


# ============================================================
# CONFIG PEPM PROACTIF
# ============================================================

SEQ_LEN     = FdqnConfig.PEPM_WINDOW   # longueur séquence LSTM
INPUT_SIZE  = 3                         # (e_norm, trend_norm, load_norm)
HIDDEN_SIZE = FdqnConfig.PEPM_HIDDEN
LR          = 0.001
TTD_CAP = 5000.0   # plafond TTD — noeuds frais (trend~0)
HORIZON = 300.0   # calibre : score=0.5


# ============================================================
# LSTM MODEL (PyTorch) — défini seulement si torch est disponible
# ============================================================

if TORCH_AVAILABLE:
    class PEPM_LSTM(nn.Module):
        """
        LSTM qui prédit le TTD (Time To Death) en steps.
        Entrée : séquence de (e_norm, trend_norm, load_norm)
        Sortie : TTD prédit ∈ [0, +∞)
        """
        def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc   = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x : (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            out = out[:, -1, :]        # dernier timestep
            out = self.fc(out)
            return torch.relu(out)     # TTD ≥ 0
else:
    PEPM_LSTM = None  # placeholder — non utilisé sans PyTorch


# ============================================================
# FALLBACK LSTM NUMPY (si PyTorch absent)
# ============================================================

class LSTMCell:
    """Cellule LSTM numpy — fallback si PyTorch non disponible."""
    def __init__(self, input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE):
        self.hidden_dim = hidden_dim
        d = input_dim + hidden_dim
        scale = 1.0 / math.sqrt(d)
        self.Wf = np.random.randn(d, hidden_dim) * scale
        self.Wi = np.random.randn(d, hidden_dim) * scale
        self.Wg = np.random.randn(d, hidden_dim) * scale
        self.Wo = np.random.randn(d, hidden_dim) * scale
        self.bf = np.ones(hidden_dim) * 0.5
        self.bi = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([x, h_prev])
        f = 1.0 / (1.0 + np.exp(-(concat @ self.Wf + self.bf)))
        i = 1.0 / (1.0 + np.exp(-(concat @ self.Wi + self.bi)))
        g = np.tanh(concat @ self.Wg + self.bg)
        o = 1.0 / (1.0 + np.exp(-(concat @ self.Wo + self.bo)))
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c


# ============================================================
# NORMALISATION
# ============================================================

def normalize_state(energy: float, max_energy: float,
                    trend: float, load: float) -> np.ndarray:
    """
    Construit le vecteur d'état normalisé pour le LSTM.
      e_norm  = energie / E_init           ∈ [0,1]
      t_norm  = tanh(trend * 1000)         ∈ [-1,1]  (stabilise les petites variations)
      l_norm  = load / (load + 1)          ∈ [0,1)   (normalisation soft)
    """
    e_norm = float(np.clip(energy / (max_energy + 1e-9), 0.0, 1.0))
    t_norm = float(np.tanh(trend * 1000.0))
    l_norm = float(load / (load + 1.0))
    return np.array([e_norm, t_norm, l_norm], dtype=np.float32)


# ============================================================
# TIME TO DEATH
# ============================================================

def compute_ttd(energy: float, drain_per_step: float) -> float:
    """
    Calcule le TTD réel utilisé comme supervision pour l'entraînement.
    drain_per_step : taux de drain en J/step (>0 si décharge).
    Retourne le nombre de steps avant épuisement (plafonné à 1000).
    Si le drain est nul → nœud stable → TTD très grand.
    """
    if drain_per_step <= 1e-9:
        return TTD_CAP
    return float(np.clip(energy / drain_per_step, 0.0, TTD_CAP))


def compute_pepm_score(ttd: float, horizon: float = HORIZON) -> float:
    """
    Convertit le TTD en score de risque ∈ [0,1].
    score = exp(-ttd / horizon)

      ttd = 0          → score = 1.000  (mort imminente)
      ttd = horizon    → score = 0.368  (zone d'alerte)
      ttd = 2*horizon  → score = 0.135  (précaution)
      ttd → ∞          → score → 0.000  (sûr)

    Propriété PROACTIVE clé : un nœud à 80% d'énergie mais avec un
    drain élevé (CH surchargé) peut avoir TTD=50 steps → score=0.78 → ALERTE
    """
    return float(np.exp(-ttd / max(horizon, 1.0)))


# ============================================================
# AGENT PEPM (un par nœud)
# ============================================================

class PEPM_Agent:
    """
    Agent PEPM proactif par nœud.
    Maintient un historique de SEQ_LEN états et entraîne le LSTM
    en ligne à prédire le TTD (Time To Death).
    """

    def __init__(self, node_id: int, max_energy: float = FdqnConfig.E_INIT):
        self.node_id    = node_id
        self.max_energy = max_energy

        # Modèle LSTM PyTorch ou fallback numpy
        if TORCH_AVAILABLE:
            self.model     = PEPM_LSTM()
            self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
            self.criterion = nn.MSELoss()
            self.use_torch = True
        else:
            self.lstm_np  = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
            self.h_state  = np.zeros(HIDDEN_SIZE)
            self.c_state  = np.zeros(HIDDEN_SIZE)
            self.use_torch = False

        # Séquence glissante des états normalisés
        self.memory: deque = deque(maxlen=SEQ_LEN)

        # EWMA tendance drain
        # En mode fallback (sans PyTorch), alpha plus eleve pour convergence rapide
        # alpha=0.15 en fallback : convergence 2x plus rapide que 0.1
        # sans sur-reaction au premier drain (alpha=0.3 alertait le CH a 99% energie)
        self.alpha       = 0.15 if not TORCH_AVAILABLE else FdqnConfig.PEPM_ALPHA
        self.trend_ewma  = 0.0    # drain normalisé lissé (unités E_init/step)
        self.prev_energy = None   # énergie normalisée du step précédent
        self.load        = 0.5    # charge estimée (mise à jour dynamique)

        # Compteur d'entraînements — le LSTM n'est utilisé qu'après LSTM_WARMUP steps
        self.n_train_steps = 0
        # LSTM_WARMUP : nb de steps d'entraînement avant de faire confiance au LSTM
        # En dessous : TTD analytique seul (fiable dès le 1er step)
        self.LSTM_WARMUP = 200

        # État courant exposé aux autres modules
        self.current_risk = 0.0
        self.current_ttd  = 1000.0
        self.risk_history: deque = deque(maxlen=100)

    # ----------------------------------------------------------
    # Interface interne
    # ----------------------------------------------------------

    def remember(self, state: np.ndarray):
        # Ajoute un état à la séquence glissante.
        self.memory.append(state.tolist())

    def predict(self) -> Tuple[float, float]:
        """
        Prédit (pepm_score, ttd) à partir de la séquence courante.

        Stratégie de fusion TTD analytique / LSTM :
          - TTD analytique : fiable dès le 1er step (energy / trend_ewma)
          - TTD LSTM       : fiable seulement après LSTM_WARMUP entraînements
          - Fusion linéaire : w_lstm = min(n_train_steps / LSTM_WARMUP, 1.0)
        """
        # TTD analytique — toujours calculé, référence fiable
        ttd_analytic = compute_ttd(
            (self.prev_energy or 1.0) * self.max_energy,
            self.trend_ewma * self.max_energy
        )

        # Pas encore assez d'historique pour le LSTM
        if len(self.memory) < SEQ_LEN or not self.use_torch:
            return compute_pepm_score(ttd_analytic), ttd_analytic

        # Poids LSTM : croît de 0 → 0.4 sur LSTM_WARMUP entraînements
        # Plafonné à 0.4 : le TTD analytique garde toujours 60% du poids
        # car il est exact par construction (energy/trend) alors que le LSTM
        # peut diverger sur des séquences courtes ou des profils atypiques.
        w_lstm = min(float(self.n_train_steps) / self.LSTM_WARMUP, 1.0) * 0.4

        # Inférence LSTM
        seq = torch.FloatTensor([list(self.memory)])
        with torch.no_grad():
            ttd_lstm = self.model(seq).item()

        # Sanity-clamp : le LSTM ne peut pas prédire un TTD
        # inférieur à 20% ni supérieur à 300% du TTD analytique
        ttd_lstm = float(np.clip(
            ttd_lstm,
            ttd_analytic * 0.2,
            ttd_analytic * 3.0
        ))

        # Fusion pondérée — analytique toujours majoritaire
        ttd_final = (1.0 - w_lstm) * ttd_analytic + w_lstm * ttd_lstm

        return compute_pepm_score(ttd_final), ttd_final

    def train_step(self, target_ttd: float):
        # Entraîne le LSTM sur la vraie valeur de TTD (supervision en ligne).
        if not self.use_torch or len(self.memory) < SEQ_LEN:
            return
        seq    = torch.FloatTensor([list(self.memory)])
        target = torch.FloatTensor([[target_ttd]])
        pred   = self.model(seq)
        loss   = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.n_train_steps += 1

    # ----------------------------------------------------------
    # Interface publique principale
    # ----------------------------------------------------------

    def update(self, energy_j: float, load: float = 0.5) -> float:
        """
        Point d'entrée appelé à chaque step RL pour ce nœud.

        Flux proactif :
          1. Calcule la tendance EWMA (drain lissé)
          2. Construit l'état normalisé (e_norm, t_norm, l_norm)
          3. Prédit le TTD via LSTM → score = exp(-TTD/HORIZON)
          4. Supervise l'entraînement avec le TTD réel (analytique)
          5. Retourne le score de risque PEPM ∈ [0,1]
        """
        self.load  = load
        e_norm     = float(np.clip(energy_j / self.max_energy, 0.0, 1.0))

        # 1. Tendance EWMA
        self._step_count = getattr(self, '_step_count', 0) + 1
        if self.prev_energy is not None:
            instant_drain   = self.prev_energy - e_norm      # >0 si décharge
            self.trend_ewma = ((1.0 - self.alpha) * self.trend_ewma
                               + self.alpha * max(0.0, instant_drain))
        self.prev_energy = e_norm

        # 2. État normalisé
        state = normalize_state(energy_j, self.max_energy, self.trend_ewma, load)
        self.remember(state)

        # 3. Prédiction LSTM → TTD prédit → score
        # Warmup : 3 premiers steps ignorés (EWMA non encore stabilisée)
        if self._step_count <= 3:
            self.current_risk = 0.0
            self.current_ttd  = TTD_CAP
            self.risk_history.append(0.0)
            return 0.0
        pepm_score, ttd_pred = self.predict()

        # 4. Supervision en ligne avec TTD réel
        real_ttd = compute_ttd(energy_j, self.trend_ewma * self.max_energy)
        self.train_step(real_ttd)

        # 5. Mise à jour état exposé
        self.current_risk = float(np.clip(pepm_score, 0.0, 1.0))
        self.current_ttd  = ttd_pred
        self.risk_history.append(self.current_risk)

        return self.current_risk

    # ----------------------------------------------------------
    # API publique
    # ----------------------------------------------------------

    def get_risk(self) -> float:
        return self.current_risk

    def get_ttd(self) -> float:
        # Retourne le TTD prédit en steps.
        return self.current_ttd

    def is_at_risk(self, threshold: float = FdqnConfig.PEPM_RISK_THRESHOLD) -> bool:
        return self.current_risk > threshold

    def decision(self) -> str:
        # Décision PEPM textuelle pour le logging.
        if self.current_risk > 0.8:
            return "CRITICAL"
        elif self.current_risk > 0.6:
            return "WARNING"
        elif self.current_risk > 0.3:
            return "NORMAL"
        else:
            return "SAFE"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "node_id":     self.node_id,
            "risk":        round(self.current_risk, 4),
            "ttd_steps":   round(self.current_ttd, 1),
            "ttd_seconds": round(self.current_ttd * 5.0, 1),   # à RL_STEP=5s
            "trend_ewma":  round(self.trend_ewma, 6),
            "drain_J_step": round(self.trend_ewma * self.max_energy, 6),
            "energy":      round((self.prev_energy or 0.0) * self.max_energy, 4),
            "decision":    self.decision(),
            "seq_ready":   len(self.memory) >= SEQ_LEN,
        }

    def save(self, path: str):
        data: Dict[str, Any] = {
            "node_id":      self.node_id,
            "trend_ewma":   self.trend_ewma,
            "prev_energy":  self.prev_energy,
            "current_risk": self.current_risk,
            "current_ttd":  self.current_ttd,
            "memory":       list(self.memory),
        }
        if self.use_torch:
            data["model_state"] = {
                k: v.tolist() for k, v in self.model.state_dict().items()
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.trend_ewma   = data.get("trend_ewma", 0.0)
        self.prev_energy  = data.get("prev_energy", None)
        self.current_risk = data.get("current_risk", 0.0)
        self.current_ttd  = data.get("current_ttd", 1000.0)
        self.memory       = deque(data.get("memory", []), maxlen=SEQ_LEN)
        if self.use_torch and "model_state" in data:
            state_dict = {k: torch.tensor(v)
                          for k, v in data["model_state"].items()}
            self.model.load_state_dict(state_dict)


# ============================================================
# PEPM POOL
# ============================================================

class PEPMPool:
    # Pool d'agents PEPM proactifs — un par nœud capteur.


    def __init__(
        self,
        window: int           = FdqnConfig.PEPM_WINDOW,
        hidden_dim: int       = FdqnConfig.PEPM_HIDDEN,
        risk_threshold: float = FdqnConfig.PEPM_RISK_THRESHOLD,
        e_init: float         = FdqnConfig.E_INIT,
    ):
        self.agents: Dict[int, PEPM_Agent] = {}
        self.window         = window
        self.hidden_dim     = hidden_dim
        self.risk_threshold = risk_threshold
        self.e_init         = e_init

        mode = "PyTorch LSTM + TTD" if TORCH_AVAILABLE else "TTD analytique (fallback)"
        print(f"[PEPM] Mode PROACTIF — {mode}")

    def get_or_create(self, node_id: int) -> PEPM_Agent:
        if node_id not in self.agents:
            self.agents[node_id] = PEPM_Agent(node_id, max_energy=self.e_init)
        return self.agents[node_id]

    def update_node(self, node_id: int, energy: float, load: float = 0.5) -> float:
        agent = self.get_or_create(node_id)
        return agent.update(energy, load)   # retourne risk ∈ [0,1] directement

    def get_risk(self, node_id: int) -> float:
        agent = self.agents.get(node_id)
        return agent.get_risk() if agent else 0.0

    def get_ttd(self, node_id: int) -> float:
        agent = self.agents.get(node_id)
        return agent.get_ttd() if agent else 1000.0

    def get_at_risk_nodes(self) -> List[int]:
        return [nid for nid, ag in self.agents.items()
                if ag.is_at_risk(self.risk_threshold)]

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return [ag.get_stats() for ag in self.agents.values()]

    def get_summary(self) -> Dict[str, Any]:
        if not self.agents:
            return {}
        risks = [ag.get_risk() for ag in self.agents.values()]
        ttds  = [ag.get_ttd()  for ag in self.agents.values()]
        return {
            "n_nodes":        len(self.agents),
            "mean_risk":      float(np.mean(risks)),
            "max_risk":       float(np.max(risks)),
            "min_risk":       float(np.min(risks)),
            "nodes_at_risk":  len(self.get_at_risk_nodes()),
            "mean_ttd_steps": float(np.mean(ttds)),
            "min_ttd_steps":  float(np.min(ttds)),
        }


# ============================================================
# PEPMModule
# ============================================================

class PEPMModule(PEPM_Agent):

    _NOMINAL_DRAIN = 7e-4
    _CH_DRAIN      = 5e-3

    def __init__(self, node_id, window=FdqnConfig.PEPM_WINDOW,
                 hidden_dim=FdqnConfig.PEPM_HIDDEN,
                 te_max=FdqnConfig.PEPM_TE_MAX,
                 alpha=FdqnConfig.PEPM_ALPHA,
                 e_init=FdqnConfig.E_INIT):
        super().__init__(node_id, max_energy=e_init)
        self.alpha  = alpha
        self.te_max = te_max

    def get_threshold(self) -> float:
        return self.current_risk * self.te_max


# ============================================================
# TEST / VALIDATION
# ============================================================

if __name__ == "__main__":
    print("=== Test PEPM PROACTIF — validation TTD + alerte précoce ===")
    print(f"  E_INIT={FdqnConfig.E_INIT} J | seuil={FdqnConfig.PEPM_RISK_THRESHOLD}")
    print(f"  PyTorch: {'✓ disponible' if TORCH_AVAILABLE else '✗ absent (fallback analytique)'}")
    print()

    pool = PEPMPool(e_init=FdqnConfig.E_INIT)
    DRAIN_MEMBER = 0.00085   # J/step — drain LEACH membre typ.
    DRAIN_CH     = 0.006     # J/step — drain CH (×7)

    for label, node_id, drain in [("MEMBRE", 1, DRAIN_MEMBER), ("CH", 2, DRAIN_CH)]:
        print(f"--- Nœud {label} (drain={drain*1000:.3f} mJ/step) ---")
        print(f"  {'t':>4} | {'E(J)':>6} | {'e_norm':>6} | {'risk':>6} | "
              f"{'TTD(s)':>7} | Décision")
        print("  " + "-" * 55)

        energy       = FdqnConfig.E_INIT
        first_alert  = None
        steps_dead   = int(FdqnConfig.E_INIT / drain)

        for t in range(1, steps_dead + 20):
            energy = max(0.0, energy - drain)
            risk   = pool.get_or_create(node_id).update(energy)
            ttd    = pool.get_ttd(node_id)
            dec    = pool.agents[node_id].decision()

            if risk >= FdqnConfig.PEPM_RISK_THRESHOLD and first_alert is None:
                first_alert = (t, energy, risk, ttd)

            if t % 200 == 0 or (first_alert and t <= first_alert[0] + 3):
                e_norm = energy / FdqnConfig.E_INIT
                print(f"  {t:>4} | {energy:>6.4f} | {e_norm:>6.3f} | "
                      f"{risk:>6.3f} | {ttd*5:>7.1f} | {dec}")

        if first_alert:
            advance = steps_dead - first_alert[0]
            print(f"\n  → FND à step {steps_dead} (E→0)")
            print(f"  → 1ère alerte PEPM à step {first_alert[0]} "
                  f"(E={first_alert[1]:.4f}J, risk={first_alert[2]:.3f}, "
                  f"TTD_prédit={first_alert[3]*5:.0f}s)")
            print(f"  → Avance d'alerte : {advance} steps "
                  f"= {advance*5}s ✓\n")
        else:
            print("  ⚠ Aucune alerte déclenchée — HORIZON trop grand ?\n")

    print("Résumé global:", pool.get_summary())
