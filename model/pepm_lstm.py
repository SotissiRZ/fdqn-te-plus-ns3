"""
pepm_lstm.py — PEPM (Predictive Energy Proactive Mechanism)
Version corrigée avec :
✓ Modèle hybride LSTM (structure) mais apprentissage simplifié (EWMA)
✓ Documentation des limitations
✓ Intégration avec IFO pour re-clustering
"""

import numpy as np
import math
import json
import os
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

try:
    from fdqn_config import FdqnConfig
except ImportError:
    class FdqnConfig:
        PEPM_WINDOW = 10
        PEPM_HIDDEN = 64
        PEPM_RISK_THRESHOLD = 0.5
        PEPM_TE_MAX = 0.5
        PEPM_ALPHA = 0.1


# ============================================================
# LSTM Cell (structure seulement - pas de backprop complète)
# ============================================================

class LSTMCell:
    """
    Cellule LSTM - utilisée pour la représentation d'état uniquement.
    NOTE: L'apprentissage réel utilise une EWMA pour rester compatible
    avec les contraintes de calcul des nœuds IoT.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = FdqnConfig.PEPM_HIDDEN):
        self.hidden_dim = hidden_dim
        d = input_dim + hidden_dim
        scale = 1.0 / math.sqrt(d)

        # Poids des portes (non utilisés pour l'apprentissage réel)
        self.Wf = np.random.randn(d, hidden_dim) * scale
        self.Wi = np.random.randn(d, hidden_dim) * scale
        self.Wg = np.random.randn(d, hidden_dim) * scale
        self.Wo = np.random.randn(d, hidden_dim) * scale

        self.bf = np.ones(hidden_dim) * 0.5
        self.bi = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagation avant (pour inférence)"""
        concat = np.concatenate([x, h_prev])

        f = 1.0 / (1.0 + np.exp(-(concat @ self.Wf + self.bf)))
        i = 1.0 / (1.0 + np.exp(-(concat @ self.Wi + self.bi)))
        g = np.tanh(concat @ self.Wg + self.bg)
        o = 1.0 / (1.0 + np.exp(-(concat @ self.Wo + self.bo)))

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        return h, c


# ============================================================
# Dense Layer
# ============================================================

class Dense:
    """Couche dense simple"""

    def __init__(self, in_dim: int, out_dim: int, activation: str = None):
        scale = math.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagation avant"""
        z = x @ self.W + self.b

        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        else:
            return z


# ============================================================
# PEPM Module (CORRIGÉ)
# ============================================================

class PEPMModule:
    """
    Module de prédiction énergétique proactive

    Architecture hybride:
    - Structure LSTM pour la représentation d'état
    - Apprentissage simplifié par EWMA (Exponential Weighted Moving Average)
      pour rester compatible avec les contraintes IoT

    Risque = fonction sigmoïde de la tendance de décharge
    """

    def __init__(
        self,
        node_id: int,
        window: int = FdqnConfig.PEPM_WINDOW,
        hidden_dim: int = FdqnConfig.PEPM_HIDDEN,
        te_max: float = FdqnConfig.PEPM_TE_MAX,
        alpha: float = FdqnConfig.PEPM_ALPHA,
        e_init: float = FdqnConfig.E_INIT,   # ← depuis config (était 2.0 hardcodé)
        early_warning_threshold: float = 0.70  # [FIX-PEPM-1] Risque progressif dès 70% résiduel
    ):
        self.node_id = node_id
        self.window = window
        self.hidden_dim = hidden_dim
        self.te_max = te_max
        self.alpha = alpha  # Taux d'apprentissage EWMA
        self.e_init = e_init
        # [FIX-PEPM-1] Seuil d'alerte précoce : risque progressif dès ce niveau d'énergie
        # 0.70 = détection à 70% résiduel → ~400s avant FND (avec E_INIT=1.2J, drain=22mJ/step)
        self.early_warning_threshold = early_warning_threshold

        # Modèle LSTM (pour la structure, pas pour l'apprentissage)
        self.lstm = LSTMCell(1, hidden_dim)
        self.dense1 = Dense(hidden_dim, 32, "relu")
        self.dense2 = Dense(32, 1, "sigmoid")
        # Initialiser le biais de sortie à -3 → sigmoid(-3)≈0.047
        # Le risque part bas (nœud frais) et monte avec la décharge
        self.dense2.b[:] = -3.0

        # États internes
        self.h_state = np.zeros(hidden_dim)
        self.c_state = np.zeros(hidden_dim)
        self.energy_history = deque(maxlen=window)

        # EWMA pour la tendance
        self.trend = 0.0
        self.prev_energy = None

        # Statistiques
        self.risk_history = deque(maxlen=100)
        # [FIX Problem 2] risque initial à 0.0 — un nœud frais n'est pas à risque
        self.current_risk = 0.0
        self.current_threshold = 0.0
        # Prédicteur de drain : taux de décharge exponentiel lissé
        self._drain_rate = 0.0          # décharge normalisée par step (EWMA)
        self._drain_alpha = 0.15        # réactivité du lisseur drain

    # ----------------------------------------------------------
    # Mise à jour principale
    # ----------------------------------------------------------

    def update(self, energy_j: float) -> float:
        """
        Met à jour le module avec une nouvelle mesure d'énergie.
        [FIX Problem 2] Fusion dynamique :
          - Avant warm-up (< 3 samples) : abs×60% + trend×40%  (LSTM muet)
          - Après warm-up              : abs×40% + trend×30% + lstm×30%
        Ajoute un prédicteur steps_to_empty basé sur le taux de drain lissé.

        Args:
            energy_j: Énergie résiduelle en Joules

        Returns:
            Seuil prédictif dynamique TE_pred ∈ [0, TE_MAX]
        """
        e_norm = energy_j / self.e_init
        self.energy_history.append(e_norm)

        # ── Taux de drain lissé (EWMA) ─────────────────────────────────────────
        if self.prev_energy is not None:
            instant_drain = self.prev_energy - e_norm   # positif si décharge
            self._drain_rate = ((1 - self._drain_alpha) * self._drain_rate
                                + self._drain_alpha * max(0.0, instant_drain))
            instant_trend = e_norm - self.prev_energy   # négatif si décharge
            self.trend = (1 - self.alpha) * self.trend + self.alpha * instant_trend
        self.prev_energy = e_norm

        warm_up = len(self.energy_history) >= 3

        # ── Risque EWMA (tendance) ─────────────────────────────────────────────
        if warm_up:
            norm_trend = np.clip(-self.trend * 20, -5, 5)
            ewma_risk  = float(1.0 / (1.0 + np.exp(-norm_trend)))
        else:
            ewma_risk = 0.0   # [FIX] 0 au démarrage, pas 0.1

        # ── Risque LSTM (mémoire longue) ───────────────────────────────────────
        if warm_up:
            lstm_risk = self._lstm_predict()
        else:
            lstm_risk = 0.0   # [FIX] 0 pendant le warm-up

        # ── Risque absolu proactif [FIX-PEPM-1] ───────────────────────────────
        # Ancienne formule : risque nul jusqu'à e_norm < te_max=0.5 → trop tardif.
        # Nouvelle formule à 3 zones :
        #   Zone 1 [early_warning .. 1.0] : risque croissant 0→0.15 (signal précoce)
        #   Zone 2 [te_max .. early_warning] : risque croissant 0.15→0.70
        #   Zone 3 [0 .. te_max]            : risque croissant 0.70→1.0 (danger)
        ewt = self.early_warning_threshold   # 0.70
        if e_norm >= ewt:
            # Zone 1 : très faible risque mais non nul → signal précoce pour ADDQN
            abs_risk = 0.15 * (1.0 - (e_norm - ewt) / (1.0 - ewt)) if ewt < 1.0 else 0.0
        elif e_norm >= self.te_max:
            # Zone 2 : risque modéré et croissant
            abs_risk = 0.15 + 0.55 * (1.0 - (e_norm - self.te_max) / (ewt - self.te_max))
        else:
            # Zone 3 : danger — risque élevé (0.70 → 1.0)
            abs_risk = 0.70 + 0.30 * (1.0 - e_norm / self.te_max) if self.te_max > 0 else 1.0
        abs_risk = float(np.clip(abs_risk, 0.0, 1.0))

        # ── Prédicteur steps_to_empty ─────────────────────────────────────────
        # Estime dans combien de steps RL le nœud sera épuisé.
        # [FIX-PEPM-2] Horizon réduit : risque=1 si ≤ 10 steps, 0 si ≥ 30 steps
        # (ancienne valeur : 50 steps → déclenchement trop tardif)
        horizon_risk = 0.0
        if self._drain_rate > 1e-6:
            steps_to_empty = e_norm / self._drain_rate
            # Risque = 1 si ≤ 10 steps restants, 0 si ≥ 30 steps restants
            horizon_risk = float(np.clip(1.0 - steps_to_empty / 30.0, 0.0, 1.0))

        # ── Fusion dynamique ──────────────────────────────────────────────────
        if not warm_up:
            # Avant warm-up : LSTM muet → abs×60% + trend×40%
            combined_risk = 0.60 * abs_risk + 0.40 * ewma_risk
        else:
            # Après warm-up : abs×35% + trend×25% + lstm×25% + horizon×15%
            combined_risk = (0.35 * abs_risk
                             + 0.25 * ewma_risk
                             + 0.25 * lstm_risk
                             + 0.15 * horizon_risk)

        combined_risk = float(np.clip(combined_risk, 0.0, 1.0))

        self.current_risk = combined_risk
        self.risk_history.append(self.current_risk)

        # Seuil prédictif : TE_pred = risque × TE_MAX
        self.current_threshold = self.current_risk * self.te_max
        return self.current_threshold

    # ----------------------------------------------------------
    # Inférence LSTM (optionnelle, pour la compatibilité)
    # ----------------------------------------------------------

    def _lstm_predict(self) -> float:
        """
        Inférence LSTM sur la fenêtre glissante.
        Persiste les états h,c pour la mémoire longue entre appels.
        Retourne σ ∈ [0, 1] : probabilité d'épuisement.
        """
        if len(self.energy_history) < 2:
            return 0.05

        h = self.h_state.copy()
        c = self.c_state.copy()

        # Traiter toute la fenêtre pour enrichir l'état caché
        for e in self.energy_history:
            x = np.array([e], dtype=np.float64)
            h, c = self.lstm.forward(x, h, c)

        # Persister les états (mémoire longue)
        self.h_state = h
        self.c_state = c

        d1  = self.dense1.forward(h)
        out = self.dense2.forward(d1)
        return float(np.clip(out[0], 0.0, 1.0))

    # ----------------------------------------------------------
    # API publique
    # ----------------------------------------------------------

    def get_risk(self) -> float:
        """Retourne le risque actuel"""
        return self.current_risk

    def get_threshold(self) -> float:
        """Retourne le seuil dynamique"""
        return self.current_threshold

    def is_at_risk(self, threshold: float = None) -> bool:
        """Vérifie si le nœud est à risque"""
        thr = threshold or FdqnConfig.PEPM_RISK_THRESHOLD
        return self.current_risk > thr

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du module"""
        steps_to_empty = (
            round(self.prev_energy / self._drain_rate)
            if self._drain_rate > 1e-6 and self.prev_energy is not None
            else None
        )
        return {
            "node_id":        self.node_id,
            "risk":           round(self.current_risk, 4),
            "threshold":      round(self.current_threshold, 4),
            "trend":          round(self.trend, 4),
            "drain_rate":     round(self._drain_rate, 6),
            "steps_to_empty": steps_to_empty,
            "energy":         self.prev_energy
        }

    # ----------------------------------------------------------
    # Persistance
    # ----------------------------------------------------------

    def save(self, path: str):
        """Sauvegarde l'état"""
        data = {
            "node_id": self.node_id,
            "h_state": self.h_state.tolist(),
            "c_state": self.c_state.tolist(),
            "energy_history": list(self.energy_history),
            "trend": self.trend,
            "current_risk": self.current_risk,
            "te_max": self.te_max,
            "alpha": self.alpha
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Charge l'état"""
        with open(path) as f:
            data = json.load(f)

        self.h_state = np.array(data["h_state"])
        self.c_state = np.array(data["c_state"])
        self.energy_history = deque(data["energy_history"], maxlen=self.window)
        self.trend = data.get("trend", 0.0)
        self.current_risk = data.get("current_risk", 0.5)
        self.current_threshold = self.current_risk * self.te_max


# ============================================================
# PEPM Pool (Gestionnaire de modules)
# ============================================================

class PEPMPool:
    """Pool de modules PEPM (un par nœud)"""

    def __init__(
        self,
        window: int = FdqnConfig.PEPM_WINDOW,
        hidden_dim: int = FdqnConfig.PEPM_HIDDEN,
        risk_threshold: float = FdqnConfig.PEPM_RISK_THRESHOLD,
        e_init = FdqnConfig.E_INIT
    ):
        self.modules: Dict[int, PEPMModule] = {}
        self.window = window
        self.hidden_dim = hidden_dim
        self.risk_threshold = risk_threshold
        self.e_init = e_init

    def get_or_create(self, node_id: int) -> PEPMModule:
        """Récupère ou crée un module pour un nœud"""
        if node_id not in self.modules:
            self.modules[node_id] = PEPMModule(
                node_id,
                window=self.window,
                hidden_dim=self.hidden_dim,
                e_init=self.e_init
            )
        return self.modules[node_id]

    def update_node(self, node_id: int, energy: float) -> float:
        """Met à jour un nœud et retourne son seuil"""
        module = self.get_or_create(node_id)
        return module.update(energy)

    def get_risk(self, node_id: int) -> float:
        """Retourne le risque d'un nœud"""
        module = self.modules.get(node_id)
        return module.get_risk() if module else 0.5

    def get_at_risk_nodes(self) -> List[int]:
        """Retourne la liste des nœuds à risque"""
        return [
            nid for nid, mod in self.modules.items()
            if mod.is_at_risk(self.risk_threshold)
        ]

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Statistiques de tous les modules"""
        return [mod.get_stats() for mod in self.modules.values()]

    def get_summary(self) -> Dict[str, Any]:
        """Résumé global"""
        if not self.modules:
            return {}

        risks = [mod.get_risk() for mod in self.modules.values()]
        at_risk = len(self.get_at_risk_nodes())

        return {
            "n_nodes": len(self.modules),
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "min_risk": float(np.min(risks)),
            "nodes_at_risk": at_risk
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=== Test PEPM (version corrigée) ===")

    pool = PEPMPool()
    energy = FdqnConfig.E_INIT  # ← depuis config (était 2.0 hardcodé)

    for t in range(100):
        # Simuler une décharge progressive
        energy -= 0.015
        if t > 50:
            energy -= 0.02  # Accélération

        threshold = pool.update_node(1, energy)
        risk = pool.get_risk(1)

        if t % 10 == 0:
            print(f"t={t:2d} | E={energy:.3f} | risk={risk:.3f} | thr={threshold:.4f} | at_risk={risk>0.7}")

    print("\nRésumé:", pool.get_summary())
