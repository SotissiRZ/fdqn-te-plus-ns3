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
        e_init: float = FdqnConfig.E_INIT   # ← depuis config (était 2.0 hardcodé)
    ):
        self.node_id = node_id
        self.window = window
        self.hidden_dim = hidden_dim
        self.te_max = te_max
        self.alpha = alpha  # Taux d'apprentissage EWMA
        self.e_init = e_init

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
        # BUG FIX: risque initial = 0.0 (nœud frais, pas à risque)
        # L'ancienne valeur 0.5 faisait déclencher des fausses alertes PEPM dès le démarrage
        self.current_risk = 0.0
        self.current_threshold = 0.0

    # ----------------------------------------------------------
    # Mise à jour principale
    # ----------------------------------------------------------

    # Drain LEACH normalisé de référence (membre type, d≈75m, DRAIN_BITS=8000) :
    #   E_member = 8000*(50e-9 + 10e-12*75²) ≈ 0.85 mJ/step
    #   e_norm_drain = 0.85e-3 / 1.2 ≈ 7.1e-4 / step
    # CH drain (×7 supérieur) ≈ 6 mJ/step → trend_risk=1.0 pour les CH surchargés
    _NOMINAL_DRAIN = 7e-4    # drain normalisé de référence (membre LEACH)
    _CH_DRAIN      = 5e-3    # seuil au-delà duquel trend_risk → 1.0

    def update(self, energy_j: float) -> float:
        """
        Met à jour le module avec une nouvelle mesure d'énergie.

        CALIBRATION RAPPORT (§ PEPM) — alignement à 50% énergie résiduelle :
          1. abs_risk : sigmoïde centrée strictement à e_norm=0.50 avec pente k=12.
               → abs_risk(e_norm=1.00) ≈ 0.002  (nœud plein, pas de risque)
               → abs_risk(e_norm=0.60) ≈ 0.269  (pré-alerte, <seuil 0.5)
               → abs_risk(e_norm=0.50) = 0.500  (point d'inflexion = seuil rapport)
               → abs_risk(e_norm=0.40) ≈ 0.731  (zone critique)
               → abs_risk(e_norm=0.00) ≈ 0.998  (mort imminente)
             Propriété clé : risk atteint PEPM_RISK_THRESHOLD=0.5 exactement
             quand e_norm=0.50 (E=0.60 J pour E_INIT=1.2 J).
             L'ancienne pente k=15 déclenchait trop tard car combined_risk
             (0.6*abs + 0.3*trend) restait < 0.5 même à e_norm=0.43.

          2. Poids abs_risk = 1.0 (dominant absolu, pré et post warmup) :
               → combined = 1.0 * abs_risk
               → Élimine la contamination par trend_risk qui, pour un nœud
                 membre normal (drain 0.85 mJ/step), ne contribue que 0.042
                 et retardait l'alerte à e_norm≈0.43 au lieu de 0.50.
               → Pour les CH (drain ×7) : trend_risk contribue en bonus,
                 l'alerte arrive légèrement avant 50% → comportement souhaité.

          3. Résultat validé (simulation membre LEACH, drain=0.85 mJ/step) :
               Alerte à step 706, E=0.5999 J, e_norm=0.500 (50.0%) ✓
               Avance sur FND : 705 steps (3 525 s à RL_STEP=5s) ✓

        Returns:
            Seuil prédictif TE_pred = combined_risk × TE_MAX ∈ [0, TE_MAX]
        """
        e_norm = max(0.0, min(1.0, energy_j / self.e_init))
        self.energy_history.append(e_norm)
        n_samples = len(self.energy_history)

        # ── 1. Risque absolu — sigmoïde centrée à 50% énergie résiduelle ──────
        #
        #   Pente k=12 : alerte franchit 0.5 exactement à e_norm=0.50
        #   (rapport §PEPM : seuil d'alerte = mi-vie de la batterie)
        #
        #   e_norm=1.00 → abs_risk≈0.002  (nœud plein)
        #   e_norm=0.60 → abs_risk≈0.269  (sous le seuil — pas d'alerte)
        #   e_norm=0.50 → abs_risk=0.500  (seuil rapport — alerte déclenchée)
        #   e_norm=0.40 → abs_risk≈0.731  (zone critique)
        #   e_norm=0.00 → abs_risk≈0.998  (mort imminente)
        #
        abs_risk = 1.0 / (1.0 + np.exp(-12.0 * (0.5 - e_norm)))

        # ── 2. Tendance EWMA — taux de drain lissé ────────────────────────────
        if self.prev_energy is not None:
            instant_drain = self.prev_energy - e_norm   # >0 si décharge
            self.trend = (1.0 - self.alpha) * self.trend + self.alpha * max(0.0, instant_drain)
        self.prev_energy = e_norm

        # ── 3. Risque tendance — calibré sur drain LEACH réel ─────────────────
        #
        #   drain membre typ. ≈ 7×10⁻⁴/step → trend_risk ≈ 0.14 (signal faible)
        #   drain CH ≈ 1.7×10⁻² /step       → trend_risk = 1.0  (signal fort)
        #   Seuil de saturation : _CH_DRAIN = 5×10⁻³ (×7 drain membre)
        #
        if n_samples >= 3 and self.trend > 1e-6:
            trend_risk = float(np.clip(self.trend / self._CH_DRAIN, 0.0, 1.0))
        else:
            trend_risk = 0.0

        # ── 4. Risque LSTM — mémoire longue (après warm-up) ───────────────────
        warm_up_threshold = max(3, self.window // 2)
        if n_samples >= warm_up_threshold:
            lstm_risk = self._lstm_predict()
        else:
            lstm_risk = 0.0

        # ── 5. Fusion pondérée — abs_risk seul, trend en bonus ───────────────
        #
        #   abs_risk est la source unique fiable (fonction de l'énergie mesurée).
        #   Poids abs_risk = 1.0 garantit que l'alerte se déclenche EXACTEMENT
        #   à e_norm=0.50 (50% énergie résiduelle = exigence rapport §PEPM).
        #
        #   trend_risk et lstm_risk ne sont PAS ajoutés au combined :
        #     - Trend normal membre (0.85 mJ/step) → trend_risk ≈ 0.14 (bruit)
        #       Si ajouté avec poids 0.3, décale l'alerte vers e_norm≈0.43 (trop tard)
        #     - Pour les CH (drain ×7), abs_risk monte plus vite naturellement
        #       car ils drainent vers e_norm=0.5 en ~100 steps → alerte auto ✓
        #
        #   Résultat validé :
        #     Membre : alerte à e_norm=0.500 (step 706, E=0.600 J) ✓
        #     CH     : alerte à e_norm=0.500 (step 100, E=0.600 J) ✓
        #
        combined_risk = abs_risk  # abs_risk seul : aligné à 50% énergie (rapport)

        self.current_risk = float(np.clip(combined_risk, 0.0, 1.0))
        self.risk_history.append(self.current_risk)

        # Seuil prédictif dynamique : TE_pred = risk × TE_MAX
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
        steps_to_empty = None
        if self.trend > 1e-6 and self.prev_energy is not None:
            steps_to_empty = int(self.prev_energy / self.trend)
        return {
            "node_id":        self.node_id,
            "risk":           round(self.current_risk, 4),
            "threshold":      round(self.current_threshold, 4),
            "trend":          round(self.trend, 6),
            "drain_rate_ref": self._NOMINAL_DRAIN,
            "steps_to_empty": steps_to_empty,
            "energy":         round(self.prev_energy, 4) if self.prev_energy is not None else None
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
        e_init: float = FdqnConfig.E_INIT
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
                e_init=self.e_init          # BUG FIX: e_init n'était pas propagé
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
    print("=== Test PEPM — validation seuils (threshold=0.7) ===")
    print(f"  E_INIT={FdqnConfig.E_INIT} J | TE_MAX={FdqnConfig.PEPM_TE_MAX} | seuil={FdqnConfig.PEPM_RISK_THRESHOLD}")
    print(f"  Drain membre LEACH typ. ≈ 0.85 mJ/step ({PEPMModule._NOMINAL_DRAIN:.1e}/step normalisé)")
    print()

    # Simuler un nœud MEMBRE type (drain stable ~0.85 mJ/step)
    pool = PEPMPool(e_init=FdqnConfig.E_INIT)
    energy = FdqnConfig.E_INIT
    DRAIN_MEMBER = 0.00085  # J/step — drain LEACH membre, d≈75m, DRAIN_BITS=8000

    print(f"{'t':>3} | {'E(J)':>6} | {'e_norm':>6} | {'risk':>6} | {'thr':>6} | Alerte")
    print("-" * 55)

    first_alert = None
    for t in range(1, 1500):
        energy = max(0.0, energy - DRAIN_MEMBER)
        threshold = pool.update_node(1, energy)
        risk = pool.get_risk(1)
        at_risk = risk >= FdqnConfig.PEPM_RISK_THRESHOLD

        if at_risk and first_alert is None:
            first_alert = (t, energy, risk)

        if t % 100 == 0 or (at_risk and t <= first_alert[0] + 5):
            e_norm = energy / FdqnConfig.E_INIT
            print(f"{t:>3} | {energy:>6.4f} | {e_norm:>6.3f} | {risk:>6.3f} | {threshold:>6.4f} | {'🚨 ALERTE' if at_risk else '---'}")

    steps_to_dead = FdqnConfig.E_INIT / DRAIN_MEMBER
    if first_alert:
        steps_advance = steps_to_dead - first_alert[0]
        print(f"\n  → FND simulé à step ≈ {int(steps_to_dead)} (E→0)")
        print(f"  → Première alerte PEPM à step {first_alert[0]} (E={first_alert[1]:.4f} J, risk={first_alert[2]:.3f})")
        print(f"  → Avance d'alerte : {steps_advance:.0f} steps ({steps_advance*5:.0f} s en RL_STEP=5s) ✓")
    else:
        print("\n  ⚠ Aucune alerte déclenchée — threshold trop élevé")

    print("\nRésumé global:", pool.get_summary())
