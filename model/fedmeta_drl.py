"""
fedmeta_drl.py — FedMeta-DRL (Federated Meta Deep Reinforcement Learning)
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

try:
    from fdqn_config import FdqnConfig
except ImportError:
    class FdqnConfig:
        FED_PERIOD = 50
        META_ALPHA = 0.01
        FED_MOMENTUM = 0.9


# ============================================================
# FedAvg Aggregation
# ============================================================

def fed_avg(models: List[Dict[str, Any]], n_samples: List[int]) -> Dict[str, Any]:
    """
    Agrégation par moyenne pondérée (Federated Averaging).

    Robuste aux cas dégénérés :
      - modèles sans clé "weights" ou weights vide → ignorés
      - n_layers incohérentes entre modèles → aligné sur le minimum
      - n_samples vide ou tous nuls → poids uniformes

    Args:
        models:    Liste de dicts {"weights": [[W, b], ...]}
        n_samples: Nombre d'échantillons par modèle (même longueur que models)

    Returns:
        {"weights": [...]} agrégé, ou {} si aucun modèle valide
    """
    if not models:
        return {}

    # Filtrer les modèles avec des weights exploitables
    valid = [
        (m, s) for m, s in zip(models, n_samples)
        if m.get("weights") and len(m["weights"]) > 0
    ]
    if not valid:
        return {}

    valid_models, valid_samples = zip(*valid)

    total = sum(valid_samples) or 1
    agg_weights = [s / total for s in valid_samples]

    # Nombre de couches : minimum commun pour éviter IndexError
    n_layers = min(len(m["weights"]) for m in valid_models)
    if n_layers == 0:
        return {}

    avg_weights = []
    for layer_idx in range(n_layers):
        try:
            layer_arrays = [
                np.array(m["weights"][layer_idx], dtype=np.float64)
                for m in valid_models
            ]
            # Vérifier que toutes les couches ont la même forme
            shapes = [a.shape for a in layer_arrays]
            if len(set(shapes)) > 1:
                # Formes incompatibles → ignorer cette couche (ne pas planter)
                avg_weights.append(layer_arrays[0].tolist())
                continue
            layer_avg = sum(w * arr for w, arr in zip(agg_weights, layer_arrays))
            avg_weights.append(layer_avg.tolist())
        except (IndexError, TypeError, ValueError):
            # Couche corrompue → garder la première valeur disponible
            avg_weights.append(valid_models[0]["weights"][layer_idx])

    return {"weights": avg_weights}


# ============================================================
# FedMeta-DRL Server
# ============================================================

class FedMetaDRLServer:
    """
    Serveur d'agrégation fédérée avec méta-apprentissage

    Architecture à deux niveaux:
    1. Intra-cluster: agrégation des membres par CH
    2. Global: agrégation des CH par le sink
    """

    def __init__(
        self,
        fed_period: int = FdqnConfig.FED_PERIOD,
        meta_alpha: float = FdqnConfig.META_ALPHA,
        momentum: float = FdqnConfig.FED_MOMENTUM
    ):
        self.fed_period = fed_period
        self.meta_alpha = meta_alpha
        self.momentum = momentum

        self.round = 0
        self.global_model: Optional[Dict[str, Any]] = None
        self.prev_model: Optional[Dict[str, Any]] = None

        # Buffers pour les modèles
        self.node_models: Dict[int, Dict[str, Any]] = {}      # node_id → model
        self.cluster_models: Dict[int, Dict[str, Any]] = {}   # cluster_id → model

        # Historique
        self.round_history = deque(maxlen=100)

    # --------------------------------------------------------
    # Réception des modèles
    # --------------------------------------------------------

    def receive_node_model(self, node_id: int, params: Dict[str, Any]):
        # Reçoit le modèle d'un nœud
        self.node_models[node_id] = params.copy()

    def receive_cluster_model(self, cluster_id: int, params: Dict[str, Any]):
        # Reçoit le modèle d'un cluster (CH)
        self.cluster_models[cluster_id] = params.copy()

    # --------------------------------------------------------
    # Agrégation intra-cluster
    # --------------------------------------------------------

    def aggregate_cluster(self, cluster_id: int, member_ids: List[int]) -> Dict[str, Any]:
        """
        Agrège les modèles des membres d'un cluster

        Args:
            cluster_id: ID du cluster
            member_ids: Liste des IDs des membres (incluant le CH)

        Returns:
            Modèle agrégé du cluster
        """
        member_models = []
        member_samples = []

        for mid in member_ids:
            if mid not in self.node_models:
                continue

            model = self.node_models[mid]
            member_models.append({"weights": model.get("weights", [])})
            member_samples.append(model.get("n_samples", 1))

        if not member_models:
            return {}

        aggregated = fed_avg(member_models, member_samples)

        cluster_model = {
            "cluster_id": cluster_id,
            "n_samples": sum(member_samples),
            "weights": aggregated["weights"],
            "round": self.round
        }

        self.cluster_models[cluster_id] = cluster_model
        return cluster_model

    # --------------------------------------------------------
    # Agrégation globale
    # --------------------------------------------------------

    def aggregate_global(self) -> Dict[str, Any]:
        """
        Agrège tous les modèles de clusters en un modèle global

        Returns:
            Modèle global
        """
        if not self.cluster_models:
            return self.global_model or {}

        models = list(self.cluster_models.values())
        samples = [m.get("n_samples", 1) for m in models]

        # Agrégation FedAvg
        raw_global = fed_avg(
            [{"weights": m["weights"]} for m in models],
            samples
        )

        # Méta-adaptation
        if self.global_model and raw_global:
            raw_global = self._meta_adapt(self.global_model, raw_global)

        # Momentum smoothing
        if self.prev_model and raw_global:
            raw_global = self._apply_momentum(self.prev_model, raw_global)

        # Sauvegarder l'ancien modèle
        self.prev_model = self.global_model

        # Nouveau modèle global
        self.global_model = {
            "weights": raw_global["weights"],
            "round": self.round,
            "n_clusters": len(self.cluster_models),
            "total_samples": sum(samples)
        }

        # Historique
        self.round_history.append({
            "round": self.round,
            "n_clusters": len(self.cluster_models),
            "total_samples": sum(samples)
        })

        self.round += 1
        return self.global_model

    # --------------------------------------------------------
    # Méta-apprentissage
    # --------------------------------------------------------

    def _meta_adapt(self, prev: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Méta-adaptation pour convergence rapide
        θ_new = θ_prev + α * (θ_raw - θ_prev)
        """
        prev_w = prev.get("weights", [])
        new_w = new.get("weights", [])

        if not prev_w or not new_w or len(prev_w) != len(new_w):
            return new

        adapted = []
        for p, n in zip(prev_w, new_w):
            p_arr = np.array(p)
            n_arr = np.array(n)
            adapted.append((p_arr + self.meta_alpha * (n_arr - p_arr)).tolist())

        return {"weights": adapted}

    def _apply_momentum(self, prev: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique un momentum pour lisser les mises à jour
        θ = β * θ_prev + (1-β) * θ_current
        """
        prev_w = prev.get("weights", [])
        curr_w = current.get("weights", [])

        if not prev_w or not curr_w or len(prev_w) != len(curr_w):
            return current

        smoothed = []
        for p, c in zip(prev_w, curr_w):
            p_arr = np.array(p)
            c_arr = np.array(c)
            smoothed.append((self.momentum * p_arr + (1 - self.momentum) * c_arr).tolist())

        return {"weights": smoothed}

    # --------------------------------------------------------
    # Utilitaires
    # --------------------------------------------------------

    def should_aggregate(self, step: int) -> bool:
        """Vérifie si une agrégation doit avoir lieu"""
        return step > 0 and step % self.fed_period == 0

    def reset_round_buffers(self):
        """Réinitialise les buffers pour un nouveau round"""
        self.node_models.clear()
        self.cluster_models.clear()

    def get_global_model(self) -> Dict[str, Any]:
        """Retourne le modèle global actuel"""
        return self.global_model or {}

    # --------------------------------------------------------
    # Persistance
    # --------------------------------------------------------

    def save_checkpoint(self, path: str):
        # Sauvegarde un checkpoint du serveur
        checkpoint = {
            "round": self.round,
            "global_model": self.global_model,
            "prev_model": self.prev_model,
            "history": list(self.round_history),
            "config": {
                "fed_period": self.fed_period,
                "meta_alpha": self.meta_alpha,
                "momentum": self.momentum
            }
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, path: str):
        # Charge un checkpoint
        with open(path) as f:
            cp = json.load(f)

        self.round = cp["round"]
        self.global_model = cp["global_model"]
        self.prev_model = cp["prev_model"]
        self.round_history = deque(cp["history"], maxlen=100)

    def get_stats(self) -> Dict[str, Any]:
        # Statistiques du serveur
        return {
            "current_round": self.round,
            "clusters_in_buffer": len(self.cluster_models),
            "nodes_in_buffer": len(self.node_models),
            "history": list(self.round_history)[-5:],
            "has_global_model": self.global_model is not None
        }


# ============================================================
# Orchestrator
# ============================================================

class FederatedOrchestrator:
    # Orchestrateur qui coordonne les rounds fédérés

    def __init__(self, fed_period: int = FdqnConfig.FED_PERIOD):
        self.server = FedMetaDRLServer(fed_period)
        self.step = 0

    def step_advance(self):
        # Avance d'un pas de simulation
        self.step += 1

    def should_run(self) -> bool:
        # Vérifie si un round doit être exécuté
        return self.server.should_aggregate(self.step)

    def run_federation_round(
        self,
        agent_pool: Any,  # AgentPool
        clusters: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Exécute un round complet d'agrégation fédérée

        Args:
            agent_pool: Pool d'agents ADDQN
            clusters: Liste des clusters [{chId, memberIds}]

        Returns:
            Modèle global
        """
        self.server.reset_round_buffers()

        # 1. Collecter les modèles des nœuds
        for params in agent_pool.get_all_params():
            self.server.receive_node_model(params["node_id"], params)

        # 2. Agrégation intra-cluster
        for cluster in clusters:
            ch_id = cluster["chId"]
            member_ids = cluster.get("memberIds", [])
            all_ids = [ch_id] + member_ids
            self.server.aggregate_cluster(ch_id, all_ids)

        # 3. Agrégation globale
        global_model = self.server.aggregate_global()

        # 4. Distribuer le modèle global
        if global_model:
            agent_pool.broadcast_global_params(global_model)

        stats = self.server.get_stats()
        print(f"[FedMeta] Round {stats['current_round']} | "
              f"{stats['clusters_in_buffer']} clusters | "
              f"{stats['nodes_in_buffer']} nodes")

        return global_model

    def save_checkpoint(self, path: str):
        """Sauvegarde l'état de l'orchestrateur"""
        self.server.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Charge l'état de l'orchestrateur"""
        self.server.load_checkpoint(path)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=== Test FedMeta-DRL ===")

    # Simuler des clusters
    clusters = [
        {"chId": 1, "memberIds": [2, 3, 4]},
        {"chId": 5, "memberIds": [6, 7, 8, 9]},
        {"chId": 10, "memberIds": [11, 12]}
    ]

    # Simuler des modèles
    class MockAgentPool:
        def get_all_params(self):
            return [
                {"node_id": i, "weights": [[[0.1]] * 10]}  # Simplifié
                for i in range(1, 13)
            ]
        def broadcast_global_params(self, params):
            pass

    orchestrator = FederatedOrchestrator(fed_period=10)

    for step in range(1, 101):
        orchestrator.step_advance()

        if orchestrator.should_run():
            print(f"\nStep {step}: Lancement round fédéré")
            orchestrator.run_federation_round(MockAgentPool(), clusters)

    # Sauvegarde
    orchestrator.save_checkpoint("test_checkpoint.json")
    print("\nCheckpoint sauvegardé")
