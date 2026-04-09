"""
rl_server.py — Serveur RL pour FDQN-TE+ (NS-3)
"""

import json
import socket
import threading
import time
import os
import sys
from typing import Dict, Any, Optional
from collections import deque

# Import des modules
try:
    from fdqn_config import FdqnConfig
    from addqn_agent import AgentPool
    from pepm_lstm import PEPMPool
    from fedmeta_drl import FederatedOrchestrator
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que tous les modules sont dans le même dossier")
    sys.exit(1)


# ============================================================
# Serveur FDQN-TE+
# ============================================================

class FDQNServer:
    """
    Serveur central pour FDQN-TE+
    Gère les connexions NS-3 et coordonne ADDQN, PEPM et FedMeta
    """

    def __init__(self):
        # Configuration
        self.e_init = FdqnConfig.E_INIT    # Energie initial
        self.n_nodes = FdqnConfig.N_NODES  # Nombred de Noeuds
        self.n_clusters = FdqnConfig.N_CLUSTERS # Nombre de clusters MAX

        # Pool d'agents ADDQN - adapter les paramètres à addqn_agent.py
        self.agent_pool = AgentPool(
            state_dim=FdqnConfig.STATE_DIM,
            max_neighbors=FdqnConfig.MAX_NEIGHBORS,
            gamma=FdqnConfig.GAMMA,
            lr=FdqnConfig.LR,
            epsilon_max=FdqnConfig.EPSILON_MAX,
            epsilon_min=FdqnConfig.EPSILON_MIN,
            epsilon_delta=FdqnConfig.EPSILON_DELTA,   # décroissance linéaire
            epsilon_decay=FdqnConfig.EPSILON_DECAY,
            batch_size=FdqnConfig.BATCH_SIZE,
            replay_size=FdqnConfig.REPLAY_SIZE,
            tau=FdqnConfig.TAU,
            lr_min=1e-5,
            lr_decay=0.0005,
            target_update=FdqnConfig.TARGET_UPDATE
        )

        # Pool PEPM — e_init transmis pour normalisation
        self.pepm_pool = PEPMPool(
            window=FdqnConfig.PEPM_WINDOW,
            hidden_dim=FdqnConfig.PEPM_HIDDEN,
            risk_threshold=FdqnConfig.PEPM_RISK_THRESHOLD,
            e_init=FdqnConfig.E_INIT
        )

        # Orchestrateur fédéré
        self.fed_orch = FederatedOrchestrator(
            fed_period=FdqnConfig.FED_PERIOD
        )

        # Buffer pour les clusters (reçu de NS-3)
        self.clusters = []

        # Cache pour les derniers états (nécessaire pour store_transition)
        self.last_states  = {}      # node_id -> state
        self.last_actions = {}      # node_id -> action_idx
        self.dead_nodes   = set()    # LEACH-morts (n'envoient plus de récompenses)

        # Statistiques globales
        self.stats = {
            "global_step": 0,
            "actions_requested": 0,
            "rewards_received": 0,
            "pepm_queries": 0,
            "loss_history": deque(maxlen=1000),
            "reward_history": deque(maxlen=1000)
        }

        # Lock réentrant pour éviter les deadlocks intra-dispatch
        self.lock = threading.RLock()

        print(f"\n{'='*60}")
        print(f"FDQN-TE+ Server - Démarrage")
        print(f"{'='*60}")
        print(f"  ADDQN: {FdqnConfig.STATE_DIM} états, {FdqnConfig.MAX_NEIGHBORS} actions max")
        print(f"  PEPM: fenêtre={FdqnConfig.PEPM_WINDOW}, seuil risque={FdqnConfig.PEPM_RISK_THRESHOLD}")
        print(f"  FedMeta: période={FdqnConfig.FED_PERIOD}, méta-α={FdqnConfig.META_ALPHA}")
        print(f"  Port: {FdqnConfig.RL_PORT}")
        print(f"{'='*60}\n")

    # --------------------------------------------------------
    # Dispatch des messages
    # --------------------------------------------------------

    def dispatch(self, msg_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route un message vers le bon handler
        """
        with self.lock:
            if msg_type == "init":
                return self._handle_init(payload)
            elif msg_type == "action":
                return self._handle_action(payload)
            elif msg_type == "reward":
                return self._handle_reward(payload)
            elif msg_type == "pepm":
                return self._handle_pepm(payload)
            elif msg_type == "pepm_batch":
                return self._handle_pepm_batch(payload)
            elif msg_type in ("cluster", "topology"):
                return self._handle_cluster(payload)
            elif msg_type == "stats":
                return self._handle_stats()
            else:
                return {"error": f"Type inconnu: {msg_type}"}

    # --------------------------------------------------------
    # Handlers
    # --------------------------------------------------------

    def _handle_init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Initialisation de la simulation"""
        self.e_init = float(payload.get("init_energy", FdqnConfig.E_INIT))
        self.n_nodes = int(payload.get("n_nodes", FdqnConfig.N_NODES))
        self.n_clusters = int(payload.get("n_clusters", FdqnConfig.N_CLUSTERS))

        print(f"\n[INIT] Simulation démarrée")
        print(f"  Énergie initiale: {self.e_init} J")
        print(f"  Nombre de nœuds: {self.n_nodes}")
        print(f"  Nombre de clusters cible: {self.n_clusters}")

        return {"ok": True}

    def _handle_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Demande d'action de routage"""
        node_id = int(payload.get("node_id", 0))
        state = payload.get("state", [0] * FdqnConfig.STATE_DIM)
        neighbors = payload.get("neighbors", [])

        if not neighbors:
            return {"next_hop": node_id, "q_value": 0.0, "action_index": 0}

        # Clamp neighbors à MAX_NEIGHBORS — le réseau Q a max_actions sorties
        # Les voisins au-delà sont éligibles via exploration aléatoire normale
        max_n = FdqnConfig.MAX_NEIGHBORS
        neighbors_capped = neighbors[:max_n]

        # Récupérer l'agent
        agent = self.agent_pool.get(node_id)

        # Convertir l'état en numpy array (attendu par l'agent)
        import numpy as np
        state_array = np.array(state, dtype=np.float32)

        # Sélectionner l'action (sur neighbors_capped, jamais > max_n)
        action_idx, q_value, is_exploration = agent.select_action(state_array, neighbors_capped)

        # Double sécurité : borner l'index
        action_idx = min(action_idx, len(neighbors_capped) - 1)

        # Sauvegarder pour la récompense (on sauvegarde l'index dans la liste complète)
        self.last_states[node_id] = np.array(state_array, dtype=np.float32)
        self.last_actions[node_id] = action_idx

        self.stats["actions_requested"] += 1
        self.fed_orch.step_advance()

        return {
            "next_hop": int(neighbors_capped[action_idx]),
            "q_value": float(q_value),
            "action_index": int(action_idx)
        }

    def _handle_reward(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Soumission de récompense"""
        node_id = int(payload.get("node_id", 0))
        action_idx = int(payload.get("action_idx", payload.get("action", 0)))
        reward = float(payload.get("reward", 0.0))
        next_state = payload.get("next_state", [0] * FdqnConfig.STATE_DIM)

        # Récupérer l'agent
        agent = self.agent_pool.get(node_id)

        # Récupérer l'état précédent et l'action
        last_state = self.last_states.get(node_id)
        last_action = self.last_actions.get(node_id, action_idx)

        if last_state is not None:
            import numpy as np
            next_array = np.array(next_state, dtype=np.float32)
            done_flag  = bool(payload.get("done", False))
            agent.store_transition(last_state, last_action, reward, next_array, done=done_flag)
            # enregistrer la mort du nœud
            if done_flag:
                self.dead_nodes.add(node_id)

        # Apprendre
        loss = agent.learn()

        # Statistiques
        self.stats["rewards_received"] += 1
        self.stats["loss_history"].append(loss)
        self.stats["reward_history"].append(reward)
        self.stats["global_step"] += 1

        # Vérifier si round fédéré — lancer hors lock pour éviter deadlock
        should_federate = self.fed_orch.should_run() and bool(self.clusters)

        return_val = {
            "ok": True,
            "loss": float(loss)
        }

        if should_federate:
            threading.Thread(target=self._run_federation, daemon=True).start()

        return return_val

    def _handle_pepm_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        nodes_payload = payload.get("nodes", [])
        risks: Dict[int, float] = {}
        at_risk: list = []

        for entry in nodes_payload:
            nid    = int(entry.get("node_id", 0))
            energy = float(entry.get("energy", self.e_init))
            energy = max(0.0, min(energy, self.e_init))

            # update_node retourne le seuil, mais on veut le risque
            self.pepm_pool.update_node(nid, energy)
            risk = self.pepm_pool.get_risk(nid)  #
            risks[nid] = float(max(0.0, min(1.0, risk)))  # clamp
            if risk > FdqnConfig.PEPM_RISK_THRESHOLD:
                at_risk.append(nid)

        self.stats["pepm_queries"] += len(nodes_payload)

        return {
            "risks":    {str(k): v for k, v in risks.items()},
            "at_risk":  at_risk,
            "n_updated": len(nodes_payload)
        }

    def _handle_pepm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demande de prédiction PEPM pour un nœud.
        Reçoit uniquement node_id + energy (en Joules).
        Le module pepm_lstm.py gère tout l'historique et le modèle LSTM.
        """
        node_id = int(payload.get("node_id", 0))
        energy  = float(payload.get("energy", self.e_init))

        # Borne défensive : l'énergie ne peut pas dépasser E_INIT ni être négative
        energy = max(0.0, min(energy, self.e_init))

        # Déléguer au module PEPMModule via PEPMPool
        # update_node() appelle PEPMModule.update() → LSTM + EWMA + abs_risk
        threshold = self.pepm_pool.update_node(node_id, energy)
        risk      = self.pepm_pool.get_risk(node_id)

        # Validation de sortie
        risk = float(max(0.0, min(1.0, risk)))
        is_at_risk = risk > FdqnConfig.PEPM_RISK_THRESHOLD

        self.stats["pepm_queries"] += 1

        return {
            "risk":       risk,
            "threshold":  float(threshold),
            "is_at_risk": bool(is_at_risk)
        }

    def _handle_cluster(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mise à jour de la topologie des clusters (depuis IFO via NS-3)"""
        raw = payload.get("clusters", [])

        # NS-3 envoie {"ch":X, "members":[...]}
        # fedmeta_drl attend {"chId":X, "memberIds":[...]}
        normalized = []
        for c in raw:
            normalized.append({
                "chId":      c.get("chId",      c.get("ch", 0)),
                "memberIds": c.get("memberIds", c.get("members", []))
            })
        self.clusters = normalized

        print(f"\n[CLUSTER] Reçu {len(self.clusters)} clusters")

        # La fédération est déclenchée dans _handle_reward (sur should_run())
        # et non ici pour éviter une boucle sur chaque mise à jour topologie

        return {"ok": True}

    def _handle_stats(self) -> Dict[str, Any]:
        """Statistiques du serveur"""
        # Calcul inline — agent_pool.get_stats() n'existe pas dans cette version
        import numpy as np
        agents = list(self.agent_pool.agents.values()) \
                 if hasattr(self.agent_pool, "agents") else []
        mean_epsilon = float(np.mean([a.epsilon for a in agents])) if agents else 1.0
        n_agents     = len(agents)
        buffer_size  = sum(len(a.memory) for a in agents)

        pepm_summary = self.pepm_pool.get_summary()
        fed_stats    = self.fed_orch.server.get_stats()

        # Calculer la loss moyenne récente
        recent_losses = list(self.stats["loss_history"])[-50:]
        mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0

        return {
            "global_step":   self.stats["global_step"],
            "fed_round":     fed_stats.get("current_round", 0),
            "epsilon":       mean_epsilon,
            "mean_loss":     float(mean_loss),
            "mean_pepm":     pepm_summary.get("mean_risk", 0.0),
            "nodes_at_risk": pepm_summary.get("nodes_at_risk", 0),
            "n_agents":      n_agents,
            "buffer_size":   buffer_size,
        }

    # --------------------------------------------------------
    # Fédération
    # --------------------------------------------------------

    def _run_federation(self):
        """Exécute un round fédéré (dans un thread séparé)"""
        try:
            with self.lock:
                if not self.clusters:
                    return

                n_active   = len([nid for nid in self.agent_pool.agents
                                   if nid not in self.dead_nodes]) \
                             if hasattr(self.agent_pool, "agents") else self.n_nodes
                n_dead     = len(self.dead_nodes)
                n_clusters = len(self.clusters)
                fed_round  = self.fed_orch.server.round

                print(f"[FedMeta] Round {fed_round} | {n_clusters} clusters | "
                      f"{n_active}/{self.n_nodes} nodes vivants | {n_dead} morts")

                # ── Snapshot des paramètres actifs ────────────────────────────
                agents_snap = {}
                if hasattr(self.agent_pool, "agents"):
                    for nid, agent in self.agent_pool.agents.items():
                        if nid not in self.dead_nodes:
                            agents_snap[nid] = agent.get_model_params()

                if not agents_snap:
                    return

                clusters_snap = list(self.clusters)

            # ── Agrégation intra-cluster (hors lock — calcul NumPy) ───────────
            from fedmeta_drl import fed_avg

            cluster_models = {}
            for cluster in clusters_snap:
                ch_id    = cluster["chId"]
                all_ids  = [ch_id] + cluster.get("memberIds", [])
                models   = []
                samples  = []
                for mid in all_ids:
                    if mid in agents_snap:
                        p = agents_snap[mid]
                        models.append({"weights": p.get("weights", [])})
                        samples.append(p.get("n_samples", 1))
                if models:
                    agg = fed_avg(models, samples)
                    cluster_models[ch_id] = {
                        "weights":   agg.get("weights", []),
                        "n_samples": sum(samples),
                    }

            if not cluster_models:
                return

            # ── Agrégation globale ────────────────────────────────────────────
            all_w = [{"weights": v["weights"]}  for v in cluster_models.values()]
            all_s = [v["n_samples"]             for v in cluster_models.values()]
            global_w = fed_avg(all_w, all_s).get("weights", [])

            if not global_w:
                return

            # ── Distribution du modèle global (sous lock) ─────────────────────
            # Remplace broadcast_global_params() — boucle directe sur les agents
            with self.lock:
                distributed = 0
                if hasattr(self.agent_pool, "agents"):
                    for nid, agent in self.agent_pool.agents.items():
                        if nid not in self.dead_nodes:
                            agent.set_model_params({"weights": global_w})
                            distributed += 1

                # Avancer le compteur fédéral de l'orchestrateur
                self.fed_orch.server.round += 1

                n_dead = len(self.dead_nodes)
                print(f"[FedMeta] Round terminé — actifs={distributed}/{self.n_nodes} "
                      f"morts={n_dead}")

        except Exception as e:
            print(f"[FedMeta] Erreur: {e}")

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    def export_history(self, path: str = "fdqnte_rl_history.json"):
        """Exporte l'historique des métriques"""
        data = {
            "config": {
                "e_init": self.e_init,
                "n_nodes": self.n_nodes, "n_alive": len([nid for nid in self.agent_pool.agents if nid not in self.dead_nodes]) if hasattr(self.agent_pool, "agents") else self.n_nodes,
                "gamma": FdqnConfig.GAMMA,
                "lr": FdqnConfig.LR,
                "epsilon_decay": FdqnConfig.EPSILON_DECAY,
                "fed_period": FdqnConfig.FED_PERIOD
            },
            "stats": {
                "global_step": self.stats["global_step"],
                "actions_requested": self.stats["actions_requested"],
                "rewards_received": self.stats["rewards_received"],
                "pepm_queries": self.stats["pepm_queries"]
            },
            "history": {
                "loss": list(self.stats["loss_history"]),
                "reward": list(self.stats["reward_history"])
            },
            "pepm_summary": self.pepm_pool.get_summary(),
            "fed_stats": self.fed_orch.server.get_stats()
        }

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n[EXPORT] Historique sauvegardé dans {path}")
        except Exception as e:
            print(f"\n[EXPORT] Erreur: {e}")


# ============================================================
# Gestionnaire de connexion client
# ============================================================

class ClientHandler(threading.Thread):
    """Gère une connexion client NS-3"""

    def __init__(self, conn: socket.socket, addr: tuple, server: FDQNServer):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.server = server

    def run(self):
        buf = ""
        try:
            while True:
                data = self.conn.recv(4096)
                if not data:
                    break

                buf += data.decode(errors="replace")

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                        msg_type = payload.get("type", payload.get("cmd", ""))

                        # Traiter la requête
                        response = self.server.dispatch(msg_type, payload)

                        # Envoyer la réponse
                        self.conn.sendall((json.dumps(response) + "\n").encode())

                    except json.JSONDecodeError as e:
                        print(f"JSON invalide: {e}")
                        self.conn.sendall((json.dumps({"error": "Invalid JSON"}) + "\n").encode())
                    except Exception as e:
                        print(f"Erreur: {e}")
                        self.conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())

        except Exception as e:
            print(f"Erreur connexion {self.addr}: {e}")
        finally:
            self.conn.close()


# ============================================================
# Fonction principale
# ============================================================

def run_server(host: str = "0.0.0.0", port: int = FdqnConfig.RL_PORT):
    """Lance le serveur"""

    # S'assurer que le port est bien un entier
    if isinstance(port, str):
        port = int(port)

    server = FDQNServer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"Erreur bind {host}:{port} - {e}")
        print("Vérifiez que le port n'est pas déjà utilisé")
        return

    sock.listen(32)
    print(f"\n✅ Serveur FDQN-TE+ en écoute sur {host}:{port}")
    print("   Appuyez sur Ctrl+C pour arrêter\n")

    # Export périodique (toutes les 30 secondes)
    def periodic_export():
        while True:
            time.sleep(30)
            try:
                server.export_history()
            except Exception as e:
                print(f"Export error: {e}")

    export_thread = threading.Thread(target=periodic_export, daemon=True)
    export_thread.start()

    try:
        while True:
            conn, addr = sock.accept()
            print(f"\n[CONNEXION] Client {addr}")
            ClientHandler(conn, addr, server).start()

    except KeyboardInterrupt:
        print("\n\nArrêt demandé...")
    finally:
        print("Export final...")
        server.export_history()
        sock.close()
        print("Serveur arrêté")


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    run_server()
