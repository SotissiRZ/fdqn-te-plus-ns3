"""
rl_server_dqnleach.py — Serveur RL pour DQN-LEACH
===================================================
Baseline DRL externe : ADDQN (Double DQN) seul.
Pas d'IFO, pas de PEPM, pas de fédération.
Clustering : LEACH probabiliste standard (géré côté C++).
Port : 5559 (dédié DQN-LEACH)

Objectif dans l'évaluation :
  Mesure l'apport des 3 composants IFO+PEPM+Fed combinés.
  Si FDQN-TE+ >> DQN-LEACH : les 3 composants sont tous utiles.
  Complète les ablations unitaires (noIFO, noPEPM, noFed).
"""

import json
import socket
import threading
import time
import os
import sys
from typing import Dict, Any
from collections import deque

try:
    from fdqn_config import FdqnConfig
    from addqn_agent import AgentPool
except ImportError as e:
    print(f"Erreur d'import: {e}")
    sys.exit(1)

# Port dédié DQN-LEACH (différent des autres serveurs)
PORT = 5559


class DQNLeachServer:
    """
    Serveur DQN-LEACH : ADDQN seul, sans PEPM ni fédération.
    - L'agent DQN apprend le routage optimal
    - Le clustering est géré par LEACH probabiliste côté C++
    - pepm_batch → risk=0.0 (pas de PEPM)
    - reward → no federation (federated=False toujours)
    """

    def __init__(self):
        self.agent_pool = AgentPool(
            state_dim=FdqnConfig.STATE_DIM,
            max_neighbors=FdqnConfig.MAX_NEIGHBORS,
            gamma=FdqnConfig.GAMMA,
            lr=FdqnConfig.LR,
            epsilon_max=FdqnConfig.EPSILON_MAX,
            epsilon_min=FdqnConfig.EPSILON_MIN,
            epsilon_delta=FdqnConfig.EPSILON_DELTA,
            epsilon_decay=FdqnConfig.EPSILON_DECAY,
            batch_size=FdqnConfig.BATCH_SIZE,
            replay_size=FdqnConfig.REPLAY_SIZE,
            tau=FdqnConfig.TAU,
            lr_min=1e-5,
            lr_decay=0.0005,
            target_update=FdqnConfig.TARGET_UPDATE
        )

        self.last_states  = {}
        self.last_actions = {}
        self.dead_nodes   = set()
        self.clusters     = []

        self.stats = {
            "global_step": 0,
            "actions_requested": 0,
            "rewards_received": 0,
            "loss_history": deque(maxlen=1000),
            "reward_history": deque(maxlen=1000),
        }

        self.lock = threading.RLock()

        print(f"\n{'='*60}")
        print(f"DQN-LEACH Server (Baseline DRL externe)")
        print(f"  ADDQN seul — sans IFO, sans PEPM, sans Fédération")
        print(f"  Clustering : LEACH probabiliste (géré côté C++)")
        print(f"  Port : {PORT}")
        print(f"{'='*60}\n")

    def dispatch(self, msg_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if msg_type == "action":
                return self._handle_action(payload)
            elif msg_type == "reward":
                return self._handle_reward(payload)
            elif msg_type in ("pepm", "pepm_batch"):
                # PEPM désactivé : risk=0.0 fixe
                return self._handle_pepm_null(payload, msg_type)
            elif msg_type in ("cluster", "topology"):
                return self._handle_cluster(payload)
            elif msg_type == "stats":
                return self._handle_stats()
            else:
                return {"error": f"Type inconnu: {msg_type}"}

    def _handle_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id   = int(payload.get("node_id", 0))
        state     = payload.get("state", [0] * FdqnConfig.STATE_DIM)
        neighbors = payload.get("neighbors", [])

        if not neighbors:
            return {"next_hop": node_id, "q_value": 0.0, "action_index": 0}

        import numpy as np
        neighbors_capped = neighbors[:FdqnConfig.MAX_NEIGHBORS]
        agent = self.agent_pool.get(node_id)
        state_array = np.array(state, dtype=np.float32)
        action_idx, q_value, _ = agent.select_action(state_array, neighbors_capped)
        action_idx = min(action_idx, len(neighbors_capped) - 1)

        self.last_states[node_id]  = np.array(state_array, dtype=np.float32)
        self.last_actions[node_id] = action_idx
        self.stats["actions_requested"] += 1

        return {
            "next_hop":     int(neighbors_capped[action_idx]),
            "q_value":      float(q_value),
            "action_index": int(action_idx)
        }

    def _handle_reward(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id    = int(payload.get("node_id", 0))
        action_idx = int(payload.get("action_idx", payload.get("action", 0)))
        reward     = float(payload.get("reward", 0.0))
        next_state = payload.get("next_state", [0] * FdqnConfig.STATE_DIM)

        agent      = self.agent_pool.get(node_id)
        last_state = self.last_states.get(node_id)
        last_action= self.last_actions.get(node_id, action_idx)

        if last_state is not None:
            import numpy as np
            next_array = np.array(next_state, dtype=np.float32)
            done_flag  = bool(payload.get("done", False))
            agent.store_transition(last_state, last_action, reward,
                                   next_array, done=done_flag)
            if done_flag:
                self.dead_nodes.add(node_id)

        loss = agent.learn()
        self.stats["rewards_received"] += 1
        self.stats["global_step"] += 1
        self.stats["loss_history"].append(loss)
        self.stats["reward_history"].append(reward)

        # Pas de fédération — federated=False toujours
        return {
            "ok":        True,
            "loss":      float(loss),
            "federated": False
        }

    def _handle_pepm_null(self, payload: Dict[str, Any],
                          msg_type: str) -> Dict[str, Any]:
        """PEPM désactivé : retourne risk=0.0 sans calcul."""
        if msg_type == "pepm_batch":
            nodes = payload.get("nodes", [])
            risks = {str(int(e.get("node_id", 0))): 0.0 for e in nodes}
            return {"risks": risks, "at_risk": [], "n_updated": len(nodes)}
        else:
            return {"risk": 0.0, "threshold": 0.0, "is_at_risk": False}

    def _handle_cluster(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mise à jour topologie (reçue de C++ mais non utilisée pour fédération)."""
        raw = payload.get("clusters", [])
        self.clusters = [
            {"chId": c.get("chId", c.get("ch", 0)),
             "memberIds": c.get("memberIds", c.get("members", []))}
            for c in raw
        ]
        return {"ok": True}

    def _handle_stats(self) -> Dict[str, Any]:
        agents = list(self.agent_pool.agents.values()) \
                 if hasattr(self.agent_pool, "agents") else []
        import numpy as np
        mean_eps = float(np.mean([a.epsilon for a in agents])) if agents else 1.0
        recent   = list(self.stats["loss_history"])[-50:]
        mean_loss= sum(recent) / len(recent) if recent else 0.0
        return {
            "global_step": self.stats["global_step"],
            "epsilon":     mean_eps,
            "mean_loss":   float(mean_loss),
            "n_agents":    len(agents),
        }

    def export_history(self, path: str = "dqn_leach_history.json"):
        data = {
            "model": "DQN-LEACH",
            "description": "ADDQN seul — sans IFO, PEPM, Fédération",
            "stats": {
                "global_step":       self.stats["global_step"],
                "actions_requested": self.stats["actions_requested"],
                "rewards_received":  self.stats["rewards_received"],
            },
            "history": {
                "loss":   list(self.stats["loss_history"]),
                "reward": list(self.stats["reward_history"]),
            }
        }
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n[EXPORT] {path}")
        except Exception as e:
            print(f"\n[EXPORT] Erreur: {e}")


class ClientHandler(threading.Thread):
    def __init__(self, conn, addr, server):
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
                        payload  = json.loads(line)
                        msg_type = payload.get("type", payload.get("cmd", ""))
                        response = self.server.dispatch(msg_type, payload)
                        self.conn.sendall(
                            (json.dumps(response) + "\n").encode())
                    except json.JSONDecodeError as e:
                        self.conn.sendall(
                            (json.dumps({"error": "Invalid JSON"}) + "\n").encode())
                    except Exception as e:
                        self.conn.sendall(
                            (json.dumps({"error": str(e)}) + "\n").encode())
        except Exception as e:
            print(f"Erreur connexion {self.addr}: {e}")
        finally:
            self.conn.close()


def run_server(host: str = "0.0.0.0", port: int = PORT):
    server = DQNLeachServer()
    sock   = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"Erreur bind {host}:{port} — {e}")
        return

    sock.listen(32)
    print(f"\n✅ DQN-LEACH Server en écoute sur {host}:{port}")
    print("   Ctrl+C pour arrêter\n")

    def periodic_export():
        while True:
            time.sleep(30)
            try:
                server.export_history()
            except Exception:
                pass

    threading.Thread(target=periodic_export, daemon=True).start()

    try:
        while True:
            conn, addr = sock.accept()
            print(f"\n[CONNEXION] {addr}")
            ClientHandler(conn, addr, server).start()
    except KeyboardInterrupt:
        print("\nArrêt demandé...")
    finally:
        print("Export final...")
        server.export_history()
        sock.close()
        print("Serveur arrêté")


if __name__ == "__main__":
    run_server()
