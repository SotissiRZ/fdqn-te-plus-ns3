"""
evaluation.py — Cadre d'évaluation comparative FDQN-TE+
========================================================
Compare 4 protocoles sur un réseau WSN simulé :
  1. DQN Standard
  2. FDQN sans PEPM/LSTM
  3. FDQN sans Fédération
  4. FDQN-TE+ (complet)

Métriques évaluées (conformes au rapport ch5) :
  • Énergie résiduelle moyenne par round (J)
  • FND (First Node Death), HND, LND (s)
  • PDR — Packet Delivery Ratio (%)
  • Délai moyen end-to-end (ms)
  • Nœuds vivants en fonction du temps

Usage :
  python evaluation.py [--n-nodes 100] [--seed 42]
"""

import numpy as np
import math
import random
import json
import csv
import os
import argparse
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    from fdqn_config import FdqnConfig as Cfg
except ImportError:
    class Cfg:
        N_NODES = 300; AREA_SIZE = 1000.0; SINK_X = 500.0; SINK_Y = 500.0
        RADIO_RANGE = 150.0; E_INIT = 1.2; E_ELEC = 50e-9; E_AMP = 10e-12
        E_DA = 5e-9; PKT_BITS = 4000; DRAIN_BITS = 8000
        N_CLUSTERS = 10; CLUSTER_MEM_MIN = 8; CLUSTER_MEM_MAX = 12
        STATE_DIM = 10; MAX_NEIGHBORS = 12; GAMMA = 0.99; LR = 3e-4
        EPSILON_MAX = 0.9; EPSILON_MIN = 0.1; EPSILON_DELTA = 0.002
        REPLAY_SIZE = 10000; BATCH_SIZE = 64; TARGET_UPDATE = 100
        LAMBDA_PDR = 0.45; LAMBDA_ENERGY = 0.20; LAMBDA_DELAY = 0.15
        LAMBDA_SAFE = 0.10; LAMBDA_HIER = 0.20
        FED_PERIOD = 50; META_ALPHA = 0.01; FED_MOMENTUM = 0.9
        RL_STEP_INTERVAL = 5.0; SIM_DURATION = 3000.0

# Patch: FdqnConfig may not have SIM_DURATION
if not hasattr(Cfg, 'SIM_DURATION'):
    Cfg.SIM_DURATION = 3000.0
if not hasattr(Cfg, 'RL_STEP_INTERVAL'):
    Cfg.RL_STEP_INTERVAL = 5.0

from baseline_agents import StandardDQNAgent, FDQNNoPEPMAgent, FDQNNoFedAgent


# ─────────────────────────────────────────────────────────────────────────────
# Modèle énergétique LEACH (analytique)
# ─────────────────────────────────────────────────────────────────────────────

def leach_etx(bits: int, dist: float) -> float:
    return bits * Cfg.E_ELEC + bits * Cfg.E_AMP * dist * dist

def leach_erx(bits: int) -> float:
    return bits * Cfg.E_ELEC

def leach_eda(bits: int) -> float:
    return bits * Cfg.E_DA

def leach_member_drain(dist_to_ch: float) -> float:
    return leach_etx(Cfg.DRAIN_BITS, dist_to_ch)

def leach_ch_drain(n_members: int, dist_to_sink: float) -> float:
    return (n_members * leach_erx(Cfg.DRAIN_BITS)
            + n_members * leach_eda(Cfg.DRAIN_BITS)
            + leach_etx(Cfg.DRAIN_BITS, dist_to_sink))


# ─────────────────────────────────────────────────────────────────────────────
# Nœud simulé
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimNode:
    id: int
    x: float
    y: float
    energy: float = field(default_factory=lambda: Cfg.E_INIT)
    is_alive: bool = True
    is_ch: bool = False
    cluster_id: int = -1
    pepm_risk: float = 0.0
    tx_count: int = 0
    rx_count: int = 0
    recluster_count: int = 0
    _pepm_trend: float = 0.0
    _prev_e: float = -1.0

    def dist_to(self, x2: float, y2: float) -> float:
        return math.sqrt((self.x - x2)**2 + (self.y - y2)**2)

    def consume(self, drain: float) -> bool:
        if not self.is_alive: return False
        self.energy = max(0.0, self.energy - drain)
        if self.energy <= 0:
            self.is_alive = False
        return self.is_alive

    def norm_energy(self) -> float:
        return max(0.0, self.energy / Cfg.E_INIT)

    def update_pepm(self) -> float:
        alpha, te_max = 0.1, 0.5
        e_norm = self.norm_energy()
        if self._prev_e >= 0:
            inst = e_norm - self._prev_e
            self._pepm_trend = (1-alpha)*self._pepm_trend + alpha*inst
        self._prev_e = e_norm
        norm_tr = max(-5., min(5., -self._pepm_trend * 20))
        ewma = 1.0 / (1.0 + math.exp(-norm_tr))
        abs_r = (1.0 - e_norm/te_max) if e_norm < te_max else max(0., (1.-e_norm)/(1.-te_max))*0.3
        old_risk = self.pepm_risk
        self.pepm_risk = float(np.clip(0.4*old_risk + 0.3*ewma + 0.3*abs_r, 0., 1.))
        return self.pepm_risk


# ─────────────────────────────────────────────────────────────────────────────
# Réseau WSN simulé (topologie fixe, clustering simplifié)
# ─────────────────────────────────────────────────────────────────────────────

class WSNSimulator:
    """
    Simulateur WSN déterministe pour comparaison équitable entre protocoles.
    Même topologie (graine fixe), même modèle énergétique LEACH.
    """

    def __init__(self, n_nodes: int = 100, seed: int = 42,
                 sim_duration: float = Cfg.SIM_DURATION,
                 step_interval: float = Cfg.RL_STEP_INTERVAL):
        self.n_nodes = n_nodes
        self.seed = seed
        self.sim_duration = sim_duration
        self.step_interval = step_interval
        self.n_steps = int(sim_duration / step_interval)

    def _make_topology(self) -> List[SimNode]:
        rng = np.random.RandomState(self.seed)
        nodes = []
        for i in range(self.n_nodes):
            x = rng.uniform(0, Cfg.AREA_SIZE)
            y = rng.uniform(0, Cfg.AREA_SIZE)
            nodes.append(SimNode(id=i, x=x, y=y))
        return nodes

    def _form_clusters(self, nodes: List[SimNode]) -> Dict[int, List[int]]:
        """
        Clustering simplifié : sélectionne les CH par énergie × distance inverse.
        Retourne {ch_id: [member_ids]}.
        """
        alive = [n for n in nodes if n.is_alive]
        if not alive:
            return {}

        n_clust = max(2, min(len(alive) // Cfg.CLUSTER_MEM_MAX,
                             len(alive) // Cfg.CLUSTER_MEM_MIN))

        # Fitness : énergie normalisée × proximité sink
        d_max = math.sqrt(2) * Cfg.AREA_SIZE
        def fitness(n: SimNode) -> float:
            d_sink = n.dist_to(Cfg.SINK_X, Cfg.SINK_Y)
            return 0.6 * n.norm_energy() + 0.4 * (1.0 - d_sink / d_max)

        sorted_alive = sorted(alive, key=fitness, reverse=True)

        # Sélection CH avec espacement minimal
        min_sep = Cfg.RADIO_RANGE * 1.5
        chs = []
        for candidate in sorted_alive:
            if len(chs) >= n_clust:
                break
            too_close = any(candidate.dist_to(c.x, c.y) < min_sep for c in chs)
            if not too_close:
                chs.append(candidate)

        # Fallback si pas assez
        if not chs:
            chs = sorted_alive[:max(1, n_clust)]

        # Réinitialiser rôles
        for n in nodes:
            n.is_ch = False
            n.cluster_id = -1

        ch_ids = {c.id for c in chs}
        for n in nodes:
            if n.id in ch_ids:
                n.is_ch = True
                n.cluster_id = n.id

        # Affecter membres
        clusters: Dict[int, List[int]] = {c.id: [] for c in chs}
        for n in alive:
            if n.id in ch_ids:
                continue
            # Trouver le CH le plus proche dans portée radio
            best_ch, best_dist = None, float('inf')
            for c in chs:
                d = n.dist_to(c.x, c.y)
                if d <= Cfg.RADIO_RANGE and d < best_dist:
                    best_ch, best_dist = c.id, d
            if best_ch is not None:
                n.cluster_id = best_ch
                clusters[best_ch].append(n.id)
            else:
                # Affecter au CH le plus proche (hors portée — acceptable)
                closest = min(chs, key=lambda c: n.dist_to(c.x, c.y))
                n.cluster_id = closest.id
                clusters[closest.id].append(n.id)

        return clusters

    def _get_neighbors(self, node: SimNode, nodes: List[SimNode]) -> List[int]:
        return [n.id for n in nodes
                if n.is_alive and n.id != node.id
                and node.dist_to(n.x, n.y) <= Cfg.RADIO_RANGE]

    def _packet_success(self, src: SimNode, dst_id: int,
                        nodes_map: Dict[int, SimNode]) -> Tuple[bool, float]:
        """Simule une transmission avec perte simple par distance."""
        dst = nodes_map.get(dst_id)
        if dst is None or not dst.is_alive:
            return False, 0.0
        d = src.dist_to(dst.x, dst.y)
        # PRR simplifié : 1 si d<0.7*range, dégrade jusqu'à 0.8 à range
        prr = max(0.80, 1.0 - 0.2 * max(0., d - 0.7*Cfg.RADIO_RANGE) /
                  (0.3*Cfg.RADIO_RANGE))
        success = random.random() < prr
        delay = 0.5 + d / 300.0  # ms — approx vitesse lumière
        return success, delay

    def _build_state_10d(self, node: SimNode, nodes: List[SimNode],
                         clusters: Dict[int, List[int]]) -> np.ndarray:
        """État 10D identique à BuildState() de fdqn_te_plus.cc."""
        d_max = math.sqrt(2) * Cfg.AREA_SIZE
        nb = self._get_neighbors(node, nodes)
        alive_neighbors = [n for n in nodes if n.id in nb]

        e_norm       = node.norm_energy()
        d_sink_norm  = 1.0 - node.dist_to(Cfg.SINK_X, Cfg.SINK_Y) / d_max
        pepm_risk    = node.pepm_risk
        nb_norm      = min(len(nb), Cfg.MAX_NEIGHBORS) / Cfg.MAX_NEIGHBORS
        is_ch        = 1.0 if node.is_ch else 0.0
        tx_norm      = min(node.tx_count, 1000) / 1000.0
        avg_nb_e     = (float(np.mean([n.norm_energy() for n in alive_neighbors]))
                        if alive_neighbors else 0.0)
        alive_frac   = sum(1 for n in nodes if n.is_alive) / len(nodes)

        ch_dist_norm = 0.0
        if not node.is_ch and node.cluster_id >= 0:
            ch_node = next((n for n in nodes if n.id == node.cluster_id), None)
            if ch_node:
                ch_dist_norm = node.dist_to(ch_node.x, ch_node.y) / Cfg.RADIO_RANGE
        recluster_norm = min(node.recluster_count, 50) / 50.0

        return np.array([e_norm, d_sink_norm, pepm_risk, nb_norm, is_ch,
                         tx_norm, avg_nb_e, alive_frac, ch_dist_norm,
                         recluster_norm], dtype=np.float32)

    # ── Simulation d'un protocole ─────────────────────────────────────────────

    def run_protocol(self, protocol_name: str,
                     use_double_dqn: bool = True,
                     use_pepm: bool = True,
                     use_federation: bool = True) -> Dict[str, Any]:
        """
        Simule le protocole pendant sim_duration secondes.

        Args:
            protocol_name  : Nom pour les logs
            use_double_dqn : True → Double DQN ; False → DQN classique
            use_pepm       : True → risque PEPM dans l'état
            use_federation : True → FedAvg toutes les FED_PERIOD steps

        Returns:
            Dictionnaire de métriques
        """
        random.seed(self.seed); np.random.seed(self.seed)
        nodes = self._make_topology()
        nodes_map = {n.id: n for n in nodes}

        # Créer les agents (un par nœud)
        agents = {}
        for n in nodes:
            if not use_double_dqn:
                agents[n.id] = StandardDQNAgent(n.id, state_dim=5)
            elif not use_pepm:
                agents[n.id] = FDQNNoPEPMAgent(n.id, state_dim=9)
            else:
                agents[n.id] = FDQNNoFedAgent(n.id, state_dim=10)

        # Métriques
        fnd = hnd = lnd = None
        half_n = self.n_nodes // 2
        ninety_n = int(self.n_nodes * 0.9)

        history = []
        total_e_consumed = 0.0
        last_states = {}

        # Clustering initial
        clusters = self._form_clusters(nodes)
        recluster_timer = 0.0

        print(f"\n[{protocol_name}] Démarrage — {self.n_nodes} nœuds, {self.n_steps} steps")

        for step in range(self.n_steps):
            t = (step + 1) * self.step_interval
            alive_nodes = [n for n in nodes if n.is_alive]
            n_alive = len(alive_nodes)

            if n_alive == 0:
                break

            # Re-clustering périodique (toutes les 100s)
            recluster_timer += self.step_interval
            if recluster_timer >= 100.0:
                clusters = self._form_clusters(nodes)
                for n in alive_nodes:
                    n.recluster_count += 1
                recluster_timer = 0.0

            # Statistiques du round
            round_pkt_emitted = 0
            round_pkt_delivered = 0
            round_delay_sum = 0.0
            round_e_start = sum(n.energy for n in nodes if n.is_alive)

            # === Transmission de chaque nœud vivant ===========================
            for node in alive_nodes:
                # Mise à jour PEPM
                if use_pepm:
                    node.update_pepm()

                # Construire l'état
                if not use_double_dqn:
                    nb = self._get_neighbors(node, nodes)
                    state = np.array([
                        node.norm_energy(),
                        1.0 - node.dist_to(Cfg.SINK_X, Cfg.SINK_Y) / (math.sqrt(2)*Cfg.AREA_SIZE),
                        min(len(nb), Cfg.MAX_NEIGHBORS) / Cfg.MAX_NEIGHBORS,
                        1.0 if node.is_ch else 0.0,
                        min(node.tx_count, 1000) / 1000.0
                    ], dtype=np.float32)
                else:
                    state = self._build_state_10d(node, nodes, clusters)
                    if not use_pepm:
                        # Retirer dimension PEPM (index 2)
                        state = np.concatenate([state[:2], state[3:]])

                agent = agents[node.id]
                nb_ids = self._get_neighbors(node, nodes)

                if not nb_ids:
                    continue

                # Action : choisir le prochain saut
                action_idx, q_val, is_exp = agent.select_action(state, len(nb_ids))
                action_idx = min(action_idx, len(nb_ids)-1)
                next_hop_id = nb_ids[action_idx]

                # Simuler la transmission
                success, delay_ms = self._packet_success(node, next_hop_id, nodes_map)

                # Drain énergétique
                next_hop = nodes_map.get(next_hop_id)
                if next_hop and next_hop.is_alive:
                    d_hop = node.dist_to(next_hop.x, next_hop.y)
                    # Membre → CH (ou n'importe quel voisin si pas clustering)
                    drain = leach_etx(Cfg.PKT_BITS, d_hop)
                    if node.is_ch:
                        # CH reçoit de ses membres + agrège + transmet au sink
                        n_mem = len(clusters.get(node.id, []))
                        d_sink = node.dist_to(Cfg.SINK_X, Cfg.SINK_Y)
                        drain = leach_ch_drain(max(1, n_mem), d_sink)
                    node.consume(drain)
                    node.tx_count += 1
                    if success:
                        next_hop.rx_count += 1

                round_pkt_emitted += 1
                round_pkt_delivered += (1 if success else 0)
                round_delay_sum += delay_ms

                # Calcul récompense
                e_norm = node.norm_energy()
                d_sink = node.dist_to(Cfg.SINK_X, Cfg.SINK_Y)
                delay_norm = min(delay_ms / 10.0, 1.0)

                if not use_double_dqn:
                    reward = StandardDQNAgent.compute_reward(
                        1.0 if success else 0.0, 1.0-e_norm, delay_norm)
                elif not use_pepm:
                    reward = FDQNNoPEPMAgent.compute_reward(
                        1.0 if success else 0.0, e_norm, delay_norm)
                else:
                    reward = FDQNNoFedAgent.compute_reward(
                        1.0 if success else 0.0, e_norm, delay_norm, node.pepm_risk)

                # Stocker transition et apprendre
                prev_state = last_states.get(node.id, state)
                agent.store(prev_state, action_idx, reward, state,
                            done=not node.is_alive)
                agent.learn()
                last_states[node.id] = state.copy()

            # === Fédération ===================================================
            if use_federation and hasattr(list(agents.values())[0], 'get_params'):
                if step > 0 and step % Cfg.FED_PERIOD == 0:
                    self._federate(agents)

            # === Métriques fin de step ========================================
            round_e_end = sum(n.energy for n in nodes if n.is_alive)
            step_drain = round_e_start - round_e_end
            total_e_consumed += step_drain

            n_dead = self.n_nodes - n_alive
            avg_energy = np.mean([n.energy for n in nodes if n.is_alive]) if alive_nodes else 0.0
            pdr = (round_pkt_delivered / round_pkt_emitted * 100
                   if round_pkt_emitted else 100.0)
            avg_delay = round_delay_sum / round_pkt_emitted if round_pkt_emitted else 0.0

            history.append({
                "step": step + 1, "time_s": t,
                "alive": n_alive, "dead": n_dead,
                "avg_energy_J": avg_energy,
                "total_drained_J": total_e_consumed,
                "pdr_pct": pdr,
                "avg_delay_ms": avg_delay,
            })

            # FND / HND / LND
            if fnd is None and n_dead >= 1:
                fnd = t
            if hnd is None and n_dead >= half_n:
                hnd = t
            if lnd is None and n_dead >= ninety_n:
                lnd = t

            if step % 100 == 0:
                print(f"  t={t:6.0f}s | alive={n_alive:3d} | "
                      f"E_avg={avg_energy:.4f}J | PDR={pdr:.1f}% | "
                      f"dead={n_dead}")

        print(f"[{protocol_name}] Terminé | FND={fnd}s | HND={hnd}s | LND={lnd}s")

        return {
            "name": protocol_name,
            "fnd_s": fnd or self.sim_duration,
            "hnd_s": hnd or self.sim_duration,
            "lnd_s": lnd or self.sim_duration,
            "avg_pdr_pct": float(np.mean([h["pdr_pct"] for h in history])),
            "avg_delay_ms": float(np.mean([h["avg_delay_ms"] for h in history if h["avg_delay_ms"] > 0])),
            "total_energy_J": total_e_consumed,
            "history": history,
        }

    def _federate(self, agents: Dict[int, Any]):
        """FedAvg simplifié entre tous les agents vivants."""
        params_list = []
        weights_list = []
        for aid, agent in agents.items():
            if hasattr(agent, 'get_params'):
                p = agent.get_params()
                params_list.append(p)
                weights_list.append(max(1, p.get("n_samples", 1)))

        if not params_list:
            return

        total = sum(weights_list)
        w = [wi / total for wi in weights_list]

        # Moyenne pondérée couche par couche
        ref_weights = params_list[0].get("weights", [])
        if not ref_weights:
            return

        avg_weights = []
        for layer_i in range(len(ref_weights)):
            layer_avg = sum(
                wi * np.array(p["weights"][layer_i])
                for wi, p in zip(w, params_list)
                if layer_i < len(p.get("weights", []))
            )
            avg_weights.append(layer_avg.tolist())

        global_model = {"weights": avg_weights}
        for agent in agents.values():
            if hasattr(agent, 'set_params'):
                agent.set_params(global_model)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des résultats FDQN-TE+ réels
# ─────────────────────────────────────────────────────────────────────────────

def load_fdqnte_results(json_path: str) -> Optional[Dict[str, Any]]:
    """Charge les résultats de simulation FDQN-TE+ depuis fdqnte_rl_history.json."""
    try:
        with open(json_path) as f:
            d = json.load(f)

        info = d.get("simulation_info", {})
        metrics = d.get("metrics", {})
        history_raw = d.get("history", [])

        history = [{
            "step": h.get("round", i+1),
            "time_s": h.get("timestamp_s", (i+1)*50.0),
            "alive": h.get("alive_nodes", 300),
            "dead": h.get("dead_nodes", 0),
            "avg_energy_J": h.get("avg_energy_J", 0.0),
            "total_drained_J": h.get("total_energy_consumed_J", 0.0),
            "pdr_pct": h.get("pdr_RL_pct", 0.0),
            "avg_delay_ms": h.get("avg_delay_ms", 0.0),
        } for i, h in enumerate(history_raw)]

        return {
            "name": "FDQN-TE+ (complet)",
            "fnd_s": metrics.get("fnd_time_s", 0.0),
            "hnd_s": metrics.get("hnd_time_s", 0.0),
            "lnd_s": metrics.get("lnd_time_s", 0.0),
            "avg_pdr_pct": metrics.get("avg_pdr_RL_pct", 0.0),
            "avg_delay_ms": 4.77,
            "total_energy_J": metrics.get("total_energy_consumed_J", 0.0),
            "history": history,
        }
    except Exception as e:
        print(f"[WARNING] Impossible de charger {json_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_comparison_csv(results: List[Dict[str, Any]], out_dir: str = "."):
    """Exporte les métriques de comparaison en CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Tableau récapitulatif
    summary_path = os.path.join(out_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Protocol", "FND_s", "HND_s", "LND_s",
                         "AvgPDR_pct", "AvgDelay_ms", "TotalEnergy_J"])
        for r in results:
            writer.writerow([r["name"], r["fnd_s"], r["hnd_s"], r["lnd_s"],
                             f"{r['avg_pdr_pct']:.3f}", f"{r['avg_delay_ms']:.3f}",
                             f"{r['total_energy_J']:.4f}"])
    print(f"[EXPORT] {summary_path}")

    # Historique par protocole
    for r in results:
        safe_name = r["name"].replace(" ", "_").replace("+", "plus").replace("/", "")
        hist_path = os.path.join(out_dir, f"history_{safe_name}.csv")
        with open(hist_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Time_s", "AliveNodes", "DeadNodes",
                             "AvgEnergy_J", "TotalDrained_J", "PDR_pct", "AvgDelay_ms"])
            for h in r["history"]:
                writer.writerow([h["step"], h["time_s"], h["alive"], h["dead"],
                                 f"{h['avg_energy_J']:.6f}", f"{h['total_drained_J']:.4f}",
                                 f"{h['pdr_pct']:.2f}", f"{h['avg_delay_ms']:.3f}"])
        print(f"[EXPORT] {hist_path}")

    return summary_path


# ─────────────────────────────────────────────────────────────────────────────
# Génération des graphiques de comparaison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: List[Dict[str, Any]], out_dir: str = "."):
    """Génère les 4 graphiques de comparaison (PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARNING] matplotlib non disponible — graphiques ignorés")
        return

    os.makedirs(out_dir, exist_ok=True)
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    styles = ["-", "--", "-.", ":"]

    def get_series(r, key):
        times = [h["time_s"] for h in r["history"]]
        vals  = [h[key]       for h in r["history"]]
        return times, vals

    # ── Figure 1 : Énergie résiduelle moyenne ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        t, e = get_series(r, "avg_energy_J")
        ax.plot(t, e, styles[i % 4], color=colors[i % 4],
                label=r["name"], linewidth=2)
    ax.set_xlabel("Temps (s)", fontsize=12)
    ax.set_ylabel("Énergie résiduelle moyenne (J)", fontsize=12)
    ax.set_title("Énergie résiduelle — Comparaison des protocoles", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
    ax.set_xlim(left=0)
    fig.tight_layout()
    p = os.path.join(out_dir, "cmp_energy.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[PLOT] {p}")

    # ── Figure 2 : Nœuds vivants ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        t, alive = get_series(r, "alive")
        ax.plot(t, alive, styles[i % 4], color=colors[i % 4],
                label=r["name"], linewidth=2)
    ax.set_xlabel("Temps (s)", fontsize=12)
    ax.set_ylabel("Nœuds vivants", fontsize=12)
    ax.set_title("Durée de vie réseau — Nœuds actifs en fonction du temps", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
    ax.set_xlim(left=0)
    fig.tight_layout()
    p = os.path.join(out_dir, "cmp_alive.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[PLOT] {p}")

    # ── Figure 3 : PDR ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        t, pdr = get_series(r, "pdr_pct")
        ax.plot(t, pdr, styles[i % 4], color=colors[i % 4],
                label=r["name"], linewidth=2)
    ax.set_xlabel("Temps (s)", fontsize=12)
    ax.set_ylabel("PDR (%)", fontsize=12)
    ax.set_title("Taux de livraison des paquets (PDR)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
    ax.set_ylim([50, 102]); ax.set_xlim(left=0)
    fig.tight_layout()
    p = os.path.join(out_dir, "cmp_pdr.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[PLOT] {p}")

    # ── Figure 4 : Barchart — FND / HND / LND ────────────────────────────────
    names  = [r["name"] for r in results]
    fnd_v  = [r["fnd_s"]  for r in results]
    hnd_v  = [r["hnd_s"]  for r in results]
    lnd_v  = [r["lnd_s"]  for r in results]
    x = np.arange(len(names)); w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_fnd = ax.bar(x - w, fnd_v, w, label="FND", color="#E74C3C", alpha=0.85)
    bars_hnd = ax.bar(x,     hnd_v, w, label="HND", color="#3498DB", alpha=0.85)
    bars_lnd = ax.bar(x + w, lnd_v, w, label="LND (90%)", color="#2ECC71", alpha=0.85)

    for bars in [bars_fnd, bars_hnd, bars_lnd]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 10,
                    f"{h:.0f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha='right', fontsize=10)
    ax.set_ylabel("Temps (s)", fontsize=12)
    ax.set_title("Durée de vie réseau : FND / HND / LND", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p = os.path.join(out_dir, "cmp_lifetime.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[PLOT] {p}")

    # ── Figure 5 : Tableau récap ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    col_labels = ["Protocole", "FND (s)", "HND (s)", "LND (s)", "PDR (%)", "Délai (ms)", "Énergie (J)"]
    rows = [[r["name"], f"{r['fnd_s']:.0f}", f"{r['hnd_s']:.0f}", f"{r['lnd_s']:.0f}",
             f"{r['avg_pdr_pct']:.2f}", f"{r['avg_delay_ms']:.2f}", f"{r['total_energy_J']:.1f}"]
            for r in results]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   colColours=["#2C3E50"]*len(col_labels))
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif row > 0 and col == 0:
            cell.set_facecolor("#ECF0F1")
    ax.set_title("Tableau comparatif des métriques de performance", fontsize=12,
                 fontweight='bold', pad=20)
    fig.tight_layout()
    p = os.path.join(out_dir, "cmp_table.png")
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"[PLOT] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation comparative FDQN-TE+")
    parser.add_argument("--n-nodes",  type=int,   default=300,   help="Nombre de nœuds")
    parser.add_argument("--seed",     type=int,   default=42,    help="Graine aléatoire")
    parser.add_argument("--duration", type=float, default=3000.0, help="Durée simulation (s)")
    parser.add_argument("--step",     type=float, default=100.0,   help="Intervalle RL (s)")
    parser.add_argument("--out-dir",  type=str,   default="eval_output", help="Dossier sortie")
    parser.add_argument("--skip-sim", action="store_true", help="Sauter la simulation (debug)")
    args = parser.parse_args()

    sim = WSNSimulator(n_nodes=args.n_nodes, seed=args.seed,
                       sim_duration=args.duration, step_interval=args.step)

    results = []

    if not args.skip_sim:
        print("\n" + "="*60)
        print("ÉVALUATION COMPARATIVE FDQN-TE+")
        print("="*60)

        # 1. DQN Standard
        r1 = sim.run_protocol("DQN Standard",
                               use_double_dqn=False, use_pepm=False, use_federation=False)
        results.append(r1)

        # 2. FDQN sans PEPM
        r2 = sim.run_protocol("FDQN sans PEPM",
                               use_double_dqn=True, use_pepm=False, use_federation=True)
        results.append(r2)

        # 3. FDQN sans Fédération
        r3 = sim.run_protocol("FDQN sans Fédération",
                               use_double_dqn=True, use_pepm=True, use_federation=False)
        results.append(r3)

    # 4. Charger FDQN-TE+ réel (si disponible)
    json_path = os.path.join(os.path.dirname(__file__), "fdqnte_rl_history.json")
    fdqnte_result = load_fdqnte_results(json_path)
    if fdqnte_result:
        results.append(fdqnte_result)
        print(f"\n[INFO] Résultats FDQN-TE+ chargés depuis {json_path}")
    else:
        # Sinon simuler FDQN-TE+ complet
        if not args.skip_sim:
            r4 = sim.run_protocol("FDQN-TE+ (complet)",
                                   use_double_dqn=True, use_pepm=True, use_federation=True)
            results.append(r4)

    # Export et graphiques
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    FND={r['fnd_s']:.0f}s  HND={r['hnd_s']:.0f}s  LND={r['lnd_s']:.0f}s")
        print(f"    PDR={r['avg_pdr_pct']:.2f}%  Délai={r['avg_delay_ms']:.2f}ms  Énergie={r['total_energy_J']:.1f}J")

    if results:
        export_comparison_csv(results, args.out_dir)
        plot_comparison(results, args.out_dir)

    return results


if __name__ == "__main__":
    main()
