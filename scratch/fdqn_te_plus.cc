/* =============================================================================
 * fdqn_te_plus.cc — Simulation FDQN-TE+
 *
 * Architecture modulaire :
 *   modules/fdqn_config.h   — Tous les paramètres (source unique de vérité)
 *   modules/leach_energy.h  — Modèle énergétique LEACH (Heinzelman 2002)
 *   modules/node_state.h    — Structures NodeState / ClusterInfo partagées
 *   modules/ifo_clustering.h — Algorithme IFO (header-only)
 *
 * Sorties dans results/ :
 *   results/topology/   fdqnte_topology.csv      (état initial + après chaque recluster)
 *   results/energy/     fdqnte_energy.csv         (énergie par round en J)
 *   results/routing/    fdqnte_routing.csv         (traces de routage)
 *   results/rl/         fdqnte_rl_history.json     (HISTORIQUE RL)
 *   results/            fdqnte_summary.csv         (résumé final)
 *   results/            comparison_metrics.csv     (métriques pour comparaison article)
 *
 * Usage :
 *   Copier scratch/ et modules/ dans <ns3>/scratch/
 *   ./ns3 run scratch/fdqn_te_plus
 *   ./ns3 run "scratch/fdqn_te_plus --nNodes=400 --initEnergy=1.2 --simDuration=3500 --areaSize=1000"
 * ============================================================================= */

// ── Modules NS-3 ──────────────────────────────────────────────────────────────
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/energy-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/olsr-module.h"

// ── Socket POSIX ────────────────────────────────────────────────────────────
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>

// ── STL ───────────────────────────────────────────────────────────────────────
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <filesystem>
#include <chrono>

// ── Modules locaux ────────────────────────────────────────────────────────────
#include "fdqn_config.h"
#include "leach_energy.h"
#include "node_state.h"
#include "ifo_clustering.h"

using namespace ns3;
namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("FdqnTePlus");

// =============================================================================
// SECTION 1 — PARAMÈTRES CLI
// =============================================================================

struct SimParams {
    uint32_t nNodes        = FdqnCfg::N_NODES;
    double   areaSize      = FdqnCfg::AREA_SIZE;
    double   sinkX         = FdqnCfg::SINK_X;
    double   sinkY         = FdqnCfg::SINK_Y;
    double   radioRange    = FdqnCfg::RADIO_RANGE;
    double   initEnergy    = FdqnCfg::E_INIT;
    uint32_t nClusters     = 0;
    uint32_t ifoIterations = FdqnCfg::IFO_ITER;
    double   simDuration   = FdqnCfg::SIM_DURATION;
    uint32_t seed          = 42;
    std::string resultsDir = "results";
};

// =============================================================================
// SECTION 2 — VARIABLES GLOBALES
// =============================================================================

static std::set<uint32_t> g_deadNodes;
static bool     g_fndDone = false, g_hndDone = false, g_lndDone = false;
static double   g_fndTime = 0.0,   g_hndTime = 0.0,   g_lndTime = 0.0;
static uint32_t g_nNodes  = FdqnCfg::N_NODES;

// =============================================================================
// SECTION 3 — STRUCTURES POUR MÉTRIQUES ET HISTORIQUE RL
// =============================================================================

struct RLHistoryEntry {
    double timestamp;
    uint32_t round;
    uint32_t aliveNodes;
    uint32_t deadNodes;
    double avgEnergy_J;
    double totalEnergyConsumed_J;
    double pdr_RL_pct;
    double pdr_NS3_pct;
    double avgDelay_ms;
    uint32_t rlSteps;
    uint32_t fedRound;
    uint32_t nClusters;
    uint32_t atRiskPEPM;
    std::vector<double> rewardDistribution;
    std::vector<double> qValueDistribution;
};

static std::vector<RLHistoryEntry> g_rlHistory;

struct ComparisonMetrics {
    // Energy metrics - en JOULES
    std::vector<double> energyConsumption_J;
    std::vector<double> totalEnergyDrained_J;

    // Network lifetime metrics
    double fndTime = 0.0;
    double hndTime = 0.0;
    double lndTime = 0.0;
    uint32_t totalRounds = 0;

    // PDR metrics
    std::vector<double> pdrRLPerRound;
    std::vector<double> pdrNS3PerRound;
    double avgPDR_RL = 0.0;
    double avgPDR_NS3 = 0.0;

    // PDR restreint à la phase stable (avant FND) — référence article
    uint64_t pdrRL_preFND_emitted   = 0;
    uint64_t pdrRL_preFND_delivered = 0;
    bool     pdrRL_preFND_locked    = false;  // vrai dès que FND est atteint
    double   avgPDR_RL_preFND       = 100.0;  // valeur finale calculée dans main()

    // Delay metrics
    std::vector<double> endToEndDelay_ms;
    double avgDelay_ms = 0.0;

    // Node status
    std::vector<uint32_t> aliveNodesPerRound;
    std::vector<uint32_t> deadNodesPerRound;
    std::vector<uint32_t> atRiskPEPMPerRound;

    // RL metrics
    std::vector<uint32_t> rlStepsPerRound;
    std::vector<uint32_t> fedRoundsPerRound;

    // Additional metrics
    double totalEnergyConsumed_J = 0.0;
    uint64_t totalPacketsSent = 0;
    uint64_t totalPacketsReceived = 0;
    double simulationStartTime = 0.0;
    double simulationEndTime = 0.0;
};

static ComparisonMetrics g_compMetrics;

// =============================================================================
// SECTION 4 — BRIDGE TCP → SERVEUR PYTHON RL (AVEC PEPM)
// =============================================================================

class RLBridge {
public:
    RLBridge() : m_sock(-1), m_connected(false), m_steps(0), m_fedRound(0) {}

    ~RLBridge() { if (m_sock >= 0) close(m_sock); }

    bool Connect(const char* host = "127.0.0.1", int port = FdqnCfg::RL_PORT) {
        m_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (m_sock < 0) return false;

        struct timeval tv { 2, 0 };
        setsockopt(m_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(m_sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port);
        addr.sin_addr.s_addr = inet_addr(host);

        if (connect(m_sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
            close(m_sock); m_sock = -1;
            NS_LOG_UNCOND("[RL] ⚠ Serveur Python indisponible → mode Q-table fallback");
            return false;
        }
        m_connected = true;
        NS_LOG_UNCOND("[RL] ✓ Connecté à " << host << ":" << port);
        return true;
    }

    bool IsConnected() const { return m_connected; }

    bool TryReconnect(const char* host = "127.0.0.1", int port = FdqnCfg::RL_PORT) {
        if (m_connected) return true;
        if (m_sock >= 0) { ::close(m_sock); m_sock = -1; }
        m_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (m_sock < 0) return false;
        struct timeval tv { 1, 0 };
        setsockopt(m_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(m_sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port);
        addr.sin_addr.s_addr = inet_addr(host);
        if (connect(m_sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(m_sock); m_sock = -1;
            return false;
        }
        m_connected = true;
        NS_LOG_UNCOND("[RL] ✓ Reconnecté à " << host << ":" << port);
        return true;
    }

    int RequestAction(const std::vector<double>& state,
                      const std::vector<uint32_t>& neighbors,
                      uint32_t nodeId = 0) {
        if (!m_connected || neighbors.empty()) return 0;

        std::ostringstream os;
        os << "{\"cmd\":\"action\",\"node_id\":" << nodeId << ",\"state\":[";
        for (size_t i = 0; i < state.size(); i++)
            os << (i ? "," : "") << std::fixed << std::setprecision(4) << state[i];
        os << "],\"neighbors\":[";
        for (size_t i = 0; i < neighbors.size(); i++)
            os << (i ? "," : "") << neighbors[i];
        os << "]}\n";

        if (!SendLine(os.str())) return 0;

        std::string resp = RecvLine();
        return ParseInt(resp, "action_index", 0);
    }

    /**
     * Demande le risque PEPM pour un seul nœud.
     * @return risque ∈ [0.0, 1.0], ou -1.0 si échec
     */
    double RequestPEPM(uint32_t nodeId, double energy) {
        if (!m_connected) return -1.0;

        std::ostringstream os;
        os << "{\"cmd\":\"pepm\",\"node_id\":" << nodeId
           << ",\"energy\":" << std::fixed << std::setprecision(6) << energy
           << "}\n";

        if (!SendLine(os.str())) return -1.0;

        const std::string resp = RecvLine();
        if (resp.empty()) return -1.0;

        const double risk = ParseDouble(resp, "risk", -1.0);
        if (risk < 0.0 || risk > 1.0) return -1.0;
        return risk;
    }

    /**
     * Interrogation PEPM groupée pour tous les nœuds vivants en une seule
     * requête TCP (évite N round-trips séquentiels à chaque step RL).
     *
     * @param nodeEnergies  Paires (nodeId, energyJ) pour les nœuds vivants
     * @param[out] risks    Rempli avec {nodeId → risk} pour chaque nœud traité
     * @return              true si la requête a réussi
     */
    bool RequestPEPMBatch(const std::vector<std::pair<uint32_t,double>>& nodeEnergies,
                           std::map<uint32_t,double>& risks) {
        if (!m_connected || nodeEnergies.empty()) return false;

        std::ostringstream os;
        os << "{\"cmd\":\"pepm_batch\",\"nodes\":[";
        for (size_t i = 0; i < nodeEnergies.size(); i++) {
            os << (i ? "," : "")
               << "{\"node_id\":" << nodeEnergies[i].first
               << ",\"energy\":"  << std::fixed << std::setprecision(6)
               << nodeEnergies[i].second << "}";
        }
        os << "]}\n";

        if (!SendLine(os.str())) return false;

        const std::string resp = RecvLine();
        if (resp.empty()) return false;

        // Parser {"risks": {"123": 0.42, ...}, "at_risk": [...]}
        // Format simple : cherche "\"<id>\": <valeur>" dans le bloc "risks"
        const std::string risksKey = "\"risks\":";
        auto rpos = resp.find(risksKey);
        if (rpos == std::string::npos) return false;

        // Extraction manuelle des paires "id": valeur dans le bloc risks
        size_t brace = resp.find('{', rpos + risksKey.size());
        size_t end   = resp.find('}', brace);
        if (brace == std::string::npos || end == std::string::npos) return false;

        std::string block = resp.substr(brace + 1, end - brace - 1);

        // Parcours des paires "id": val
        size_t pos = 0;
        while (pos < block.size()) {
            // Chercher la prochaine clé
            size_t q1 = block.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = block.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            std::string idStr = block.substr(q1 + 1, q2 - q1 - 1);

            // Chercher la valeur après ':'
            size_t colon = block.find(':', q2);
            if (colon == std::string::npos) break;
            size_t vstart = colon + 1;
            while (vstart < block.size() && (block[vstart] == ' ')) vstart++;

            try {
                uint32_t nid  = static_cast<uint32_t>(std::stoul(idStr));
                double   risk = std::stod(block.substr(vstart));
                if (risk >= 0.0 && risk <= 1.0)
                    risks[nid] = risk;
            } catch (...) {}

            pos = block.find(',', vstart);
            if (pos == std::string::npos) break;
            pos++;
        }
        return !risks.empty();
    }

    void SubmitReward(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done,
                      uint32_t nodeId = 0) {
        if (!m_connected) return;

        std::ostringstream os;
        os << "{\"cmd\":\"reward\","
           << "\"node_id\":" << nodeId << ","
           << "\"state\":[";
        for (size_t i = 0; i < state.size(); i++)
            os << (i?",":"") << std::fixed << std::setprecision(4) << state[i];
        os << "],\"action\":" << action
           << ",\"reward\":" << std::setprecision(4) << reward
           << ",\"next_state\":[";
        for (size_t i = 0; i < nextState.size(); i++)
            os << (i?",":"") << std::fixed << std::setprecision(4) << nextState[i];
        os << "],\"done\":" << (done ? 1 : 0) << "}\n";

        if (SendLine(os.str())) {
            std::string resp = RecvLine();
            m_steps++;
            if (m_steps % FdqnCfg::FED_PERIOD == 0) m_fedRound++;
        }
    }

    uint32_t GetSteps()    const { return m_steps; }
    uint32_t GetFedRound() const { return m_fedRound; }

    void SendTopology(const std::vector<ClusterInfo>& clusters) {
        if (!m_connected) return;
        std::ostringstream os;
        os << "{\"cmd\":\"topology\",\"clusters\":[";
        for (size_t ci = 0; ci < clusters.size(); ci++) {
            const auto& c = clusters[ci];
            os << (ci?",":"") << "{\"ch\":" << c.chId << ",\"members\":[";
            for (size_t mi = 0; mi < c.members.size(); mi++)
                os << (mi?",":"") << c.members[mi];
            os << "]}";
        }
        os << "]}\n";
        SendLine(os.str());
        RecvLine();
    }

private:
    bool SendLine(const std::string& s) {
        if (m_sock < 0) return false;
        ssize_t sent = send(m_sock, s.c_str(), s.size(), MSG_NOSIGNAL);
        if (sent <= 0) {
            ::close(m_sock);
            m_sock = -1;
            m_connected = false;
            return false;
        }
        return true;
    }

    std::string RecvLine() {
        if (m_sock < 0) return "";

        // Buffer 16 384 B — suffisant pour la réponse pepm_batch de 300 nœuds
        // (~6 400 chars) avec marge × 2.5.  Pour des simulations > 600 nœuds,
        // augmenter à 32 768.
        static constexpr int RECV_BUF = 16384;
        std::string result;
        result.reserve(RECV_BUF);

        char buf[RECV_BUF];
        // Lecture en boucle jusqu'au '\n' ou fermeture
        while (true) {
            int n = recv(m_sock, buf, sizeof(buf) - 1, 0);
            if (n <= 0) {
                ::close(m_sock);
                m_sock = -1;
                m_connected = false;
                return "";
            }
            result.append(buf, n);
            // Le serveur Python termine chaque réponse par '\n'
            if (result.back() == '\n') break;
            // Si la réponse est très longue, continuer à lire
        }
        return result;
    }

    int ParseInt(const std::string& s, const std::string& key, int def) {
        const std::string k = "\"" + key + "\":";
        auto pos = s.find(k);
        if (pos == std::string::npos) return def;
        pos += k.size();
        while (pos < s.size() && (s[pos]==' ' || s[pos]=='\n')) pos++;
        return std::stoi(s.substr(pos));
    }

    double ParseDouble(const std::string& s, const std::string& key, double def) {
        const std::string k = "\"" + key + "\":";
        auto pos = s.find(k);
        if (pos == std::string::npos) return def;
        pos += k.size();
        while (pos < s.size() && (s[pos]==' ' || s[pos]=='\n')) pos++;
        return std::stod(s.substr(pos));
    }

    int      m_sock;
    bool     m_connected;
    uint32_t m_steps;
    uint32_t m_fedRound;
};

static RLBridge g_rl;

// =============================================================================
// SECTION 5 — AGENT Q-TABLE (fallback)
// =============================================================================

class QAgent {
public:
    QAgent() : m_epsilon(FdqnCfg::EPSILON_MAX) {}

    int SelectAction(uint32_t nodeId, const std::vector<uint32_t>& neighbors) {
        if (neighbors.empty()) return -1;
        if ((double)rand() / RAND_MAX < m_epsilon)
            return rand() % neighbors.size();
        int best = 0;
        double bestQ = -1e18;
        for (size_t i = 0; i < neighbors.size(); i++) {
            double q = m_qtable[{nodeId, neighbors[i]}];
            if (q > bestQ) { bestQ = q; best = i; }
        }
        return best;
    }

    void Update(uint32_t nodeId, uint32_t nextHop, double reward,
                const std::vector<uint32_t>& nextNeighbors) {
        double maxQ = 0.0;
        for (uint32_t nb : nextNeighbors) {
            double q = m_qtable[{nodeId, nb}];
            if (q > maxQ) maxQ = q;
        }
        auto key = std::make_pair(nodeId, nextHop);
        m_qtable[key] = m_qtable[key]
                      + FdqnCfg::LEARNING_RATE * (reward + FdqnCfg::GAMMA * maxQ - m_qtable[key]);
        m_epsilon = std::max(FdqnCfg::EPSILON_MIN,
                             m_epsilon * FdqnCfg::EPSILON_DECAY);
    }

private:
    std::map<std::pair<uint32_t,uint32_t>, double> m_qtable;
    double m_epsilon;
};

// =============================================================================
// SECTION 6 — FONCTIONS UTILITAIRES
// =============================================================================

void InitResultsDirs(const std::string& base) {
    for (const auto& sub : {"topology", "energy", "routing", "rl", "fed"}) {
        fs::create_directories(base + "/" + sub);
    }
}

void ExportTopology(const std::string& path,
                    const std::vector<NodeState>& nodes,
                    const std::string& tag = "") {
    std::ofstream f(path);
    f << "# FDQN-TE+ Topology Export";
    if (!tag.empty()) f << " | " << tag;
    f << "\nNodeId,X,Y,ClusterId,IsClusterHead,Energy,EnergyNorm,"
         "DistToSink,PEPMRisk,IsAlive,TxCount,ReclusterCount,Fitness\n";
    for (const auto& n : nodes) {
        f << std::fixed << std::setprecision(2)
          << n.id       << "," << n.x << "," << n.y << ","
          << n.clusterId << "," << (n.isClusterHead ? 1 : 0) << ","
          << std::setprecision(6) << n.energy << ","
          << std::setprecision(4) << n.NormEnergy() << ","
          << std::setprecision(2) << n.distToSink << ","
          << std::setprecision(4) << n.pepmRisk << ","
          << (n.isAlive ? 1 : 0) << ","
          << n.txCount << "," << n.reclusterCount << ","
          << std::setprecision(4) << n.fitness << "\n";
    }
}

void InitEnergyCSV(std::ofstream& f, const std::string& path) {
    f.open(path);
    // Colonnes alignées avec dashboard — unités SI strictes
    // Énergie=J  Temps=s  Délai=ms  PDR=%  Risque=[0,1]
    f << "# FDQN-TE+ Energy per Round\n"
         "# Units: Energy=J, Time=s, Delay=ms, PDR=%, PEPM_risk=[0,1]\n"
         "Round,"            // Numéro du round
         "Time_s,"           // Temps simulé (s)
         "AliveNodes,"       // Nœuds vivants
         "DeadNodes,"        // Nœuds morts
         "EnergyMean_J,"     // Énergie résiduelle moyenne / nœud vivant (J)
         "EnergyStdDev_J,"   // Écart-type énergie — mesure déséquilibre (J)
         "EnergyMin_J,"      // Énergie min résiduelle (J)
         "EnergyMax_J,"      // Énergie max résiduelle (J)
         "TotalDrained_J,"   // Énergie totale consommée depuis t=0 (J)
         "PDR_RL_pct,"       // PDR logique ADDQN cumulatif (%)
         "PDR_RL_round_pct," // PDR logique ADDQN du round courant (%)
         "PDR_NS3_pct,"      // PDR physique FlowMonitor NS-3 (%)
         "AvgDelay_ms,"      // Délai bout-en-bout moyen FlowMonitor (ms)
         "AtRiskPEPM,"       // Nœuds avec risque PEPM > seuil
         "PEPMRiskMean,"     // Risque PEPM moyen nœuds vivants [0,1]
         "FND_s,"            // First Node Death (s) — 0 si non atteint
         "HND_s,"            // Half Node Death (s) — 0 si non atteint
         "LND_s,"            // 90% Node Death (s) — 0 si non atteint
         "RLSteps,"          // Steps RL cumulés
         "FedRound,"         // Rounds fédérés cumulés
         "IFORound,"         // Rounds IFO exécutés
         "NClusters,"        // Clusters actifs
         "RL_PktEmitted,"    // Paquets émis logique RL
         "RL_PktDelivered,"  // Paquets livrés logique RL
         "TotalEnergy_J\n";  // Énergie totale résiduelle (J)
}

void InitRoutingCSV(std::ofstream& f, const std::string& path) {
    f.open(path);
    f << "# FDQN-TE+ Routing Traces\n"
         "Time_s,SrcId,SrcX,SrcY,ClusterId,IsCH,"
         "NextHop,HopCount,Delivered,Delay_ms,EnergyDrain_J,PEPMRisk\n";
}

// =============================================================================
// SECTION 7 — FONCTIONS DE CALCUL DES MÉTRIQUES
// =============================================================================

double ComputeEnergyConsumed_J(const std::vector<NodeState>& nodes, double initEnergy) {
    double totalConsumed = 0.0;
    for (const auto& node : nodes) {
        totalConsumed += (initEnergy - node.energy);
    }
    return totalConsumed;
}

double ComputeAverageDelay(Ptr<FlowMonitor> fm) {
    double totalDelay = 0.0;
    uint64_t totalRxPackets = 0;

    fm->CheckForLostPackets();
    const auto& stats = fm->GetFlowStats();

    for (const auto& flow : stats) {
        totalRxPackets += flow.second.rxPackets;
        totalDelay += flow.second.delaySum.GetSeconds() * 1000;
    }

    return totalRxPackets > 0 ? totalDelay / totalRxPackets : 0.0;
}

// =============================================================================
// SECTION 8 — CONTEXTE PARTAGÉ
// =============================================================================

struct SimContext {
    uint32_t  nNodes;
    double    simDuration;
    double    initEnergy;
    double    radioRange;
    double    sinkX, sinkY;
    double    areaSize;
    uint32_t  seed = FdqnCfg::DEFAULT_SEED;  // BUG C FIX: seed from CLI, not hardcoded 42

    EnergySourceContainer*       pSrc;
    std::vector<NodeState>*      pSt;
    NodeContainer*               pSens;
    NetDeviceContainer*          pDevs;   // pour désactiver la radio NS-3 à la mort du nœud
    IFOClustering*               pIfo;
    std::map<uint32_t,uint32_t>* pIdIdx;
    std::map<uint32_t,QAgent>*   pQA;
    std::mt19937*                pRng;

    std::ofstream* pEnergyCSV;
    std::ofstream* pRoutingCSV;
    Ptr<FlowMonitor> pFlowMonitor;

    std::string resultsDir;
    uint32_t    topoExportCount  = 0;
    uint64_t    rlPktEmitted     = 0;
    uint64_t    rlPktDelivered   = 0;
    // Compteurs du round précédent — pour PDR DELTA (par round, pas cumulatif)
    uint64_t    rlPktEmittedPrev   = 0;
    uint64_t    rlPktDeliveredPrev = 0;
    std::map<uint32_t, Ptr<Application>> nodeApps;
    double      lastDelayCalcTime = 0.0;
    double      lastPDR_NS3 = 100.0;
};

static SimContext ctx;

// =============================================================================
// SECTION 9 — CALLBACKS
// =============================================================================

std::vector<double> BuildState(const NodeState& n,
                                const std::vector<NodeState>& nodes) {
    const double dMax = ctx.areaSize * std::sqrt(2.0);

    uint32_t nbAlive = 0; double sumENb = 0.0;
    for (const auto& nb : nodes) {
        if (!nb.isAlive || nb.id == n.id) continue;
        if (NodeDist(n.x, n.y, nb.x, nb.y) <= ctx.radioRange) {
            nbAlive++;
            sumENb += nb.NormEnergy();
        }
    }
    const double avgENb = nbAlive > 0 ? sumENb / nbAlive : 0.5;

    double distToCH = 0.0;
    if (!n.isClusterHead) {
        for (const auto& nb : nodes) {
            if (nb.id == n.clusterId) {
                distToCH = NodeDist(n, nb);
                break;
            }
        }
    }

    uint32_t alive = 0;
    for (const auto& nb : nodes) if (nb.isAlive) alive++;
    const double fracAlive = static_cast<double>(alive) / nodes.size();

    return {
        n.NormEnergy(),
        std::min(1.0, n.distToSink / dMax),
        n.pepmRisk,
        std::min(1.0, static_cast<double>(nbAlive) / 20.0),
        n.isClusterHead ? 1.0 : 0.0,
        std::min(1.0, static_cast<double>(n.txCount) / 200.0),
        avgENb,
        fracAlive,
        std::min(1.0, distToCH / dMax),
        std::min(1.0, static_cast<double>(n.reclusterCount) / 20.0)
    };
}

static std::function<void()> rlStep;

static void InitRLStep() {
    rlStep = [&]() {
        const double now = Simulator::Now().GetSeconds();

        // ── Pré-passe PEPM batch ─────────────────────────────────────────────────
        // Toutes les mises à jour PEPM sont groupées en UNE SEULE requête TCP
        // vers Python pepm_lstm.py, au lieu de N requêtes séquentielles.
        // Si Python est indisponible : tentative de reconnexion unique ici,
        // et les pepmRisk des nœuds restent à leur dernière valeur Python reçue.
        // ─────────────────────────────────────────────────────────────────────────
        if (!g_rl.IsConnected())
            g_rl.TryReconnect();

        if (g_rl.IsConnected()) {
            // Construire la liste (nodeId, energy) de tous les nœuds vivants
            std::vector<std::pair<uint32_t,double>> pepmInput;
            pepmInput.reserve(ctx.nNodes);
            for (uint32_t i = 0; i < ctx.nNodes; i++) {
                const NodeState& ns = (*ctx.pSt)[i];
                if (ns.isAlive && !g_deadNodes.count(ns.id))
                    pepmInput.push_back({ns.id, ns.energy});
            }

            // Requête batch → Python met à jour le LSTM pour tous les nœuds
            std::map<uint32_t,double> pepmRisks;
            if (g_rl.RequestPEPMBatch(pepmInput, pepmRisks)) {
                // Appliquer les risques reçus à chaque NodeState
                for (uint32_t i = 0; i < ctx.nNodes; i++) {
                    NodeState& ns = (*ctx.pSt)[i];
                    auto it = pepmRisks.find(ns.id);
                    if (it != pepmRisks.end())
                        ns.pepmRisk = it->second;
                }
            }
            // Si RequestPEPMBatch échoue (socket rompu en cours de requête) :
            // les pepmRisk gardent leur valeur précédente — pas de recalcul C++.
        }
        // ─────────────────────────────────────────────────────────────────────────

        for (uint32_t i = 0; i < ctx.nNodes; i++) {
            NodeState& ns = (*ctx.pSt)[i];
            if (!ns.isAlive || g_deadNodes.count(ns.id)) {
                ns.isAlive = false;
                continue;
            }

            std::vector<uint32_t> neighbors;

            if (ns.isClusterHead) {
                for (uint32_t j = 0; j < ctx.nNodes; j++) {
                    if (i == j) continue;
                    const NodeState& nb = (*ctx.pSt)[j];
                    if (!nb.isAlive || !nb.isClusterHead) continue;
                    if (NodeDist(ns.x, ns.y, nb.x, nb.y) <= ctx.radioRange)
                        neighbors.push_back(nb.id);
                }
                if (neighbors.empty()) {
                    for (uint32_t j = 0; j < ctx.nNodes; j++) {
                        if (i == j) continue;
                        const NodeState& nb = (*ctx.pSt)[j];
                        if (!nb.isAlive) continue;
                        if (NodeDist(ns.x, ns.y, nb.x, nb.y) <= ctx.radioRange)
                            neighbors.push_back(nb.id);
                    }
                }
            } else {
                const uint32_t chId = ns.clusterId;
                const NodeState* chPtr = nullptr;
                for (uint32_t j = 0; j < ctx.nNodes; j++) {
                    if ((*ctx.pSt)[j].id == chId && (*ctx.pSt)[j].isAlive) {
                        chPtr = &(*ctx.pSt)[j]; break;
                    }
                }
                const bool chDirect = chPtr &&
                    NodeDist(ns.x, ns.y, chPtr->x, chPtr->y) <= ctx.radioRange;

                if (chDirect) {
                    neighbors.push_back(chId);
                } else {
                    const double tX = chPtr ? chPtr->x : ctx.sinkX;
                    const double tY = chPtr ? chPtr->y : ctx.sinkY;
                    struct Relay { uint32_t id; double d; };
                    std::vector<Relay> rel;
                    for (uint32_t j = 0; j < ctx.nNodes; j++) {
                        if (i == j) continue;
                        const NodeState& nb = (*ctx.pSt)[j];
                        if (!nb.isAlive) continue;
                        if (NodeDist(ns.x, ns.y, nb.x, nb.y) > ctx.radioRange) continue;
                        rel.push_back({nb.id, NodeDist(nb.x, nb.y, tX, tY)});
                    }
                    std::sort(rel.begin(), rel.end(),
                        [](const Relay& a, const Relay& b){ return a.d < b.d; });
                    for (size_t k = 0; k < std::min((size_t)3, rel.size()); k++)
                        neighbors.push_back(rel[k].id);
                }
            }
            if (neighbors.empty()) continue;

            // pepmRisk déjà mis à jour par la pré-passe batch PEPM en début de step.
            const std::vector<double> state = BuildState(ns, *ctx.pSt);

            int actionIdx;
            uint32_t nextHop;

            if (g_rl.IsConnected()) {
                actionIdx = g_rl.RequestAction(state, neighbors, ns.id);
                actionIdx = std::max(0, std::min(actionIdx,
                            (int)neighbors.size() - 1));
                nextHop   = neighbors[actionIdx];
            } else {
                actionIdx = (*ctx.pQA)[ns.id].SelectAction(ns.id, neighbors);
                if (actionIdx < 0) continue;
                nextHop   = neighbors[actionIdx];
            }

            const NodeState& nhState = (*ctx.pSt)[(*ctx.pIdIdx)[nextHop]];

            // Drain énergétique
            {
                double drain;
                if (ns.isClusterHead) {
                    const uint32_t nMem = static_cast<uint32_t>(
                        [&](){
                            for (const auto& ci : ctx.pIfo->GetClusters())
                                if (ci.chId == ns.id) return ci.members.size();
                            return (size_t)1;
                        }());
                    drain = LeachCHRound(nMem, ns.distToSink, FdqnCfg::DRAIN_BITS);
                } else {
                    const double dTx = NodeDist(ns, nhState);
                    drain = LeachMemberRound(dTx, FdqnCfg::DRAIN_BITS);

                    const double drainRx = FdqnCfg::E_ELEC * FdqnCfg::DRAIN_BITS;
                    NodeState& relay = (*ctx.pSt)[(*ctx.pIdIdx)[nextHop]];
                    if (relay.isAlive && relay.id != ns.clusterId) {
                        if (!relay.Consume(drainRx) && !g_deadNodes.count(relay.id)) {
                            g_deadNodes.insert(relay.id);
                            auto itR = ctx.nodeApps.find(relay.id);
                            if (itR != ctx.nodeApps.end() && itR->second)
                                itR->second->SetStopTime(Seconds(now));
                            if (!g_fndDone) {
                                g_fndDone = true; g_fndTime = now;
                                g_compMetrics.fndTime = now;
                                NS_LOG_UNCOND("⭐ [FND] Premier nœud mort à t=" << now << "s");
                            }
                            if (!g_hndDone && g_deadNodes.size() >= g_nNodes / 2) {
                                g_hndDone = true; g_hndTime = now;
                                g_compMetrics.hndTime = now;
                                NS_LOG_UNCOND("⭐ [HND] Moitié nœuds morts à t=" << now << "s");
                            }
                            if (!g_lndDone && g_deadNodes.size() >= (uint32_t)(g_nNodes * 0.9)) {
                                g_lndDone = true; g_lndTime = now;
                                g_compMetrics.lndTime = now;
                                NS_LOG_UNCOND("⭐ [LND-90%] 90% nœuds morts à t=" << now << "s");
                                Simulator::Stop();
                            }
                        }
                    }
                }

                const bool stillAlive = ns.Consume(drain);

                if (!stillAlive && !g_deadNodes.count(ns.id)) {
                    g_deadNodes.insert(ns.id);

                    // 1. Arrêter l'application UDP (plus d'émission)
                    auto itApp = ctx.nodeApps.find(ns.id);
                    if (itApp != ctx.nodeApps.end() && itApp->second) {
                        itApp->second->SetStopTime(Seconds(now));
                    }

                    // 2. Désactiver la radio NS-3 physiquement — CORRECTION PDR NS-3
                    // Sans ça, la couche WiFi reste active, OLSR continue à router
                    // autour du nœud mort et le FlowMonitor affiche 100% en permanence.
                    for (uint32_t devIdx = 0; devIdx < ctx.pDevs->GetN(); devIdx++) {
                        Ptr<NetDevice> dev = ctx.pDevs->Get(devIdx);
                        if (dev->GetNode()->GetId() == ns.id) {
                            // Mettre le nœud hors ligne au niveau IP (retire les routes OLSR)
                            Ptr<Ipv4> ipv4 = dev->GetNode()->GetObject<Ipv4>();
                            if (ipv4) {
                                int32_t ifIndex = ipv4->GetInterfaceForDevice(dev);
                                if (ifIndex >= 0) ipv4->SetDown(ifIndex);
                            }
                            break;
                        }
                    }
                    if (!g_fndDone) {
                        g_fndDone = true; g_fndTime = now;
                        g_compMetrics.fndTime = now;
                        NS_LOG_UNCOND("⭐ [FND] Premier nœud mort à t=" << now << "s");
                    }
                    // Snapshot PDR pré-FND : on gèle les compteurs au moment exact du FND
                    if (!g_compMetrics.pdrRL_preFND_locked) {
                        g_compMetrics.pdrRL_preFND_emitted   = ctx.rlPktEmitted;
                        g_compMetrics.pdrRL_preFND_delivered = ctx.rlPktDelivered;
                        g_compMetrics.pdrRL_preFND_locked    = true;
                    }
                    if (!g_hndDone && g_deadNodes.size() >= g_nNodes / 2) {
                        g_hndDone = true; g_hndTime = now;
                        g_compMetrics.hndTime = now;
                        NS_LOG_UNCOND("⭐ [HND] Moitié nœuds morts à t=" << now << "s");
                    }
                    if (!g_lndDone && g_deadNodes.size() >= (uint32_t)(g_nNodes * 0.9)) {
                        g_lndDone = true; g_lndTime = now;
                        g_compMetrics.lndTime = now;
                        NS_LOG_UNCOND("⭐ [LND-90%] 90% nœuds morts à t=" << now << "s");
                        Simulator::Stop();
                    }
                }
            }

            // Récompense
            {
                const double dMax  = ctx.radioRange * 10.0;
                const double eNorm = ns.NormEnergy();

                double hierBonus = 0.0;
                if (!ns.isClusterHead) {
                    const uint32_t chId2 = ns.clusterId;
                    const NodeState* chPtr2 = nullptr;
                    for (uint32_t j = 0; j < ctx.nNodes; j++) {
                        if ((*ctx.pSt)[j].id == chId2) {
                            chPtr2 = &(*ctx.pSt)[j]; break;
                        }
                    }
                    const bool chDir = chPtr2 && chPtr2->isAlive &&
                        NodeDist(ns.x, ns.y, chPtr2->x, chPtr2->y) <= ctx.radioRange;
                    if (chDir) {
                        hierBonus = (nextHop == chId2) ? 1.0 : -0.5;
                    } else {
                        const double tX2 = (chPtr2 && chPtr2->isAlive) ? chPtr2->x : ctx.sinkX;
                        const double tY2 = (chPtr2 && chPtr2->isAlive) ? chPtr2->y : ctx.sinkY;
                        const double dS2 = NodeDist(ns.x, ns.y, tX2, tY2);
                        const double dN2 = NodeDist(nhState.x, nhState.y, tX2, tY2);
                        hierBonus = (dS2 > 1.0)
                            ? std::max(-1.0, std::min(1.0, (dS2 - dN2) / dS2))
                            : 1.0;
                    }
                } else {
                    const double dS = ns.distToSink, dN = nhState.distToSink;
                    hierBonus = dS > 1.0
                        ? std::max(-1.0, std::min(1.0, (dS - dN) / dS))
                        : 1.0;
                }

                //  PDR signal réel (±1) — agent pénalise les paquets perdus
                const double pdrSignal = nhState.isAlive ? 1.0 : -1.0;
                const double reward
                    = FdqnCfg::LAMBDA_PDR    * pdrSignal
                    + FdqnCfg::LAMBDA_ENERGY * eNorm
                    - FdqnCfg::LAMBDA_DELAY  * std::min(1.0, nhState.distToSink / dMax)
                    + FdqnCfg::LAMBDA_SAFE   * (1.0 - ns.pepmRisk)
                    + FdqnCfg::LAMBDA_HIER   * hierBonus;

                const std::vector<double> nextState = BuildState(ns, *ctx.pSt);
                const bool done = !ns.isAlive;

                if (g_rl.IsConnected())
                    g_rl.SubmitReward(state, actionIdx, reward, nextState, done, ns.id);
                else
                    (*ctx.pQA)[ns.id].Update(ns.id, nextHop, reward, neighbors);

                ns.totalReward += reward;
            }

            // PDR logique RL
            ctx.rlPktEmitted++;
            {
                bool delivered = false;

                if (ns.isAlive && nhState.isAlive) {
                    if (ns.isClusterHead) {
                        // CH→sink : livré si nextHop CH se rapproche du sink
                        delivered = (nhState.distToSink < ns.distToSink)
                                 || nhState.isClusterHead;
                    } else {
                        // Membre→CH : livré dès que nextHop est vivant et est le CH
                        // ou se rapproche du CH (relay intermédiaire)
                        const uint32_t myChId = ns.clusterId;
                        if (nextHop == myChId) {
                            delivered = true;  // livraison directe au CH
                        } else {
                            // relay : livré si le CH est vivant (le relay achemine vers CH)
                            bool chAlive = false;
                            for (uint32_t j = 0; j < ctx.nNodes; j++) {
                                if ((*ctx.pSt)[j].id == myChId && (*ctx.pSt)[j].isAlive) {
                                    chAlive = true; break;
                                }
                            }
                            delivered = chAlive;  // relay vers un CH vivant = livré
                        }
                    }
                }
                if (delivered) ctx.rlPktDelivered++;
            }

            // Trace routage
            if (ctx.pRoutingCSV && ctx.pRoutingCSV->is_open()) {
                const double drainTx = ns.isClusterHead
                    ? LeachCHRound(1, ns.distToSink, FdqnCfg::DRAIN_BITS)
                    : LeachMemberRound(NodeDist(ns, nhState), FdqnCfg::DRAIN_BITS);
                const int delivered = nhState.isAlive ? 1 : 0;
                *ctx.pRoutingCSV << std::fixed << std::setprecision(2)
                    << now << "," << ns.id << "," << ns.x << "," << ns.y << ","
                    << ns.clusterId << "," << (ns.isClusterHead?1:0) << ","
                    << nextHop << ",1," << delivered << ",0.0,"
                    << std::setprecision(6) << drainTx << ","
                    << std::setprecision(4) << ns.pepmRisk << "\n";
            }

            ns.txCount++;
        }

        if (now + FdqnCfg::RL_STEP_INTERVAL <= ctx.simDuration)
            Simulator::Schedule(Seconds(FdqnCfg::RL_STEP_INTERVAL), rlStep);
    };
}

static std::function<void(int)> doCheck;

static void InitDoCheck() {
    doCheck = [&](int round) {
        const double now = Simulator::Now().GetSeconds();

        for (auto& ns : *ctx.pSt)
            if (g_deadNodes.count(ns.id)) ns.isAlive = false;

        // ===== MÉTRIQUES ORIGINALES =====
        uint32_t alive = 0, atRisk = 0;
        double sumE = 0.0, sumE2 = 0.0;
        double minE = 1e18, maxE = 0.0;
        double drainedTotal = 0.0;
        double sumPepmRisk = 0.0;   // Pour PEPMRiskMean

        for (const auto& ns : *ctx.pSt) {
            if (ns.isAlive) {
                alive++;
                sumE  += ns.energy;
                sumE2 += ns.energy * ns.energy;
                minE   = std::min(minE, ns.energy);
                maxE   = std::max(maxE, ns.energy);
                sumPepmRisk += ns.pepmRisk;
                //  atRisk uniquement pour nœuds VIVANTS
                if (ns.pepmRisk > FdqnCfg::PEPM_RISK_THRESHOLD) atRisk++;
            }
            drainedTotal += (ctx.initEnergy - ns.energy);
        }

        const uint32_t dead = static_cast<uint32_t>(g_deadNodes.size());
        const double meanE = alive > 0 ? sumE / alive : 0.0;
        const double stdE = alive > 0 ? std::sqrt(std::max(0.0, sumE2/alive - meanE*meanE)) : 0.0;
        const double pepmRiskMean = alive > 0 ? sumPepmRisk / alive : 0.0;

        // PDR RL cumulatif (depuis t=0) — référence globale
        double pdrRLNow = 100.0;
        if (ctx.rlPktEmitted > 0)
            pdrRLNow = 100.0 * (double)ctx.rlPktDelivered / ctx.rlPktEmitted;

        // PDR RL du round courant (delta depuis le round précédent)
        const uint64_t deltaEmitted   = ctx.rlPktEmitted   - ctx.rlPktEmittedPrev;
        const uint64_t deltaDelivered = ctx.rlPktDelivered - ctx.rlPktDeliveredPrev;
        const double pdrRLRound = (deltaEmitted > 0)
            ? 100.0 * (double)deltaDelivered / deltaEmitted
            : pdrRLNow;  // fallback cumulatif si aucun paquet ce round
        ctx.rlPktEmittedPrev   = ctx.rlPktEmitted;
        ctx.rlPktDeliveredPrev = ctx.rlPktDelivered;

        // PDR NS-3 via FlowMonitor
        double pdrNS3Now = ctx.lastPDR_NS3;
        if (ctx.pFlowMonitor && (round % 5 == 0 || round == 1)) {
            uint64_t totalTx = 0, totalRx = 0;
            ctx.pFlowMonitor->CheckForLostPackets();
            for (const auto& kv : ctx.pFlowMonitor->GetFlowStats()) {
                totalTx += kv.second.txPackets;
                totalRx += kv.second.rxPackets;
            }
            pdrNS3Now = totalTx > 0 ? 100.0 * (double)totalRx / totalTx : 0.0;
            ctx.lastPDR_NS3 = pdrNS3Now;
        }

        // ===== MÉTRIQUES POUR COMPARAISON =====
        double energyConsumed_J = ComputeEnergyConsumed_J(*ctx.pSt, ctx.initEnergy);
        g_compMetrics.energyConsumption_J.push_back(energyConsumed_J);
        g_compMetrics.totalEnergyDrained_J.push_back(drainedTotal);
        g_compMetrics.aliveNodesPerRound.push_back(alive);
        g_compMetrics.deadNodesPerRound.push_back(dead);
        g_compMetrics.atRiskPEPMPerRound.push_back(atRisk);
        g_compMetrics.pdrRLPerRound.push_back(pdrRLNow);
        g_compMetrics.pdrNS3PerRound.push_back(pdrNS3Now);
        g_compMetrics.rlStepsPerRound.push_back(g_rl.GetSteps());
        g_compMetrics.fedRoundsPerRound.push_back(g_rl.GetFedRound());

        // Calcul du délai
        if (ctx.pFlowMonitor && (round % 10 == 0 || round == 1)) {
            double delay = ComputeAverageDelay(ctx.pFlowMonitor);
            g_compMetrics.endToEndDelay_ms.push_back(delay);
            g_compMetrics.avgDelay_ms = (g_compMetrics.avgDelay_ms * (g_compMetrics.endToEndDelay_ms.size() - 1) + delay)
                                        / g_compMetrics.endToEndDelay_ms.size();
            ctx.lastDelayCalcTime = now;
        }

        // Mise à jour des métriques de lifetime
        if (!g_fndDone && dead >= 1) {
            g_fndDone = true;
            g_fndTime = now;
            g_compMetrics.fndTime = now;
        }
        if (!g_hndDone && dead >= ctx.nNodes / 2) {
            g_hndDone = true;
            g_hndTime = now;
            g_compMetrics.hndTime = now;
        }
        if (!g_lndDone && dead >= (uint32_t)(ctx.nNodes * 0.9)) {
            g_lndDone = true;
            g_lndTime = now;
            g_compMetrics.lndTime = now;
        }
        g_compMetrics.totalRounds = round;

        // ===== COLLECTE POUR HISTORIQUE RL =====
        RLHistoryEntry entry;
        entry.timestamp = now;
        entry.round = round;
        entry.aliveNodes = alive;
        entry.deadNodes = dead;
        entry.avgEnergy_J = meanE;
        entry.totalEnergyConsumed_J = energyConsumed_J;
        entry.pdr_RL_pct = pdrRLNow;
        entry.pdr_NS3_pct = pdrNS3Now;
        entry.avgDelay_ms = g_compMetrics.avgDelay_ms;
        entry.rlSteps = g_rl.GetSteps();
        entry.fedRound = g_rl.GetFedRound();
        entry.nClusters = ctx.pIfo->GetClusters().size();
        entry.atRiskPEPM = atRisk;

        // Collecte des récompenses moyennes
        double sumReward = 0.0;
        double minReward = 1e18, maxReward = -1e18;
        for (const auto& ns : *ctx.pSt) {
            if (ns.isAlive) {
                sumReward += ns.totalReward;
                minReward = std::min(minReward, ns.totalReward);
                maxReward = std::max(maxReward, ns.totalReward);
            }
        }
        double meanReward = alive > 0 ? sumReward / alive : 0.0;
        entry.rewardDistribution = {minReward, maxReward, meanReward};

        // Q-values approximatives (moyenne fixe 0.5 tant que rl_server gère les poids)
        // Iteration directe supprimée → utiliser la taille du pool
        const int qCount = static_cast<int>(ctx.pQA->size());
        const double sumQ = 0.5 * qCount;   // placeholder — Q réels dans rl_server.py
        entry.qValueDistribution = {0.0, 1.0, qCount > 0 ? sumQ / qCount : 0.5};

        g_rlHistory.push_back(entry);

        // Lambda de mise à jour distToSink après tout recluster
        // distToSink est utilisé par la fitness IFO, la récompense ADDQN et
        // le drain énergétique LeachCHRound. Il doit refléter la position
        // réelle du sink, surtout si areaSize ou sinkX/Y varient via CLI.
        auto updateDistToSink = [&]() {
            for (auto& ns : *ctx.pSt) {
                if (ns.isAlive)
                    ns.distToSink = NodeDist(ns.x, ns.y, ctx.sinkX, ctx.sinkY);
            }
        };

        // Re-clustering proactif PEPM — déclenché à chaque round si CH à risque
        if (alive > 0 && atRisk > 0) {
            uint32_t rotations = ctx.pIfo->TriggerProactiveRecluster(*ctx.pSt);
            if (rotations > 0) {
                updateDistToSink();
                if (g_rl.IsConnected())
                    g_rl.SendTopology(ctx.pIfo->GetClusters());
            }
        }

        // Re-clustering IFO périodique
        if (round % 2 == 0 && alive > 0) {
            const uint32_t newNC = ctx.pIfo->ComputeNClusters(*ctx.pSt);
            ctx.pIfo->Run(*ctx.pSt, newNC);
            updateDistToSink();

            if (g_rl.IsConnected())
                g_rl.SendTopology(ctx.pIfo->GetClusters());

            const auto stats = ctx.pIfo->GetStats();
            NS_LOG_UNCOND("[IFO] Recluster round=" << round
                        << " → " << stats.nClusters << " clusters"
                        << " membres [" << stats.membersMin
                        << "-" << stats.membersMax
                        << "] moy=" << std::fixed << std::setprecision(1)
                        << stats.membersMean);

            std::ostringstream tPath;
            tPath << ctx.resultsDir << "/topology/fdqnte_topology_r"
                  << std::setfill('0') << std::setw(4) << round << ".csv";
            ExportTopology(tPath.str(), *ctx.pSt, "round=" + std::to_string(round));
            ctx.topoExportCount++;
        }

        // ===== ÉCRITURE CSV ÉNERGIE (25 colonnes — alignées avec entête) =====
        if (ctx.pEnergyCSV && ctx.pEnergyCSV->is_open()) {
            const double curDelay = g_compMetrics.endToEndDelay_ms.empty()
                                  ? 0.0
                                  : g_compMetrics.endToEndDelay_ms.back();
            const uint32_t nClusters = static_cast<uint32_t>(
                ctx.pIfo->GetClusters().size());

            *ctx.pEnergyCSV << std::fixed
                << round << ","                                               // Round
                << std::setprecision(1) << now << ","                        // Time_s
                << alive << "," << dead << ","                               // AliveNodes, DeadNodes
                << std::setprecision(6) << meanE << ","                      // EnergyMean_J
                << stdE << ","                                                // EnergyStdDev_J
                << minE << ","                                                // EnergyMin_J
                << (alive>0 ? maxE : 0.0) << ","                            // EnergyMax_J
                << std::setprecision(4) << drainedTotal << ","               // TotalDrained_J
                << std::setprecision(2) << pdrRLNow << ","                   // PDR_RL_pct (cumulatif)
                << std::setprecision(2) << pdrRLRound << ","                  // PDR_RL_round_pct (delta)
                << pdrNS3Now << ","                                           // PDR_NS3_pct
                << std::setprecision(2) << curDelay << ","                   // AvgDelay_ms
                << atRisk << ","                                              // AtRiskPEPM
                << std::setprecision(4) << pepmRiskMean << ","               // PEPMRiskMean
                << std::setprecision(1)
                << (g_fndDone ? g_fndTime : 0.0) << ","                     // FND_s
                << (g_hndDone ? g_hndTime : 0.0) << ","                     // HND_s
                << (g_lndDone ? g_lndTime : 0.0) << ","                     // LND_s
                << g_rl.GetSteps() << ","                                    // RLSteps
                << g_rl.GetFedRound() << ","                                 // FedRound
                << ctx.pIfo->GetRound() << ","                               // IFORound
                << nClusters << ","                                           // NClusters
                << ctx.rlPktEmitted << ","                                    // RL_PktEmitted
                << ctx.rlPktDelivered << ","                                  // RL_PktDelivered
                << std::setprecision(4) << sumE << "\n";                     // TotalEnergy_J
            ctx.pEnergyCSV->flush();
        }

        // ===== AFFICHAGE CONSOLE =====
        NS_LOG_UNCOND(std::fixed
            << "[Round " << round << "] t="
            << std::setprecision(0) << now << "s"
            << "|vivants=" << alive
            << "|morts=" << dead
            << "|E_moy=" << std::setprecision(3) << meanE << "J"
            << "|E_cons=" << std::setprecision(3) << drainedTotal << "J"
            << "|PDR_RL=" << std::setprecision(1) << pdrRLNow << "%"
            << "|PDR_NS3=" << pdrNS3Now << "%"
            << "|delay=" << std::setprecision(2)
            << (g_compMetrics.endToEndDelay_ms.empty() ? 0.0 : g_compMetrics.endToEndDelay_ms.back()) << "ms"
            << "|PEPM@risk=" << atRisk
            << "|PEPM_moy=" << std::setprecision(3) << pepmRiskMean
            << "|RL=" << g_rl.GetSteps()
            << "|fed=" << g_rl.GetFedRound());

        if (now + FdqnCfg::METRICS_INTERVAL <= ctx.simDuration) {
            int nr = round + 1;
            Simulator::Schedule(Seconds(FdqnCfg::METRICS_INTERVAL),
                                [nr]{ doCheck(nr); });
        }
    };
}

// =============================================================================
// SECTION 10 — EXPORT DE L'HISTORIQUE RL
// =============================================================================

void ExportRLHistory() {
    std::string rlHistoryPath = ctx.resultsDir + "/rl/fdqnte_rl_history.json";
    std::ofstream json(rlHistoryPath);

    json << "{\n";
    json << "  \"simulation_info\": {\n";
    json << "    \"nNodes\": " << ctx.nNodes << ",\n";
    json << "    \"initEnergy_J\": " << ctx.initEnergy << ",\n";
    json << "    \"simDuration_s\": " << ctx.simDuration << ",\n";
    json << "    \"areaSize_m\": " << ctx.areaSize << ",\n";
    json << "    \"radioRange_m\": " << ctx.radioRange << ",\n";
    json << "    \"seed\": " << ctx.seed << "\n";
    json << "  },\n";
    json << "  \"metrics\": {\n";
    json << "    \"fnd_time_s\": " << g_compMetrics.fndTime << ",\n";
    json << "    \"hnd_time_s\": " << g_compMetrics.hndTime << ",\n";
    json << "    \"lnd_time_s\": " << g_compMetrics.lndTime << ",\n";
    json << "    \"total_rounds\": " << g_compMetrics.totalRounds << ",\n";
    json << "    \"avg_pdr_RL_pct\": " << g_compMetrics.avgPDR_RL << ",\n";
    json << "    \"avg_pdr_NS3_pct\": " << g_compMetrics.avgPDR_NS3 << ",\n";
    json << "    \"avg_delay_ms\": " << g_compMetrics.avgDelay_ms << ",\n";
    json << "    \"total_energy_consumed_J\": " << g_compMetrics.totalEnergyConsumed_J << "\n";
    json << "  },\n";
    json << "  \"history\": [\n";

    for (size_t i = 0; i < g_rlHistory.size(); i++) {
        const auto& h = g_rlHistory[i];
        json << "    {\n";
        json << "      \"round\": " << h.round << ",\n";
        json << "      \"timestamp_s\": " << std::fixed << std::setprecision(2) << h.timestamp << ",\n";
        json << "      \"alive_nodes\": " << h.aliveNodes << ",\n";
        json << "      \"dead_nodes\": " << h.deadNodes << ",\n";
        json << "      \"avg_energy_J\": " << std::fixed << std::setprecision(6) << h.avgEnergy_J << ",\n";
        json << "      \"total_energy_consumed_J\": " << std::fixed << std::setprecision(6) << h.totalEnergyConsumed_J << ",\n";
        json << "      \"pdr_RL_pct\": " << std::fixed << std::setprecision(2) << h.pdr_RL_pct << ",\n";
        json << "      \"pdr_NS3_pct\": " << std::fixed << std::setprecision(2) << h.pdr_NS3_pct << ",\n";
        json << "      \"avg_delay_ms\": " << std::fixed << std::setprecision(2) << h.avgDelay_ms << ",\n";
        json << "      \"rl_steps\": " << h.rlSteps << ",\n";
        json << "      \"fed_round\": " << h.fedRound << ",\n";
        json << "      \"n_clusters\": " << h.nClusters << ",\n";
        json << "      \"at_risk_pepm\": " << h.atRiskPEPM << ",\n";
        json << "      \"rewards\": {\n";
        json << "        \"min\": " << h.rewardDistribution[0] << ",\n";
        json << "        \"max\": " << h.rewardDistribution[1] << ",\n";
        json << "        \"mean\": " << h.rewardDistribution[2] << "\n";
        json << "      },\n";
        json << "      \"q_values\": {\n";
        json << "        \"min\": " << h.qValueDistribution[0] << ",\n";
        json << "        \"max\": " << h.qValueDistribution[1] << ",\n";
        json << "        \"mean\": " << h.qValueDistribution[2] << "\n";
        json << "      }\n";
        json << "    }";
        if (i < g_rlHistory.size() - 1) json << ",";
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    json.close();
    NS_LOG_UNCOND("[EXPORT] Historique RL sauvegardé dans " << rlHistoryPath);
}

// =============================================================================
// SECTION 11 — EXPORT DES MÉTRIQUES DE COMPARAISON
// =============================================================================

void ExportComparisonMetrics() {
    std::string compPath = ctx.resultsDir + "/comparison_metrics.csv";
    std::ofstream comp(compPath);

    comp << "# FDQN-TE+ Comparison Metrics (vs Article) - Énergie en Joules\n"
         << "# Tables: 3 (Energy), 4 (Lifetime), 5 (PDR), 6 (Delay), 7-8 (Alive/Dead Nodes)\n\n";

    // SUMMARY
    comp << "[SUMMARY]\n";
    comp << "Metric,Value,Unit\n";
    comp << "FND (First Node Death)," << g_compMetrics.fndTime << ",s\n";
    comp << "HND (Half Node Death)," << g_compMetrics.hndTime << ",s\n";
    comp << "LND (90% Node Death)," << g_compMetrics.lndTime << ",s\n";
    comp << "Total Rounds," << g_compMetrics.totalRounds << ",rounds\n";
    comp << "Average PDR RL," << g_compMetrics.avgPDR_RL << ",%\n";
    comp << "PDR RL pre-FND (stable phase)," << g_compMetrics.avgPDR_RL_preFND << ",%\n";
    comp << "Average PDR NS-3," << g_compMetrics.avgPDR_NS3 << ",%\n";
    comp << "Average End-to-End Delay," << g_compMetrics.avgDelay_ms << ",ms\n";
    comp << "Total Energy Consumed," << g_compMetrics.totalEnergyConsumed_J << ",J\n\n";

    // TABLE 3: ENERGY CONSUMPTION
    comp << "[TABLE_3_ENERGY_CONSUMPTION_J]\n";
    comp << "Round,EnergyConsumed_J\n";
    for (size_t i = 0; i < g_compMetrics.energyConsumption_J.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << std::fixed << std::setprecision(6) << g_compMetrics.energyConsumption_J[i] << "\n";
    }
    comp << "\n";

    // TABLE 4: NETWORK LIFETIME
    comp << "[TABLE_4_NETWORK_LIFETIME]\n";
    comp << "Round,AliveNodes,DeadNodes,Time_s\n";
    for (size_t i = 0; i < g_compMetrics.aliveNodesPerRound.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << g_compMetrics.aliveNodesPerRound[i] << ","
             << g_compMetrics.deadNodesPerRound[i] << ","
             << (i+1)*FdqnCfg::METRICS_INTERVAL << "\n";
    }
    comp << "\n";

    // TABLE 5: PDR
    comp << "[TABLE_5_PDR_PERCENT]\n";
    comp << "Round,PDR_RL_pct,PDR_NS3_pct\n";
    for (size_t i = 0; i < g_compMetrics.pdrRLPerRound.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << std::fixed << std::setprecision(2) << g_compMetrics.pdrRLPerRound[i] << ","
             << std::fixed << std::setprecision(2) << g_compMetrics.pdrNS3PerRound[i] << "\n";
    }
    comp << "\n";

    // TABLE 6: DELAY
    comp << "[TABLE_6_END_TO_END_DELAY_ms]\n";
    comp << "Round,Delay_ms\n";
    for (size_t i = 0; i < g_compMetrics.endToEndDelay_ms.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL*10 << ","
             << std::fixed << std::setprecision(2) << g_compMetrics.endToEndDelay_ms[i] << "\n";
    }
    comp << "\n";

    // TABLE 7-8: ALIVE/DEAD NODES
    comp << "[TABLE_7_ALIVE_NODES]\n";
    comp << "Round,AliveNodes\n";
    for (size_t i = 0; i < g_compMetrics.aliveNodesPerRound.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << g_compMetrics.aliveNodesPerRound[i] << "\n";
    }
    comp << "\n";

    comp << "[TABLE_8_DEAD_NODES]\n";
    comp << "Round,DeadNodes\n";
    for (size_t i = 0; i < g_compMetrics.deadNodesPerRound.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << g_compMetrics.deadNodesPerRound[i] << "\n";
    }
    comp << "\n";

    // TABLE RL METRICS
    comp << "[TABLE_RL_METRICS]\n";
    comp << "Round,RL_Steps,Fed_Rounds\n";
    for (size_t i = 0; i < g_compMetrics.rlStepsPerRound.size(); i++) {
        comp << (i+1)*FdqnCfg::METRICS_INTERVAL << ","
             << g_compMetrics.rlStepsPerRound[i] << ","
             << g_compMetrics.fedRoundsPerRound[i] << "\n";
    }

    comp.close();

    NS_LOG_UNCOND("\n [EXPORT] Fichier comparison_metrics.csv sauvegardé dans " << compPath);
}

// =============================================================================
// SECTION 12 — MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    std::signal(SIGPIPE, SIG_IGN);

    auto simStartTime = std::chrono::high_resolution_clock::now();
    g_compMetrics.simulationStartTime = Simulator::Now().GetSeconds();

    SimParams p;
    CommandLine cmd;
    cmd.AddValue("nNodes",      "Nombre de nœuds capteurs",     p.nNodes);
    cmd.AddValue("seed",        "Graine aléatoire",              p.seed);
    cmd.AddValue("simDuration", "Durée simulation (s)",          p.simDuration);
    cmd.AddValue("radioRange",  "Portée radio (m)",              p.radioRange);
    cmd.AddValue("initEnergy",  "Énergie initiale (J)",          p.initEnergy);
    cmd.AddValue("nClusters",   "Clusters cibles IFO (0=auto)",  p.nClusters);
    cmd.AddValue("resultsDir",  "Dossier résultats",             p.resultsDir);
    cmd.AddValue("areaSize",    "Taille de la zone (m)",         p.areaSize);
    cmd.Parse(argc, argv);

    // Ajustement de l'aire
    if (p.areaSize == FdqnCfg::AREA_SIZE && p.nNodes != FdqnCfg::N_NODES) {
        p.areaSize = FdqnCfg::AREA_SIZE
                   * std::sqrt(static_cast<double>(p.nNodes)
                              / FdqnCfg::N_NODES);
        p.areaSize = std::max(200.0, std::min(p.areaSize, 2000.0));
    }
    p.sinkX = p.areaSize / 2.0;
    p.sinkY = p.areaSize / 2.0;

    RngSeedManager::SetSeed(p.seed);
    RngSeedManager::SetRun(1);
    std::mt19937 rng(p.seed);

    InitResultsDirs(p.resultsDir);

    NS_LOG_UNCOND("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    NS_LOG_UNCOND("║                                                     FDQN-TE+ — Démarrage                                                             ║");
    NS_LOG_UNCOND("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    NS_LOG_UNCOND("  Nœuds    : " << p.nNodes
                  << "   Énergie: " << p.initEnergy << " J"
                  << "   Durée  : " << p.simDuration << " s"
                  << "   Portée : " << p.radioRange << " m"
                  << "   Zone   : " << p.areaSize << " m"
                  << "   Seed   : " << p.seed
                  << "   Seuil PEPM : " << FdqnCfg::PEPM_RISK_THRESHOLD);
    NS_LOG_UNCOND("────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────");

    g_nNodes = p.nNodes;
    g_rl.Connect();

    // Nœuds NS-3
    NodeContainer sensors, sinkNode;
    sensors.Create(p.nNodes);
    sinkNode.Create(1);

    // Mobilité
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max="
                         + std::to_string(p.areaSize) + "]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max="
                         + std::to_string(p.areaSize) + "]"));
    mob.Install(sensors);

    mob.SetPositionAllocator("ns3::GridPositionAllocator",
        "MinX", DoubleValue(p.sinkX),
        "MinY", DoubleValue(p.sinkY));
    mob.Install(sinkNode);

    // WiFi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode",    StringValue("DsssRate1Mbps"),
                                  "ControlMode", StringValue("DsssRate1Mbps"));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    YansWifiPhyHelper phy;
    YansWifiChannelHelper wch = YansWifiChannelHelper::Default();
    wch.AddPropagationLoss("ns3::RangePropagationLossModel",
                            "MaxRange", DoubleValue(p.radioRange));
    phy.SetChannel(wch.Create());

    NodeContainer all; all.Add(sensors); all.Add(sinkNode);
    NetDeviceContainer devs = wifi.Install(phy, mac, all);

    // Réseau
    OlsrHelper olsr;
    InternetStackHelper inet;
    inet.SetRoutingHelper(olsr);
    inet.Install(all);

    Ipv4AddressHelper ip;
    ip.SetBase("10.0.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifaces = ip.Assign(devs);

    // Énergie NS-3
    BasicEnergySourceHelper eHelper;
    eHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(p.initEnergy));
    eHelper.Set("BasicEnergySupplyVoltageV",        DoubleValue(FdqnCfg::SUPPLY_VOLTAGE));

    WifiRadioEnergyModelHelper reHelper;
    reHelper.Set("TxCurrentA",    DoubleValue(FdqnCfg::TX_CURRENT));
    reHelper.Set("RxCurrentA",    DoubleValue(FdqnCfg::RX_CURRENT));
    reHelper.Set("IdleCurrentA",  DoubleValue(FdqnCfg::IDLE_CURRENT));
    reHelper.Set("SleepCurrentA", DoubleValue(FdqnCfg::SLEEP_CURRENT));

    EnergySourceContainer eSources = eHelper.Install(sensors);

    for (uint32_t i = 0; i < p.nNodes; i++) {
        reHelper.Install(devs.Get(i), eSources.Get(i));
    }

    // Application UDP
    const uint16_t port = 9;
    InetSocketAddress sinkAddr(ifaces.GetAddress(p.nNodes), port);

    PacketSinkHelper sinkApp("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sApps = sinkApp.Install(sinkNode.Get(0));
    sApps.Start(Seconds(0.0));
    sApps.Stop(Seconds(p.simDuration));

    for (uint32_t i = 0; i < p.nNodes; i++) {
        OnOffHelper s("ns3::UdpSocketFactory", sinkAddr);
        s.SetAttribute("PacketSize", UintegerValue(FdqnCfg::PKT_BITS / 8));
        s.SetAttribute("DataRate", StringValue("10kbps"));
        s.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=2]"));
        s.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
        ApplicationContainer a = s.Install(sensors.Get(i));
        a.Start(Seconds(15.0 + i * 0.01));
        a.Stop(Seconds(p.simDuration));
        const uint32_t nid_app = sensors.Get(i)->GetId();
        if (a.GetN() > 0)
            ctx.nodeApps[nid_app] = a.Get(0);
    }

    // Initialisation des NodeStates
    std::vector<NodeState> nodeStates(p.nNodes);
    std::map<uint32_t, uint32_t> idToIdx;

    for (uint32_t i = 0; i < p.nNodes; i++) {
        Ptr<MobilityModel> m = sensors.Get(i)->GetObject<MobilityModel>();
        const Vector v       = m->GetPosition();
        const uint32_t nid   = sensors.Get(i)->GetId();
        const double dSink   = NodeDist(v.x, v.y, p.sinkX, p.sinkY);

        nodeStates[i]  = NodeState(nid, v.x, v.y, p.initEnergy, dSink);
        idToIdx[nid]   = i;
    }

    // Clustering IFO initial
    IFOClustering ifo;
    ifo.SetArea(p.sinkX, p.sinkY, p.areaSize, p.radioRange,
                p.initEnergy, p.ifoIterations);

    const uint32_t nc0 = (p.nClusters > 0)
                       ? p.nClusters
                       : ifo.ComputeNClusters(nodeStates);
    ifo.Run(nodeStates, nc0);

    if (g_rl.IsConnected())
        g_rl.SendTopology(ifo.GetClusters());

    {
        const auto stats = ifo.GetStats();
        NS_LOG_UNCOND("[IFO] Initial : " << stats.nClusters << " clusters"
                    << " | membres min=" << stats.membersMin
                    << " max=" << stats.membersMax
                    << " moy=" << std::fixed << std::setprecision(1)
                    << stats.membersMean);
    }

    ExportTopology(p.resultsDir + "/topology/fdqnte_topology_initial.csv",
                   nodeStates, "initial");

    // Q-table fallback
    std::map<uint32_t, QAgent> qAgents;
    for (auto& n : nodeStates) qAgents[n.id] = QAgent();

    // Fichiers CSV
    std::ofstream energyCSV, routingCSV;
    InitEnergyCSV(energyCSV,   p.resultsDir + "/energy/fdqnte_energy.csv");
    InitRoutingCSV(routingCSV, p.resultsDir + "/routing/fdqnte_routing.csv");

    // Flow Monitor
    FlowMonitorHelper fmH;
    Ptr<FlowMonitor> fm = fmH.InstallAll();

    // Contexte partagé
    ctx.nNodes       = p.nNodes;
    ctx.simDuration  = p.simDuration;
    ctx.initEnergy   = p.initEnergy;
    ctx.radioRange   = p.radioRange;
    ctx.sinkX        = p.sinkX;
    ctx.sinkY        = p.sinkY;
    ctx.areaSize     = p.areaSize;
    ctx.seed         = p.seed;
    ctx.pSrc         = &eSources;
    ctx.pSt          = &nodeStates;
    ctx.pSens        = &sensors;
    ctx.pDevs        = &devs;
    ctx.pIfo         = &ifo;
    ctx.pIdIdx       = &idToIdx;
    ctx.pQA          = &qAgents;
    ctx.pRng         = &rng;
    ctx.pEnergyCSV   = &energyCSV;
    ctx.pRoutingCSV  = &routingCSV;
    ctx.pFlowMonitor = fm;
    ctx.resultsDir   = p.resultsDir;

    // Initialisation des callbacks
    InitRLStep();
    InitDoCheck();

    // Scheduling
    Simulator::Schedule(Seconds(2.0), rlStep);
    Simulator::Schedule(Seconds(FdqnCfg::METRICS_INTERVAL),
                        [](){ doCheck(1); });

    // +0.001 s : NS-3 stoppe AVANT les événements schedulés au même instant exact.
    // Sans ce delta, le round à t=simDuration (ex: t=3000s) est schedulé mais
    // jamais exécuté → les fichiers CSV/JSON s'arrêtent 50s avant la fin.
    Simulator::Stop(Seconds(p.simDuration + 0.001));
    NS_LOG_UNCOND("▶ Simulation démarrée pour (" << p.simDuration << "s)...");
    Simulator::Run();

    // POST-SIMULATION
    g_compMetrics.simulationEndTime = Simulator::Now().GetSeconds();

    fm->CheckForLostPackets();
    const auto& flowStats = fm->GetFlowStats();
    for (const auto& kv : flowStats) {
        g_compMetrics.totalPacketsSent += kv.second.txPackets;
        g_compMetrics.totalPacketsReceived += kv.second.rxPackets;
    }
    const double pdr = g_compMetrics.totalPacketsSent > 0
                     ? 100.0 * static_cast<double>(g_compMetrics.totalPacketsReceived) / g_compMetrics.totalPacketsSent
                     : 0.0;

    uint32_t alive = 0; double sumE = 0.0;
    for (const auto& ns : nodeStates) {
        if (ns.isAlive) { alive++; sumE += ns.energy; }
    }
    const double meanEFinal = alive > 0 ? sumE / alive : 0.0;
    g_compMetrics.totalEnergyConsumed_J = ctx.nNodes * ctx.initEnergy - sumE;
    g_compMetrics.avgPDR_RL = g_compMetrics.pdrRLPerRound.empty() ? 0.0 :
                               std::accumulate(g_compMetrics.pdrRLPerRound.begin(),
                                        g_compMetrics.pdrRLPerRound.end(), 0.0)
                                        / g_compMetrics.pdrRLPerRound.size();
    g_compMetrics.avgPDR_NS3 = g_compMetrics.pdrNS3PerRound.empty() ? 0.0 :
                                std::accumulate(g_compMetrics.pdrNS3PerRound.begin(),
                                        g_compMetrics.pdrNS3PerRound.end(), 0.0)
                                        / g_compMetrics.pdrNS3PerRound.size();

    double pdrRL = 100.0;
    if (ctx.rlPktEmitted > 0)
        pdrRL = 100.0 * (double)ctx.rlPktDelivered / ctx.rlPktEmitted;

    // PDR pré-FND (phase stable) — la vraie métrique de performance de routage
    double pdrRL_preFND = 100.0;
    if (g_compMetrics.pdrRL_preFND_locked && g_compMetrics.pdrRL_preFND_emitted > 0)
        pdrRL_preFND = 100.0 * (double)g_compMetrics.pdrRL_preFND_delivered
                             / g_compMetrics.pdrRL_preFND_emitted;
    g_compMetrics.avgPDR_RL_preFND = pdrRL_preFND;  // disponible dans ExportComparisonMetrics

    NS_LOG_UNCOND("\n╔══════════════════════════════════════════════════════════════╗");
    NS_LOG_UNCOND(  "║                         RÉSULTATS FDQN-TE+                   ║");
    NS_LOG_UNCOND(  "╠══════════════════════════════════════════════════════════════╣");
    NS_LOG_UNCOND(  "║ Nœuds vivants   : " << alive << " / " << p.nNodes);
    NS_LOG_UNCOND(  "║ Nœuds morts     : " << g_deadNodes.size());
    NS_LOG_UNCOND(  "║ Énergie moyenne : " << std::fixed << std::setprecision(4) << meanEFinal << " J");
    NS_LOG_UNCOND(  "║ Énergie totale consommée : " << g_compMetrics.totalEnergyConsumed_J << " J");
    NS_LOG_UNCOND(  "║ PDR (logique RL): " << std::setprecision(1) << pdrRL
                  << " % (" << ctx.rlPktDelivered << "/" << ctx.rlPktEmitted << ") [Fin Simulation]");
    NS_LOG_UNCOND(  "║ PDR (RL pré-FND): " << std::setprecision(1) << pdrRL_preFND
                  << " % (" << g_compMetrics.pdrRL_preFND_delivered << "/"
                  << g_compMetrics.pdrRL_preFND_emitted << ") [réseau stable]");
    NS_LOG_UNCOND(  "║ PDR (NS-3 phys) : " << std::setprecision(1) << pdr << " %");
    NS_LOG_UNCOND(  "║ Délai moyen     : " << std::setprecision(2) << g_compMetrics.avgDelay_ms << " ms");
    NS_LOG_UNCOND(  "║ Paquets NS-3 émis: " << g_compMetrics.totalPacketsSent);
    NS_LOG_UNCOND(  "║ Paquets NS-3 reçus: " << g_compMetrics.totalPacketsReceived);
    NS_LOG_UNCOND(  "║ Rounds IFO      : " << ifo.GetRound());
    NS_LOG_UNCOND(  "║ RL Steps        : " << g_rl.GetSteps());
    NS_LOG_UNCOND(  "║ Fed Rounds      : " << g_rl.GetFedRound());
    NS_LOG_UNCOND(  "║ FND             : " << (g_fndDone ? std::to_string(g_fndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ HND             : " << (g_hndDone ? std::to_string(g_hndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ LND (90%)       : " << (g_lndDone ? std::to_string(g_lndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "╚══════════════════════════════════════════════════════════════╝");

    // Export résumé final
    {
        std::ofstream summary(p.resultsDir + "/fdqnte_summary.csv");
        summary << "# FDQN-TE+ Summary\n"
                << "Param,Value\n"
                << "N," << p.nNodes << "\n"
                << "AliveNodes," << alive << "\n"
                << "DeadNodes," << g_deadNodes.size() << "\n"
                << "EnergyMean_J," << std::fixed << std::setprecision(6) << meanEFinal << "\n"
                << "EnergyTotalConsumed_J," << g_compMetrics.totalEnergyConsumed_J << "\n"
                << "PDR_RL_pct," << std::setprecision(2) << pdrRL << "\n"
                << "PDR_RL_preFND_pct," << std::setprecision(2) << pdrRL_preFND << "\n"
                << "PDR_NS3_pct," << std::setprecision(2) << pdr << "\n"
                << "AvgDelay_ms," << std::setprecision(2) << g_compMetrics.avgDelay_ms << "\n"
                << "TxPackets," << g_compMetrics.totalPacketsSent << "\n"
                << "RxPackets," << g_compMetrics.totalPacketsReceived << "\n"
                << "RL_PktEmitted," << ctx.rlPktEmitted << "\n"
                << "RL_PktDelivered," << ctx.rlPktDelivered << "\n"
                << "IFO_Rounds," << ifo.GetRound() << "\n"
                << "FND_t," << (g_fndDone ? g_fndTime : 0.0) << "\n"
                << "HND_t," << (g_hndDone ? g_hndTime : 0.0) << "\n"
                << "LND_t," << (g_lndDone ? g_lndTime : 0.0) << "\n"
                << "RL_Steps," << g_rl.GetSteps() << "\n"
                << "FedRounds," << g_rl.GetFedRound() << "\n"
                << "Seed," << p.seed << "\n"
                << "SimDuration_s," << p.simDuration << "\n"
                << "RadioRange_m," << p.radioRange << "\n"
                << "AreaSize_m," << p.areaSize << "\n"
                << "InitEnergy_J," << p.initEnergy << "\n";
    }

    ExportTopology(p.resultsDir + "/topology/fdqnte_topology_final.csv",
                   nodeStates, "final");

    // Export de l'historique RL
    ExportRLHistory();

    // Export des métriques de comparaison
    ExportComparisonMetrics();

    auto simEndTime = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(simEndTime - simStartTime);
    NS_LOG_UNCOND("\n⏱ Temps d'exécution total: " << simDuration.count()/60000 << " mn");

    Simulator::Destroy();
    return 0;
}
