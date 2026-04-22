/* =============================================================================
 * leach_sim.cc — LEACH Baseline (Heinzelman et al., 2000/2002)
 *
 * Protocole LEACH fidèle :
 *
 *   Phase SETUP (chaque RECLUSTER_PERIOD) :
 *     - Election CH probabiliste : T(n) = p / (1 - p*(r mod 1/p))
 *     - CHs broadcastent leur statut
 *     - Chaque non-CH rejoint le CH le plus proche dans RADIO_RANGE
 *     - Si aucun CH dans portée → nœud ISOLÉ (orphelin)
 *
 *   Phase STEADY-STATE :
 *     - Membre → CH (1 saut physique, dist <= RADIO_RANGE)
 *     - CH → sink (1 saut DIRECT sans limite de distance)
 *       Le sink est une Base Station à longue portée — pas RADIO_RANGE
 *       (c'est le modèle Heinzelman : E_amp*d_to_BS² sans coupure distance)
 *
 *   Modèle PDR :
 *     - CH delivery    = CH survit au drain de ce step
 *     - Membre delivery = CH dans portée radio ET CH survit ce step
 *     - Nœud isolé     = PAQUET PERDU (pas de CH dans portée)
 *
 *   Source de pertes PDR naturelles (sans contrainte artificielle) :
 *     - Nœuds isolés (~5-15% selon topologie, plus en fin de vie)
 *     - CHs qui meurent en cours de step (énergie épuisée)
 *     - Membres dont le CH meurt pendant ce step
 *
 * Usage :
 *   ./ns3 run "scratch/leach_sim --resultsDir=results_eval/LEACH"
 * ============================================================================= */

#include "eval_config.h"
#include "eval_common.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/olsr-module.h"

#include <random>
#include <deque>
#include <map>
#include <set>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <cmath>

using namespace ns3;
namespace fs = std::filesystem;
NS_LOG_COMPONENT_DEFINE("LeachSim");

// =============================================================================
// ComputeLeachLogicalDelay_ms — Délai bout-en-bout logique LEACH
//
// Remplace EvalDelay::ComputeDelay_ms(2, -1.0, nAlive) qui retournait une
// valeur statique basée sur un modèle 802.15.4 250kbps — incohérent avec
// la PHY 802.11b 1Mbps DsssRate1Mbps configurée dans la simulation.
//
// Modèle LEACH (Heinzelman 2000/2002) :
//   - Membre → CH : 1 saut physique (dist <= RADIO_RANGE)
//   - CH → sink   : 1 saut DIRECT (BS longue portée, sans contrainte RADIO_RANGE)
//   - Total membre : 2 sauts, total CH : 1 saut
//
// Paramètres 802.11b DSSS 1 Mbps (cohérents avec DsssRate1Mbps dans main()) :
//   txTime  = 500 octets × 8 bits / 1e6 bps × 1000 = 4.0 ms
//   macOH   = 0.832 ms (SIFS+ACK+DIFS+backoff moyen, mesuré NS-3 802.11b)
//   → delayPerHop = 4.832 ms
// =============================================================================
static double ComputeLeachLogicalDelay_ms(
    const std::vector<EvalNodeState>& nodes,
    double radioRange)
{
    static constexpr double PKT_BYTES       = 500.0;
    static constexpr double PHY_RATE_BPS    = 1.0e6;   // 802.11b DSSS 1Mbps
    static constexpr double MAC_OVERHEAD_MS = 0.832;   // overhead MAC 802.11b

    const double txTime_ms      = (PKT_BYTES * 8.0 / PHY_RATE_BPS) * 1000.0; // 4.0 ms
    const double delayPerHop_ms = txTime_ms + MAC_OVERHEAD_MS;                // 4.832 ms

    double   sumDelay = 0.0;
    uint32_t count    = 0;

    for (const auto& ns : nodes) {
        if (!ns.isAlive) continue;

        if (ns.isClusterHead) {
            // CH → sink : 1 saut direct (BS longue portée LEACH)
            sumDelay += 1.0 * delayPerHop_ms;
            count++;
        } else if (ns.clusterId != 0) {
            // Membre → CH : 1 saut physique (dans RADIO_RANGE par construction)
            // CH → sink   : 1 saut direct
            // Total : 2 sauts
            sumDelay += 2.0 * delayPerHop_ms;
            count++;
        } else {
            // Nœud isolé : tente TX direct vers sink (paquet perdu, mais
            // contribue au délai moyen avec la distance réelle)
            double hops = (ns.distToSink <= radioRange)
                ? 1.0
                : std::ceil(ns.distToSink / radioRange);
            sumDelay += hops * delayPerHop_ms;
            count++;
        }
    }

    return count > 0 ? sumDelay / count : 0.0;
}

// ─── Globals ─────────────────────────────────────────────────────────────────

static std::set<uint32_t> g_dead;
static bool   g_fndDone = false, g_hndDone = false, g_lndDone = false;
static double g_fndTime = 0.0,   g_hndTime = 0.0,   g_lndTime = 0.0;
static double g_pdrStable = -1.0;

// ─── Agent LEACH ─────────────────────────────────────────────────────────────

struct LeachAgent {
    uint32_t roundsSinceCH = 0;
    bool     wasCHRecently = false;
    static constexpr uint32_t ROUND_WAIT =
        static_cast<uint32_t>(1.0 / EvalCfg::LEACH_P); // 20 rounds
};

// Formule Heinzelman 2002, eq. (1)
bool IsElectedCH(uint32_t nodeId, uint32_t round,
                 std::mt19937& rng,
                 std::map<uint32_t, LeachAgent>& agents) {
    auto& ag = agents[nodeId];
    if (ag.wasCHRecently && ag.roundsSinceCH < LeachAgent::ROUND_WAIT)
        return false;
    const uint32_t slot = round % LeachAgent::ROUND_WAIT;
    double threshold;
    if (slot == 0) {
        threshold = EvalCfg::LEACH_P;
        ag.wasCHRecently = false;
    } else {
        const double denom = 1.0 - EvalCfg::LEACH_P * static_cast<double>(slot);
        threshold = (denom > 1e-9) ? EvalCfg::LEACH_P / denom : 1.0;
    }
    threshold = std::max(0.0, std::min(1.0, threshold));
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    return ud(rng) < threshold;
}

// Formation des clusters
// Nœud rejoint le CH le plus proche dans RADIO_RANGE
// Si aucun CH dans portée → clusterId=0 (nœud isolé, paquet perdu)
void FormClusters(std::vector<EvalNodeState>& nodes) {
    for (auto& n : nodes) {
        if (!n.isAlive || n.isClusterHead) continue;
        double   bestDist = 1e18;
        uint32_t bestCH   = 0;
        for (const auto& ch : nodes) {
            if (!ch.isAlive || !ch.isClusterHead) continue;
            const double d = NodeDist(n, ch);
            if (d <= EvalCfg::RADIO_RANGE && d < bestDist) {
                bestDist = d; bestCH = ch.id;
            }
        }
        n.clusterId = bestCH; // 0 = isolé
    }
}

// ─── Contexte ────────────────────────────────────────────────────────────────

struct LeachContext {
    uint32_t nNodes;
    double   initEnergy, sinkX, sinkY, simDuration;
    std::vector<EvalNodeState>*     pSt;
    std::map<uint32_t, LeachAgent>* pAgents;
    std::map<uint32_t, uint32_t>*   pIdIdx;
    std::mt19937*                   pRng;
    std::ofstream*                  pMetricsCSV;
    std::string resultsDir;

    uint64_t rlPktEmitted   = 0;
    uint64_t rlPktDelivered = 0;
    // FIX A2: per-round counters for instantaneous PDR
    uint64_t roundPktEmitted   = 0;
    uint64_t roundPktDelivered = 0;
    // FIX A4: moyenne glissante PDR sur 5 rounds (lisse les aléas élection CH)
    std::deque<double> pdrWindow;  // ring buffer taille 5
    static constexpr size_t PDR_SMOOTH = 5;

    // PDR pré-FND gelé au moment exact du FND
    uint64_t preFND_emitted   = 0;
    uint64_t preFND_delivered = 0;
    bool     preFND_locked    = false;

    Ptr<FlowMonitor> pFM;
};

static LeachContext              g_ctx;
static std::function<void()>     leachStep;
static std::function<void(int)>  doMetrics;
static ModelSummary              g_summary;
static std::vector<RoundMetrics> g_history;

// ─── Step LEACH ──────────────────────────────────────────────────────────────

static void InitLeachStep() {
    leachStep = [&]() {
        const double now   = Simulator::Now().GetSeconds();
        auto& nodes  = *g_ctx.pSt;
        auto& agents = *g_ctx.pAgents;
        auto& rng    = *g_ctx.pRng;

        // 1. Synchroniser les morts
        for (auto& n : nodes)
            if (g_dead.count(n.id)) n.isAlive = false;

        // 2. Election CH toutes les RECLUSTER_PERIOD secondes
        const uint32_t leachRound = static_cast<uint32_t>(
            now / EvalCfg::RECLUSTER_PERIOD);

        for (auto& n : nodes) {
            if (!n.isAlive) continue;
            n.isClusterHead = false;
            n.clusterId     = 0;
            agents[n.id].roundsSinceCH++;
        }

        uint32_t nCH = 0;
        for (auto& n : nodes) {
            if (!n.isAlive) continue;
            if (IsElectedCH(n.id, leachRound, rng, agents)) {
                n.isClusterHead = true;
                n.clusterId     = n.id;
                agents[n.id].wasCHRecently = true;
                agents[n.id].roundsSinceCH = 0;
                nCH++;
            }
        }

        // Garantir un minimum de CHs (évite réseau sans aucun CH)
        uint32_t nAlive = 0;
        for (const auto& n : nodes) if (n.isAlive) nAlive++;
        const uint32_t minCH = std::max(1u,
            static_cast<uint32_t>(std::ceil(
                static_cast<double>(nAlive) / EvalCfg::CLUSTER_MEM_MAX)));
        if (nCH < minCH) {
            std::vector<EvalNodeState*> cands;
            for (auto& n : nodes)
                if (n.isAlive && !n.isClusterHead) cands.push_back(&n);
            std::sort(cands.begin(), cands.end(),
                [](const EvalNodeState* a, const EvalNodeState* b) {
                    return a->energy > b->energy; });
            for (size_t k = 0; k < cands.size() && nCH < minCH; k++, nCH++) {
                cands[k]->isClusterHead = true;
                cands[k]->clusterId     = cands[k]->id;
            }
        }

        FormClusters(nodes);

        // ── 3. Pré-calcul de l'énergie après drain ───────────────────────────
        // Nécessaire pour savoir si le CH SURVIVRA ce step avant de compter
        // la delivery des membres qui lui ont envoyé leurs paquets.
        std::map<uint32_t, double> energyAfter;
        for (const auto& ns : nodes) {
            if (!ns.isAlive) continue;
            double drain = 0.0;
            if (ns.isClusterHead) {
                uint32_t nMem = 0;
                for (const auto& m : nodes)
                    if (m.isAlive && !m.isClusterHead && m.clusterId == ns.id)
                        nMem++;
                drain = EvalDrainCH(nMem, ns.distToSink);
            } else if (ns.clusterId != 0) {
                double dToCH = EvalCfg::RADIO_RANGE;
                for (const auto& ch : nodes)
                    if (ch.id == ns.clusterId && ch.isAlive) {
                        dToCH = NodeDist(ns, ch); break;
                    }
                drain = EvalDrainMember(dToCH);
            } else {
                // Nœud isolé : consomme quand même de l'énergie en tentant TX
                drain = EvalEtx(EvalCfg::DRAIN_BITS, ns.distToSink);
            }
            energyAfter[ns.id] = ns.energy - std::max(0.0, drain);
        }

        // ── 4. PDR + drain ───────────────────────────────────────────────────
        //
        // Règles de delivery LEACH (Heinzelman 2000) :
        //
        //   CH :
        //     Le sink est une Base Station à longue portée (pas RADIO_RANGE).
        //     Delivery = CH survit à son drain ce step.
        //     (EvalDrainCH calcule déjà l'énergie TX vers sink sans limite radio)
        //
        //   Membre :
        //     1 saut physique vers son CH.
        //     Delivery = CH dans portée radio ET CH survit ce step
        //     (si CH mort ce step → paquet perdu, le CH n'a pas pu relayer)
        //
        //   Nœud isolé (clusterId=0) :
        //     Aucun CH dans portée radio → paquet PERDU.
        //     C'est la source de pertes principale de LEACH.

        for (auto& ns : nodes) {
            if (!ns.isAlive) continue;

            double drain     = 0.0;
            bool   delivered = false;

            if (ns.isClusterHead) {
                // CH → sink (BS longue portée, pas de contrainte RADIO_RANGE)
                uint32_t nMem = 0;
                for (const auto& m : nodes)
                    if (m.isAlive && !m.isClusterHead && m.clusterId == ns.id)
                        nMem++;
                drain = EvalDrainCH(nMem, ns.distToSink);

                // Delivery : CH survit à ce step
                delivered = (energyAfter.count(ns.id) &&
                             energyAfter[ns.id] > 0.0);

            } else if (ns.clusterId == 0) {
                // Nœud isolé → paquet perdu
                // Consomme quand même l'énergie de la tentative TX
                drain     = EvalEtx(EvalCfg::DRAIN_BITS, ns.distToSink);
                delivered = false; // PERDU : pas de CH dans portée

            } else {
                // Membre → CH (1 saut physique)
                double dToCH = EvalCfg::RADIO_RANGE;
                const EvalNodeState* chPtr = nullptr;
                for (const auto& ch : nodes) {
                    if (ch.id == ns.clusterId && ch.isAlive) {
                        dToCH = NodeDist(ns, ch);
                        chPtr = &ch;
                        break;
                    }
                }
                drain = EvalDrainMember(dToCH);

                // Delivery : CH dans portée ET CH survit ce step
                // (si CH meurt ce step, il n'a pas pu relayer vers sink)
                const bool chInRange  = (chPtr != nullptr) &&
                                        (dToCH <= EvalCfg::RADIO_RANGE);
                const bool chSurvives = chInRange &&
                    energyAfter.count(chPtr->id) &&
                    energyAfter[chPtr->id] > 0.0;
                delivered = chInRange && chSurvives;
            }

            // Comptage PDR
            g_ctx.rlPktEmitted++;
            g_ctx.roundPktEmitted++;
            if (delivered) {
                g_ctx.rlPktDelivered++;
                g_ctx.roundPktDelivered++;
            }

            // Appliquer drain et détecter morts
            const bool wasAlive = ns.isAlive;
            if (!ns.Consume(drain) && wasAlive && !g_dead.count(ns.id)) {
                g_dead.insert(ns.id);

                if (!g_fndDone) {
                    if (!g_ctx.preFND_locked) {
                        g_ctx.preFND_emitted   = g_ctx.rlPktEmitted;
                        g_ctx.preFND_delivered = g_ctx.rlPktDelivered;
                        g_ctx.preFND_locked    = true;
                    }
                    g_pdrStable = g_ctx.preFND_emitted > 0
                        ? 100.0 * g_ctx.preFND_delivered / g_ctx.preFND_emitted
                        : 100.0;
                    g_fndDone = true; g_fndTime = now;
                    g_summary.fnd_s = now;
                    NS_LOG_UNCOND("⭐ [FND] t=" << now
                        << "s | PDR_stable="
                        << std::fixed << std::setprecision(1)
                        << g_pdrStable << "%");
                }
                if (!g_hndDone && g_dead.size() >= g_ctx.nNodes / 2) {
                    g_hndDone = true; g_hndTime = now;
                    g_summary.hnd_s = now;
                    NS_LOG_UNCOND("⭐ [HND] t=" << now << "s");
                }
                if (!g_lndDone &&
                    g_dead.size() >= static_cast<uint32_t>(g_ctx.nNodes * 0.9)) {
                    g_lndDone = true; g_lndTime = now;
                    g_summary.lnd_s = now;
                    NS_LOG_UNCOND("⭐ [LND-90%] t=" << now << "s");
                    Simulator::Stop();
                }
            }
            ns.txCount++;
        }

        if (now + EvalCfg::RL_STEP_INTERVAL <= g_ctx.simDuration)
            Simulator::Schedule(Seconds(EvalCfg::RL_STEP_INTERVAL), leachStep);
    };
}

// ─── Métriques ───────────────────────────────────────────────────────────────

static void InitDoMetrics() {
    doMetrics = [&](int round) {
        const double now = Simulator::Now().GetSeconds();
        auto& nodes = *g_ctx.pSt;
        for (auto& n : nodes) if (g_dead.count(n.id)) n.isAlive = false;

        uint32_t nCH = 0, nAlive = 0, nIsolated = 0;
        for (const auto& n : nodes) {
            if (!n.isAlive) continue;
            nAlive++;
            if (n.isClusterHead) nCH++;
            if (!n.isClusterHead && n.clusterId == 0) nIsolated++;
        }

        // FIX A2: PDR instantane par round — evite l'artefact du round 1
        // (round 1 cumulatif = 71% car peu d'echantillons; par round = valeur reelle)
        const double pdrRawRL = g_ctx.roundPktEmitted > 0
            ? 100.0 * g_ctx.roundPktDelivered / g_ctx.roundPktEmitted : 100.0;
        // FIX A4: lissage sur 5 rounds — supprime les oscillations ±20%
        // dues à la variabilité de l'élection CH aléatoire de LEACH
        g_ctx.pdrWindow.push_back(pdrRawRL);
        if (g_ctx.pdrWindow.size() > g_ctx.PDR_SMOOTH)
            g_ctx.pdrWindow.pop_front();
        double pdrSum = 0.0;
        for (double v : g_ctx.pdrWindow) pdrSum += v;
        const double pdrRoundRL = pdrSum / g_ctx.pdrWindow.size();
        const double pdrRL = g_ctx.rlPktEmitted > 0
            ? 100.0 * g_ctx.rlPktDelivered / g_ctx.rlPktEmitted : 100.0;
        // Reinitialiser pour le prochain round
        g_ctx.roundPktEmitted   = 0;
        g_ctx.roundPktDelivered = 0;

        // Délai LEACH logique bout-en-bout
        // Remplace EvalDelay::ComputeDelay_ms(2,-1.0,nAlive) — valeur statique
        // basée sur 802.15.4 250kbps, incohérente avec la PHY 802.11b 1Mbps.
        // ComputeLeachLogicalDelay_ms() calcule le délai réel depuis la topologie :
        //   - CH       : 1 saut direct vers sink (BS longue portée LEACH)
        //   - Membre   : 2 sauts (membre→CH + CH→sink)
        //   - Isolé    : ceil(distToSink/radioRange) sauts (fallback)
        // Évolue dynamiquement avec les morts de nœuds et reconfigurations.
        const double delay_ms = ComputeLeachLogicalDelay_ms(nodes, EvalCfg::RADIO_RANGE);

        // PDR NS-3 physique
        double pdrNS3 = 100.0;
        if (g_ctx.pFM) {
            g_ctx.pFM->CheckForLostPackets();
            uint64_t tx = 0, rx = 0;
            for (const auto& kv : g_ctx.pFM->GetFlowStats()) {
                tx += kv.second.txPackets;
                rx += kv.second.rxPackets;
            }
            if (tx > 0) pdrNS3 = 100.0 * rx / tx;
        }

        RoundMetrics m = ComputeRoundMetrics(
            round, now, nodes, g_ctx.initEnergy,
            g_ctx.rlPktEmitted, g_ctx.rlPktDelivered, delay_ms, nCH);
        // FIX A2: remplacer PDR cumulatif par PDR instantane du round
        m.pdr_pct = pdrRoundRL;

        g_history.push_back(m);
        if (g_ctx.pMetricsCSV && g_ctx.pMetricsCSV->is_open())
            WriteMetricsRow(*g_ctx.pMetricsCSV, m);

        NS_LOG_UNCOND(std::fixed
            << "[LEACH Round " << round
            << "] t="        << std::setprecision(0) << now << "s"
            << "|vivants="   << nAlive
            << "|morts="     << (g_ctx.nNodes - nAlive)
            << "|isolés="    << nIsolated
            << "|E_moy="     << std::setprecision(3) << m.energyMean_J << "J"
            << "|PDR_RL="    << std::setprecision(1) << pdrRL << "%"
            << "|delay="     << delay_ms << "ms"
            << "|CH="        << nCH);

        if (now + EvalCfg::METRICS_INTERVAL <= g_ctx.simDuration)
            Simulator::Schedule(Seconds(EvalCfg::METRICS_INTERVAL),
                                [r = round + 1]() { doMetrics(r); });
    };
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string resultsDir = "results/LEACH";
    uint32_t    seed       = EvalCfg::SEED;
    uint32_t    nNodes     = 300;
    CommandLine cmd;
    cmd.AddValue("resultsDir", "Dossier résultats", resultsDir);
    cmd.AddValue("seed",       "Seed aléatoire",    seed);
    cmd.AddValue("nNodes",     "Nombre de nœuds",   nNodes);
    cmd.Parse(argc, argv);

    fs::create_directories(resultsDir);
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);
    std::mt19937 rng(seed);

    NS_LOG_UNCOND("╔═══════════════════════════════╗");
    NS_LOG_UNCOND("║  LEACH Baseline — Démarrage   ║");
    NS_LOG_UNCOND("╚═══════════════════════════════╝");
    NS_LOG_UNCOND("  N=" << nNodes
        << " E=" << EvalCfg::E_INIT << "J"
        << " Seed=" << seed
        << " Portée=" << EvalCfg::RADIO_RANGE << "m"
        << " Zone=" << (int)EvalCfg::AREA_SIZE << "m");

    // ── NS-3 setup ───────────────────────────────────────────────────────────
    NodeContainer sensors, sinkNode;
    sensors.Create(nNodes);
    sinkNode.Create(1);

    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max="
                         + std::to_string((int)EvalCfg::AREA_SIZE) + "]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max="
                         + std::to_string((int)EvalCfg::AREA_SIZE) + "]"));
    mob.Install(sensors);
    mob.SetPositionAllocator("ns3::GridPositionAllocator",
        "MinX", DoubleValue(EvalCfg::SINK_X),
        "MinY", DoubleValue(EvalCfg::SINK_Y));
    mob.Install(sinkNode);

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
        "MaxRange", DoubleValue(EvalCfg::RADIO_RANGE));
    phy.SetChannel(wch.Create());

    NodeContainer all; all.Add(sensors); all.Add(sinkNode);
    NetDeviceContainer devs = wifi.Install(phy, mac, all);

    OlsrHelper olsr;
    InternetStackHelper inet;
    inet.SetRoutingHelper(olsr);
    inet.Install(all);
    Ipv4AddressHelper ip;
    ip.SetBase("10.0.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifaces = ip.Assign(devs);

    // ── États logiques ───────────────────────────────────────────────────────
    auto states = InitNodeStates(sensors, nNodes,
                                 EvalCfg::SINK_X, EvalCfg::SINK_Y,
                                 EvalCfg::E_INIT);
    std::map<uint32_t, uint32_t> idToIdx;
    for (uint32_t i = 0; i < nNodes; i++)
        idToIdx[states[i].id] = i;

    std::map<uint32_t, LeachAgent> agents;
    for (auto& n : states) agents[n.id] = LeachAgent{};

    // ── Application UDP ──────────────────────────────────────────────────────
    const uint16_t port = 9;
    Address sinkAddr(InetSocketAddress(
        ifaces.GetAddress(nNodes), port));

    PacketSinkHelper sinkApp("ns3::UdpSocketFactory", sinkAddr);
    ApplicationContainer sApps = sinkApp.Install(sinkNode);
    sApps.Start(Seconds(1.0));
    sApps.Stop(Seconds(EvalCfg::SIM_DURATION));

    for (uint32_t i = 0; i < nNodes; i++) {
        OnOffHelper src("ns3::UdpSocketFactory", sinkAddr);
        src.SetAttribute("PacketSize", UintegerValue(EvalCfg::PKT_BITS / 8));
        src.SetAttribute("DataRate",   StringValue("10kbps"));
        src.SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=2]"));
        src.SetAttribute("OffTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
        ApplicationContainer a = src.Install(sensors.Get(i));
        a.Start(Seconds(15.0 + i * 0.01));
        a.Stop(Seconds(EvalCfg::SIM_DURATION));
    }

    FlowMonitorHelper fmH;
    Ptr<FlowMonitor> fm = fmH.InstallAll();

    // ── Contexte ─────────────────────────────────────────────────────────────
    g_ctx.nNodes      = nNodes;
    g_ctx.initEnergy  = EvalCfg::E_INIT;
    g_ctx.sinkX       = EvalCfg::SINK_X;
    g_ctx.sinkY       = EvalCfg::SINK_Y;
    g_ctx.simDuration = EvalCfg::SIM_DURATION;
    g_ctx.pSt         = &states;
    g_ctx.pAgents     = &agents;
    g_ctx.pIdIdx      = &idToIdx;
    g_ctx.pRng        = &rng;
    g_ctx.pFM         = fm;
    g_ctx.resultsDir  = resultsDir;
    g_summary.modelName = EvalCfg::MODEL_LEACH;

    std::ofstream metricsCSV;
    InitMetricsCSV(metricsCSV, resultsDir + "/metrics.csv", EvalCfg::MODEL_LEACH);
    g_ctx.pMetricsCSV = &metricsCSV;

    InitLeachStep();
    InitDoMetrics();
    Simulator::Schedule(Seconds(2.0), leachStep);
    Simulator::Schedule(Seconds(EvalCfg::METRICS_INTERVAL),
                        []() { doMetrics(1); });
    Simulator::Stop(Seconds(EvalCfg::SIM_DURATION));

    NS_LOG_UNCOND("▶ Simulation LEACH démarrée (" << EvalCfg::SIM_DURATION << "s)...");
    Simulator::Run();

    // ── Post-simulation ──────────────────────────────────────────────────────
    fm->CheckForLostPackets();
    uint64_t txPkt = 0, rxPkt = 0;
    for (const auto& kv : fm->GetFlowStats()) {
        txPkt += kv.second.txPackets;
        rxPkt += kv.second.rxPackets;
    }
    const double pdrNS3 = txPkt > 0 ? 100.0 * rxPkt / txPkt : 100.0;
    const double pdrRL  = g_ctx.rlPktEmitted > 0
        ? 100.0 * g_ctx.rlPktDelivered / g_ctx.rlPktEmitted : 100.0;

    uint32_t alive = 0; double sumE = 0.0;
    for (const auto& n : states)
        if (n.isAlive) { alive++; sumE += n.energy; }

    g_summary.pdrStable_pct = (g_pdrStable >= 0.0) ? g_pdrStable : pdrRL;
    g_summary.pdrGlobal_pct = pdrRL;
    // Délai final : même modèle logique que pendant la simulation
    g_summary.avgDelay_ms   = ComputeLeachLogicalDelay_ms(states, EvalCfg::RADIO_RANGE);
    g_summary.totalEnergy_J = nNodes * EvalCfg::E_INIT - sumE;
    g_summary.totalPktSent  = g_ctx.rlPktEmitted;
    g_summary.totalPktRecv  = g_ctx.rlPktDelivered;
    g_summary.totalRounds   = static_cast<uint32_t>(g_history.size());

    WriteSummaryCSV(resultsDir + "/summary.csv", g_summary);

    NS_LOG_UNCOND("\n╔═══════════════════════════════════════╗");
    NS_LOG_UNCOND(  "║  RÉSULTATS LEACH                       ║");
    NS_LOG_UNCOND(  "╠═══════════════════════════════════════╣");
    NS_LOG_UNCOND(  "║ Vivants     : " << alive << "/" << nNodes);
    NS_LOG_UNCOND(  "║ FND         : "
        << (g_fndDone ? std::to_string((int)g_fndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ HND         : "
        << (g_hndDone ? std::to_string((int)g_hndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ LND (90%)   : "
        << (g_lndDone ? std::to_string((int)g_lndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ PDR stable  (logique, pré-FND) : "
        << std::fixed << std::setprecision(1) << g_summary.pdrStable_pct << "%");
    NS_LOG_UNCOND(  "║ PDR global  (logique)          : "
        << g_summary.pdrGlobal_pct << "%");
    NS_LOG_UNCOND(  "║ PDR global  (NS-3 FlowMonitor) : "
        << pdrNS3 << "%");
    NS_LOG_UNCOND(  "║ E consommée : "
        << std::setprecision(3) << g_summary.totalEnergy_J << " J");
    NS_LOG_UNCOND(  "╚═══════════════════════════════════════╝");

    Simulator::Destroy();
    auto t1 = std::chrono::high_resolution_clock::now();
    NS_LOG_UNCOND("⏱ "
        << std::chrono::duration_cast<std::chrono::minutes>(t1 - t0).count()
        << " mn");
    return 0;
}
