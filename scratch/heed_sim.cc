/* =============================================================================
 * heed_sim.cc — HEED Baseline (Younis & Fahmy, 2004)
 *
 * HEED : Hybrid Energy-Efficient Distributed clustering
 *
 * Principe (fidèle à Younis & Fahmy 2004, Section III) :
 *   Phase SETUP (itérée jusqu'à convergence, max HEED_ITER itérations) :
 *     1. Probabilité de devenir CH provisoire :
 *          CHprob(i) = max(C_prob * E_res(i) / E_max, p_min)
 *        où C_prob = LEACH_P = 0.05, E_max = E_INIT, p_min = 1e-4
 *     2. Un nœud se déclare CH final s'il n'a pas trouvé de CH voisin moins
 *        "coûteux" (coût = charge du cluster, proxy = nombre de membres).
 *     3. Association des membres : rejoindre le CH voisin avec le plus d'énergie
 *        (critère secondaire HEED : "average minimum reachability power")
 *        → ici : CH avec max énergie résiduelle dans RADIO_RANGE.
 *
 * Différence clé vs LEACH :
 *   - LEACH : probabilité uniforme → CH quelconques
 *   - HEED  : probabilité ∝ E_res → CH énergétiques → meilleur équilibre
 *   → FND et HND significativement plus longs que LEACH
 *   → Comparable à l'apport d'IFO pour valider l'architecture clustering

 * Usage :
 *   ./ns3 run "scratch/heed_sim --resultsDir=results_eval/HEED"
 *   ./ns3 run "scratch/heed_sim --resultsDir=results_eval/HEED --seed=43"
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
NS_LOG_COMPONENT_DEFINE("HeedSim");

// ─── Paramètres HEED ─────────────────────────────────────────────────────────

namespace HeedCfg {
    constexpr double C_PROB    = EvalCfg::LEACH_P;   // 0.05 — même base que LEACH
    constexpr double P_MIN     = 1e-4;                // probabilité minimale CH
    constexpr uint32_t MAX_IT  = 10;                  // itérations convergence HEED
}

// ─── Délai logique HEED ──────────────────────────────────────────────────────
// Même modèle physique que LEACH : 802.11b DSSS 1Mbps
// D_tx = 500 octets × 8 / 1e6 bps = 4.0 ms ; MAC overhead = 0.832 ms

static double ComputeHeedLogicalDelay_ms(
    const std::vector<EvalNodeState>& nodes,
    double radioRange)
{
    constexpr double TX_MS   = 4.0;    // 500B @ 1Mbps
    constexpr double MAC_MS  = 0.832;  // SIFS+ACK+DIFS+backoff 802.11b
    const double HOP_MS = TX_MS + MAC_MS;

    double   sumDelay = 0.0;
    uint32_t count    = 0;

    for (const auto& ns : nodes) {
        if (!ns.isAlive) continue;
        if (ns.isClusterHead) {
            // CH → sink : 1 saut direct (BS longue portée)
            sumDelay += 1.0 * HOP_MS;
        } else if (ns.clusterId != 0) {
            // Membre → CH (1 saut) + CH → sink (1 saut) = 2 sauts
            sumDelay += 2.0 * HOP_MS;
        } else {
            // Nœud isolé : paquet perdu — exclu de la moyenne
            // Cohérent avec leach_sim et qrouting_sim : pas de délai pour paquet non livré
            continue;
        }
        count++;
    }
    return count > 0 ? sumDelay / count : 0.0;
}

// ─── Globals ─────────────────────────────────────────────────────────────────

static std::set<uint32_t> g_dead;
static bool   g_fndDone = false, g_hndDone = false, g_lndDone = false;
static double g_fndTime = 0.0,   g_hndTime = 0.0,   g_lndTime = 0.0;
static double g_pdrStable = -1.0;

// ─── Algorithme HEED ─────────────────────────────────────────────────────────

/**
 * Exécute une phase SETUP HEED complète sur les nœuds vivants.
 * Met à jour isClusterHead et clusterId pour tous les nœuds.
 *
 * Algorithme (Younis & Fahmy 2004, Algo 1) :
 *   1. Chaque nœud calcule CHprob(i) = max(C_prob * E_res/E_max, p_min)
 *   2. Itérations de convergence :
 *      - Si CHprob >= 1 → CH final
 *      - Sinon : probabilité temporaire → annonce candidature
 *      - Chaque non-CH choisit le CH voisin avec max énergie
 *      - CHprob *= 2 à chaque itération
 *   3. Nœuds sans CH dans portée → isolés (clusterId=0)
 */
void RunHeedSetup(std::vector<EvalNodeState>& nodes, std::mt19937& rng)
{
    const double e_max = EvalCfg::E_INIT;

    // États HEED par nœud
    std::map<uint32_t, double> ch_prob;
    std::map<uint32_t, bool>   is_ch_final;
    std::map<uint32_t, bool>   is_ch_tent;   // candidat provisoire
    std::map<uint32_t, bool>   has_cluster;  // a déjà trouvé un CH voisin

    for (auto& n : nodes) {
        if (!n.isAlive) continue;
        n.isClusterHead = false;
        n.clusterId     = 0;
        double p = std::max(HeedCfg::C_PROB * n.energy / e_max,
                            HeedCfg::P_MIN);
        ch_prob[n.id]   = std::min(p, 1.0);
        is_ch_final[n.id] = false;
        is_ch_tent[n.id]  = false;
        has_cluster[n.id] = false;
    }

    std::uniform_real_distribution<double> ud(0.0, 1.0);

    // ── Itérations de convergence HEED (Younis & Fahmy 2004, Algo 1) ─────────
    //
    //   Condition d'arrêt du doubling :
    //   Un nœud qui a déjà trouvé un CH voisin (has_cluster=true) ne double plus
    //   sa probabilité et ne devient pas CH_FINAL.
    //   Sans cette condition, ch_prob atteint 1.0 en 5 itérations pour TOUS les
    //   nœuds → 300 CHs pour 300 nœuds (log : CH=300 rounds 1-3).
    //
    //   Un nœud devient CH_FINAL seulement si ch_prob >= 1.0 ET il n'a pas de
    //   CH voisin avec une énergie résiduelle supérieure (critère HEED original).
    //   Le fallback "self-election" de l'ancienne version transformait TOUS les
    //   nœuds sans cluster en CHs supplémentaires — supprimé.

    for (uint32_t it = 0; it < HeedCfg::MAX_IT; it++) {

        // Phase 1 : candidature provisoire
        for (auto& n : nodes) {
            if (!n.isAlive || is_ch_final[n.id] || has_cluster[n.id]) continue;
            if (ud(rng) < ch_prob[n.id])
                is_ch_tent[n.id] = true;
            // Doubling — stoppé si le nœud a déjà un CH [FIX-HEED-1]
            ch_prob[n.id] = std::min(ch_prob[n.id] * 2.0, 1.0);
        }

        // Phase 2 : association — chaque non-candidat cherche le meilleur CH voisin
        for (auto& n : nodes) {
            if (!n.isAlive || is_ch_tent[n.id] || is_ch_final[n.id]) continue;
            uint32_t bestCH = 0;
            double   bestE  = -1.0;
            for (const auto& ch : nodes) {
                if (!ch.isAlive) continue;
                if (!is_ch_tent[ch.id] && !is_ch_final[ch.id]) continue;
                const double d = NodeDist(n, ch);
                if (d <= EvalCfg::RADIO_RANGE && ch.energy > bestE) {
                    bestE  = ch.energy;
                    bestCH = ch.id;
                }
            }
            if (bestCH != 0) {
                n.clusterId      = bestCH;
                has_cluster[n.id]= true; // stoppe le doubling [FIX-HEED-1]
            }
        }

        // Phase 3 : promotion CH_FINAL si ch_prob>=1 et meilleur CH voisin [FIX-HEED-2]
        for (auto& n : nodes) {
            if (!n.isAlive || is_ch_final[n.id] || has_cluster[n.id]) continue;
            if (ch_prob[n.id] >= 1.0) {
                // Vérifier s'il existe un CH voisin avec plus d'énergie
                double bestNeighE = -1.0;
                for (const auto& ch : nodes) {
                    if (!ch.isAlive || ch.id == n.id) continue;
                    if (!is_ch_tent[ch.id] && !is_ch_final[ch.id]) continue;
                    if (NodeDist(n, ch) <= EvalCfg::RADIO_RANGE)
                        bestNeighE = std::max(bestNeighE, ch.energy);
                }
                if (bestNeighE < n.energy) {
                    // Aucun voisin CH plus énergétique → devient CH_FINAL
                    is_ch_final[n.id] = true;
                    is_ch_tent[n.id]  = true;
                } else {
                    // Un voisin CH plus énergétique existe → reste membre
                    has_cluster[n.id] = true;
                }
            }
        }
    }

    // ── Finalisation ──────────────────────────────────────────────────────────

    // Marquer les CH finaux
    for (auto& n : nodes) {
        if (!n.isAlive) continue;
        if (is_ch_final[n.id]) {
            n.isClusterHead = true;
            n.clusterId     = n.id;
        }
    }

    // Association finale : chaque membre rejoint le CH vivant le plus énergétique
    // dans sa portée radio. Les nœuds sans CH voisin restent isolés (clusterId=0).
    for (auto& n : nodes) {
        if (!n.isAlive || n.isClusterHead) continue;
        uint32_t bestCH = 0;
        double   bestE  = -1.0;
        for (const auto& ch : nodes) {
            if (!ch.isAlive || !ch.isClusterHead) continue;
            const double d = NodeDist(n, ch);
            if (d <= EvalCfg::RADIO_RANGE && ch.energy > bestE) {
                bestE  = ch.energy;
                bestCH = ch.id;
            }
        }
        n.clusterId = bestCH; // 0 = vraiment isolé (sans CH dans portée)
    }
}

// ─── Contexte ────────────────────────────────────────────────────────────────

struct HeedContext {
    uint32_t nNodes;
    double   initEnergy, sinkX, sinkY, simDuration;
    std::vector<EvalNodeState>* pSt;
    std::map<uint32_t, uint32_t>* pIdIdx;
    std::mt19937* pRng;
    std::ofstream* pMetricsCSV;
    std::string resultsDir;

    uint64_t rlPktEmitted    = 0;
    uint64_t rlPktDelivered  = 0;
    uint64_t roundPktEmitted  = 0;
    uint64_t roundPktDelivered= 0;
    std::deque<double> pdrWindow;
    static constexpr size_t PDR_SMOOTH = 5;

    uint64_t preFND_emitted  = 0;
    uint64_t preFND_delivered= 0;
    bool     preFND_locked   = false;

    Ptr<FlowMonitor> pFM;
};

static HeedContext               g_ctx;
static std::function<void()>     heedStep;
static std::function<void(int)>  doMetrics;
static ModelSummary              g_summary;
static std::vector<RoundMetrics> g_history;

// ─── Step HEED ───────────────────────────────────────────────────────────────

static void InitHeedStep() {
    heedStep = [&, lastHeedRound = uint32_t(0xFFFFFFFF)]() mutable {
        const double now  = Simulator::Now().GetSeconds();
        auto& nodes = *g_ctx.pSt;
        auto& rng   = *g_ctx.pRng;

        // 1. Synchroniser les morts
        for (auto& n : nodes)
            if (g_dead.count(n.id)) n.isAlive = false;

        // 2. Phase SETUP HEED — uniquement chaque RECLUSTER_PERIOD (100s)
        //  Reclustering toutes les 100s comme LEACH, pas à chaque step de 5s.
        // Sans ce garde, la topologie changeait 20× par round → isolés erratiques,
        // overhead prohibitif (O(N²) par step) et résultats non représentatifs.
        const uint32_t heedRound = static_cast<uint32_t>(
            now / EvalCfg::RECLUSTER_PERIOD);
        if (heedRound != lastHeedRound) {
            lastHeedRound = heedRound;
            RunHeedSetup(nodes, rng);
        }

        // 3. Pré-calcul énergie après drain (même logique que LEACH)
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
                // Nœud isolé : veille idle [FIX-1] — pas de TX vers sink dans HEED
                drain = EvalDrainIsolated(ns.distToSink);
            }
            energyAfter[ns.id] = ns.energy - std::max(0.0, drain);
        }

        // 4. PDR + drain (même logique que LEACH)
        for (auto& ns : nodes) {
            if (!ns.isAlive) continue;

            double drain     = 0.0;
            bool   delivered = false;

            if (ns.isClusterHead) {
                uint32_t nMem = 0;
                for (const auto& m : nodes)
                    if (m.isAlive && !m.isClusterHead && m.clusterId == ns.id)
                        nMem++;
                drain = EvalDrainCH(nMem, ns.distToSink);
                delivered = (energyAfter.count(ns.id) &&
                             energyAfter[ns.id] > 0.0);

            } else if (ns.clusterId == 0) {
                // Nœud isolé : veille idle [FIX-1] — paquet perdu
                drain     = EvalDrainIsolated(ns.distToSink);
                delivered = false;

            } else {
                double dToCH = EvalCfg::RADIO_RANGE;
                const EvalNodeState* chPtr = nullptr;
                for (const auto& ch : nodes)
                    if (ch.id == ns.clusterId && ch.isAlive) {
                        dToCH = NodeDist(ns, ch); chPtr = &ch; break;
                    }
                drain = EvalDrainMember(dToCH);
                const bool chInRange  = (chPtr != nullptr) &&
                                        (dToCH <= EvalCfg::RADIO_RANGE);
                const bool chSurvives = chInRange &&
                    energyAfter.count(chPtr->id) &&
                    energyAfter[chPtr->id] > 0.0;
                delivered = chInRange && chSurvives;
            }

            g_ctx.rlPktEmitted++;
            g_ctx.roundPktEmitted++;
            if (delivered) {
                g_ctx.rlPktDelivered++;
                g_ctx.roundPktDelivered++;
            }

            const bool wasAlive = ns.isAlive;
            if (!ns.Consume(drain) && wasAlive && !g_dead.count(ns.id)) {
                g_dead.insert(ns.id);

                if (!g_fndDone) {
                    if (!g_ctx.preFND_locked) {
                        g_ctx.preFND_emitted  = g_ctx.rlPktEmitted;
                        g_ctx.preFND_delivered= g_ctx.rlPktDelivered;
                        g_ctx.preFND_locked   = true;
                    }
                    g_pdrStable = g_ctx.preFND_emitted > 0
                        ? 100.0 * g_ctx.preFND_delivered / g_ctx.preFND_emitted
                        : 100.0;
                    g_fndDone = true; g_fndTime = now;
                    g_summary.fnd_s = now;
                    NS_LOG_UNCOND("⭐ [FND] t=" << now
                        << "s | PDR_stable=" << std::fixed
                        << std::setprecision(1) << g_pdrStable << "%");
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
            Simulator::Schedule(Seconds(EvalCfg::RL_STEP_INTERVAL), heedStep);
    };
}

// ─── Métriques ───────────────────────────────────────────────────────────────

static void InitDoMetrics() {
    doMetrics = [&](int round) {
        const double now  = Simulator::Now().GetSeconds();
        auto& nodes = *g_ctx.pSt;
        for (auto& n : nodes) if (g_dead.count(n.id)) n.isAlive = false;

        uint32_t nCH = 0, nAlive = 0, nIsolated = 0;
        for (const auto& n : nodes) {
            if (!n.isAlive) continue;
            nAlive++;
            if (n.isClusterHead) nCH++;
            if (!n.isClusterHead && n.clusterId == 0) nIsolated++;
        }

        const double pdrRawRL = g_ctx.roundPktEmitted > 0
            ? 100.0 * g_ctx.roundPktDelivered / g_ctx.roundPktEmitted : 100.0;
        g_ctx.pdrWindow.push_back(pdrRawRL);
        if (g_ctx.pdrWindow.size() > g_ctx.PDR_SMOOTH)
            g_ctx.pdrWindow.pop_front();
        double pdrSum = 0.0;
        for (double v : g_ctx.pdrWindow) pdrSum += v;
        const double pdrRoundRL = pdrSum / g_ctx.pdrWindow.size();
        const double pdrRL = g_ctx.rlPktEmitted > 0
            ? 100.0 * g_ctx.rlPktDelivered / g_ctx.rlPktEmitted : 100.0;
        g_ctx.roundPktEmitted  = 0;
        g_ctx.roundPktDelivered= 0;

        const double delay_ms = ComputeHeedLogicalDelay_ms(nodes,
                                    EvalCfg::RADIO_RANGE);

        // PDR NS-3 calculé uniquement en post-simulation (voir main)

        RoundMetrics m = ComputeRoundMetrics(
            round, now, nodes, g_ctx.initEnergy,
            g_ctx.rlPktEmitted, g_ctx.rlPktDelivered, delay_ms, nCH);
        m.pdr_pct    = pdrRoundRL;
        m.isolatedNodes = nIsolated;

        g_history.push_back(m);
        if (g_ctx.pMetricsCSV && g_ctx.pMetricsCSV->is_open())
            WriteMetricsRow(*g_ctx.pMetricsCSV, m);

        NS_LOG_UNCOND(std::fixed
            << "[HEED Round " << round
            << "] t=" << std::setprecision(0) << now << "s"
            << "|vivants=" << nAlive
            << "|morts="   << (g_ctx.nNodes - nAlive)
            << "|isolés="  << nIsolated
            << "|E_moy="   << std::setprecision(3) << m.energyMean_J << "J"
            << "|PDR_RL="  << std::setprecision(1) << pdrRL << "%"
            << "|delay="   << delay_ms << "ms"
            << "|CH="      << nCH);

        if (now + EvalCfg::METRICS_INTERVAL <= g_ctx.simDuration)
            Simulator::Schedule(Seconds(EvalCfg::METRICS_INTERVAL),
                                [r = round + 1]() { doMetrics(r); });
    };
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string resultsDir = "results_eval/HEED";
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

    NS_LOG_UNCOND("╔══════════════════════════════════╗");
    NS_LOG_UNCOND("║  HEED Baseline — Démarrage       ║");
    NS_LOG_UNCOND("╚══════════════════════════════════╝");
    NS_LOG_UNCOND("  N=" << nNodes
        << " E=" << EvalCfg::E_INIT << "J"
        << " Seed=" << seed
        << " Portée=" << EvalCfg::RADIO_RANGE << "m"
        << " Zone=" << (int)EvalCfg::AREA_SIZE << "m"
        << " C_prob=" << HeedCfg::C_PROB);

    // ── NS-3 setup (identique à leach_sim.cc) ────────────────────────────────
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
    g_ctx.pIdIdx      = &idToIdx;
    g_ctx.pRng        = &rng;
    g_ctx.pFM         = fm;
    g_ctx.resultsDir  = resultsDir;
    g_summary.modelName = "HEED";

    std::ofstream metricsCSV;
    InitMetricsCSV(metricsCSV, resultsDir + "/metrics.csv", "HEED");
    g_ctx.pMetricsCSV = &metricsCSV;

    InitHeedStep();
    InitDoMetrics();
    Simulator::Schedule(Seconds(2.0), heedStep);
    Simulator::Schedule(Seconds(EvalCfg::METRICS_INTERVAL),
                        []() { doMetrics(1); });
    Simulator::Stop(Seconds(EvalCfg::SIM_DURATION));

    NS_LOG_UNCOND("▶ Simulation HEED démarrée (" << EvalCfg::SIM_DURATION << "s)...");
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
    g_summary.avgDelay_ms   = ComputeHeedLogicalDelay_ms(states,
                                  EvalCfg::RADIO_RANGE);
    g_summary.totalEnergy_J = nNodes * EvalCfg::E_INIT - sumE;
    g_summary.totalPktSent  = g_ctx.rlPktEmitted;
    g_summary.totalPktRecv  = g_ctx.rlPktDelivered;
    g_summary.totalRounds   = static_cast<uint32_t>(g_history.size());

    WriteSummaryCSV(resultsDir + "/summary.csv", g_summary);

    NS_LOG_UNCOND("\n╔═══════════════════════════════════════╗");
    NS_LOG_UNCOND(  "║  RÉSULTATS HEED                        ║");
    NS_LOG_UNCOND(  "╠═══════════════════════════════════════╣");
    NS_LOG_UNCOND(  "║ Vivants     : " << alive << "/" << nNodes);
    NS_LOG_UNCOND(  "║ FND         : "
        << (g_fndDone ? std::to_string((int)g_fndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ HND         : "
        << (g_hndDone ? std::to_string((int)g_hndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ LND (90%)   : "
        << (g_lndDone ? std::to_string((int)g_lndTime) + "s" : "—"));
    NS_LOG_UNCOND(  "║ PDR stable  : "
        << std::fixed << std::setprecision(1) << g_summary.pdrStable_pct << "%");
    NS_LOG_UNCOND(  "║ PDR global  : " << g_summary.pdrGlobal_pct << "%");
    NS_LOG_UNCOND(  "║ PDR NS-3    : " << pdrNS3 << "%");
    NS_LOG_UNCOND(  "║ Énergie cons: "
        << std::setprecision(3) << g_summary.totalEnergy_J << " J");
    NS_LOG_UNCOND(  "╚═══════════════════════════════════════╝");

    auto t1 = std::chrono::high_resolution_clock::now();
    NS_LOG_UNCOND("⏱ " << std::chrono::duration_cast<std::chrono::seconds>(
        t1 - t0).count() / 60 << " mn");

    Simulator::Destroy();
    return 0;
}
