/* =============================================================================
 * qrouting_sim.cc — Q-Routing Standard (Q-learning tabulaire)
 *
 * Q-Routing classique (Boyan & Littman, 1994) adapté aux WSN :
 *   - Table Q(s, a) où s = nœud source, a = prochain saut
 *   - Pas de deep learning, pas de PEPM, pas de fédération
 *   - Clustering IFO identique à FDQN-TE+ (pour comparaison juste)
 *   - Récompense : identique à FDQN-TE+ (même lambdas)
 *
 * Usage :
 *   ./ns3 run "scratch/qrouting_sim --resultsDir=results/QRouting"
 * ============================================================================= */

#include "eval_config.h"
#include "eval_common.h"

// Réutiliser les modules partagés de FDQN-TE+
#include "fdqn_config.h"
#include "leach_energy.h"
#include "node_state.h"
#include "ifo_clustering.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/energy-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/olsr-module.h"

#include <map>
#include <set>
#include <random>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <filesystem>
#include <functional>

using namespace ns3;
namespace fs = std::filesystem;
NS_LOG_COMPONENT_DEFINE("QRoutingSim");

// ─── Q-Agent tabulaire ────────────────────────────────────────────────────────

class QAgent {
public:
    QAgent() : m_epsilon(EvalCfg::QROUTE_EPS_MAX) {}

    // ε-greedy : sélectionner le prochain saut
    uint32_t SelectAction(uint32_t nodeId,
                          const std::vector<uint32_t>& neighbors,
                          std::mt19937& rng) {
        if (neighbors.empty()) return UINT32_MAX;
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        if (ud(rng) < m_epsilon) {
            std::uniform_int_distribution<size_t> ri(0, neighbors.size()-1);
            return neighbors[ri(rng)];
        }
        uint32_t best = neighbors[0];
        double   bestQ = m_qtable[{nodeId, neighbors[0]}];
        for (size_t i = 1; i < neighbors.size(); i++) {
            double q = m_qtable[{nodeId, neighbors[i]}];
            if (q > bestQ) { bestQ = q; best = neighbors[i]; }
        }
        return best;
    }

    // Mise à jour Q-learning classique
    void Update(uint32_t nodeId, uint32_t nextHop, double reward,
                const std::vector<uint32_t>& nextNeighbors) {
        double maxQ = 0.0;
        for (uint32_t nb : nextNeighbors) {
            double q = m_qtable[{nextHop, nb}];
            if (q > maxQ) maxQ = q;
        }
        auto key = std::make_pair(nodeId, nextHop);
        m_qtable[key] += EvalCfg::QROUTE_LR
                      * (reward + EvalCfg::QROUTE_GAMMA * maxQ - m_qtable[key]);
        m_epsilon = std::max(EvalCfg::QROUTE_EPS_MIN,
                             m_epsilon * EvalCfg::QROUTE_EPS_DECAY);
    }

private:
    std::map<std::pair<uint32_t,uint32_t>, double> m_qtable;
    double m_epsilon;
};

// =============================================================================
// ComputeQRoutingLogicalDelay_ms — Délai bout-en-bout logique Q-Routing
//
// Remplace EvalDelay::ComputeDelay_ms(2, -1.0, alive) — valeur statique basée
// sur un modèle 802.15.4 250kbps incohérent avec la PHY 802.11b 1Mbps.
//
// MODÈLE PHY UNIFIÉ (identique à LEACH et toutes variantes DQN) :
//   PHY  : IEEE 802.11b DSSS @ 1 Mbps (DsssRate1Mbps — configuré dans main())
//   PKT  : 500 octets (PacketSize dans OnOffApplication)
//   MAC  : overhead = 0.832 ms (SIFS+ACK+DIFS+backoff moyen, mesuré NS-3 802.11b)
//   → delayPerHop = (500×8/1e6)×1000 + 0.832 = 4.0 + 0.832 = 4.832 ms/saut
//
// TOPOLOGIE Q-Routing (2 niveaux hiérarchiques, multi-hop réel) :
//   - Membre → CH  : ceil(distMembre-CH / radioRange) sauts
//   - CH → sink    : ceil(distCH-sink   / radioRange) sauts (multi-hop contraint)
//   - Total membre : hops1 + hops2 sauts bout-en-bout
//   - CH sans membre : compte seul avec son CH→sink direct
//
// Différence avec LEACH : LEACH suppose une BS à longue portée (CH→sink = 1 hop fixe).
// Q-Routing opère en multi-hop pur : chaque saut est contraint par RADIO_RANGE.
// Ce comportement distinct est fidèle aux protocoles respectifs et comparable
// puisque la PHY sous-jacente (802.11b 1Mbps) est identique dans toutes les simulations.
// =============================================================================
static double ComputeQRoutingLogicalDelay_ms(
    const std::vector<NodeState>& nodes,
    double radioRange)
{
    static constexpr double PKT_BYTES       = 500.0;
    static constexpr double PHY_RATE_BPS    = 1.0e6;   // 802.11b DSSS 1Mbps
    static constexpr double MAC_OVERHEAD_MS = 0.832;

    const double txTime_ms      = (PKT_BYTES * 8.0 / PHY_RATE_BPS) * 1000.0;
    const double delayPerHop_ms = txTime_ms + MAC_OVERHEAD_MS;  // 4.832 ms/saut

    double   sumDelay = 0.0;
    uint32_t count    = 0;

    for (const auto& ns : nodes) {
        if (!ns.isAlive) continue;

        if (ns.isClusterHead) {
            // CH → sink : multi-sauts selon distance réelle
            const double hops = (ns.distToSink <= radioRange)
                ? 1.0
                : std::ceil(ns.distToSink / radioRange);
            sumDelay += hops * delayPerHop_ms;
            count++;
        } else {
            // Membre → CH → sink
            bool chFound = false;
            for (const auto& nb : nodes) {
                if (nb.id == ns.clusterId && nb.isAlive) {
                    const double distToCH   = NodeDist(ns.x, ns.y, nb.x, nb.y);
                    const double hops1 = (distToCH <= radioRange)
                        ? 1.0 : std::ceil(distToCH / radioRange);
                    const double hops2 = (nb.distToSink <= radioRange)
                        ? 1.0 : std::ceil(nb.distToSink / radioRange);
                    sumDelay += (hops1 + hops2) * delayPerHop_ms;
                    count++;
                    chFound = true;
                    break;
                }
            }
            if (!chFound) {
                // CH mort — chemin direct fallback
                const double hops = (ns.distToSink <= radioRange)
                    ? 1.0 : std::ceil(ns.distToSink / radioRange);
                sumDelay += hops * delayPerHop_ms;
                count++;
            }
        }
    }

    return count > 0 ? sumDelay / count : 0.0;
}

// ─── Globals ──────────────────────────────────────────────────────────────────

static std::set<uint32_t> g_dead;
static bool   g_fndDone=false, g_hndDone=false, g_lndDone=false;
static double g_fndTime=0.0,   g_hndTime=0.0,   g_lndTime=0.0;
static double g_pdrStable = -1.0;

struct QRoutingContext {
    uint32_t nNodes;
    double   initEnergy, sinkX, sinkY, simDuration, radioRange;
    std::vector<NodeState>*          pSt;
    IFOClustering*                   pIfo;
    std::map<uint32_t, uint32_t>*    pIdIdx;
    std::map<uint32_t, QAgent>*      pAgents;
    std::mt19937*                    pRng;
    std::ofstream*                   pMetricsCSV;
    std::string resultsDir;
    uint64_t pktEmitted=0, pktDelivered=0;
    Ptr<FlowMonitor> pFM;
    double lastPDR_NS3=100.0;
};

static QRoutingContext g_ctx;
static std::function<void()> rlStep;
static std::function<void(int)> doMetrics;
static ModelSummary g_summary;
static std::vector<RoundMetrics> g_history;

// ─── Reward computation — ALIGNÉ sur fdqn_te_plus_eval.cc (FIX-COLLAPSE) ─────
//
// DIVERGENCES ORIGINALES vs FDQN-TE+ (rendaient la comparaison injuste) :
//   1. eNorm = ns.NormEnergy()    → énergie SELF, pas next-hop  (même bug que FDQN)
//   2. pdrSignal = -2.0 si mort   → asymétrie non justifiée vs -1.0 dans FDQN
//   3. LAMBDA_DELAY absent        → Q-Routing ne pénalisait pas la distance
//
// CORRECTIONS (identiques à fdqn_te_plus_eval.cc) :
//   1. eSignal basé sur nhState.NormEnergy() + pénalité quadratique sous E_CRIT=25%
//      → Q-Agent apprend à éviter les next-hops épuisés, comme FDQN-TE+
//   2. pdrSignal = -1.0 (symétrique)
//   3. LAMBDA_DELAY réintroduit (EvalCfg::LAMBDA_DELAY × dist/dMax)
//
double ComputeReward(const NodeState& ns, const NodeState& nhState,
                     uint32_t nextHop, double radioRange) {

    // 1. PDR signal — identique FDQN-TE+
    const double pdrSignal = nhState.isAlive ? 1.0 : -1.0;

    // 2. Énergie next-hop avec pénalité quadratique sous E_CRIT — identique FDQN-TE+
    const double eNh = nhState.NormEnergy();
    constexpr double E_CRIT_R = 0.25;
    const double eSignal = (eNh >= E_CRIT_R)
        ?  eNh
        : -(1.0 - eNh / E_CRIT_R) * (1.0 - eNh / E_CRIT_R);

    // 3. Délai (distance normalisée) — identique FDQN-TE+
    const double dMax      = radioRange * 10.0;
    const double delayTerm = std::min(1.0, nhState.distToSink / dMax);

    // 4. hierBonus — même logique que FDQN-TE+
    double hierBonus = 0.0;
    if (!ns.isClusterHead) {
        if (nextHop == ns.clusterId) hierBonus = 1.0;
        else {
            if (nhState.clusterId == ns.clusterId && !nhState.isClusterHead)
                hierBonus = 0.5;
            else hierBonus = -1.0;
        }
    } else {
        const double dS = ns.distToSink, dN = nhState.distToSink;
        hierBonus = dS > 1.0
            ? std::max(-1.0, std::min(1.0, (dS - dN) / dS)) : 1.0;
    }

    return EvalCfg::LAMBDA_PDR    * pdrSignal
         + EvalCfg::LAMBDA_ENERGY * eSignal
         - EvalCfg::LAMBDA_DELAY  * delayTerm
         + EvalCfg::LAMBDA_SAFE   * (1.0 - ns.pepmRisk)
         + EvalCfg::LAMBDA_HIER   * hierBonus;
}

// ─── RL Step ──────────────────────────────────────────────────────────────────

static void InitRLStep() {
    rlStep = [&]() {
        const double now = Simulator::Now().GetSeconds();
        auto& nodes   = *g_ctx.pSt;
        auto& agents  = *g_ctx.pAgents;
        auto& idIdx   = *g_ctx.pIdIdx;

        for (auto& n : nodes) if (g_dead.count(n.id)) n.isAlive = false;

        for (uint32_t i = 0; i < g_ctx.nNodes; i++) {
            NodeState& ns = nodes[i];
            if (!ns.isAlive) continue;

            // Construire la liste des voisins (identique FDQN-TE+)
            std::vector<uint32_t> neighbors;
            if (ns.isClusterHead) {
                for (uint32_t j = 0; j < g_ctx.nNodes; j++) {
                    if (i==j) continue;
                    const NodeState& nb = nodes[j];
                    if (!nb.isAlive || !nb.isClusterHead) continue;
                    if (NodeDist(ns.x,ns.y,nb.x,nb.y) <= g_ctx.radioRange)
                        neighbors.push_back(nb.id);
                }
                if (neighbors.empty())
                    for (uint32_t j=0; j<g_ctx.nNodes; j++) {
                        if (i==j) continue;
                        const NodeState& nb = nodes[j];
                        if (nb.isAlive && NodeDist(ns.x,ns.y,nb.x,nb.y)<=g_ctx.radioRange)
                            neighbors.push_back(nb.id);
                    }
            } else {
                const NodeState* chPtr = nullptr;
                for (auto& n2 : nodes) if (n2.id==ns.clusterId&&n2.isAlive){chPtr=&n2;break;}
                if (chPtr && NodeDist(ns.x,ns.y,chPtr->x,chPtr->y)<=g_ctx.radioRange) {
                    neighbors.push_back(ns.clusterId);
                } else {
                    struct AltCH { uint32_t id; double eN; };
                    std::vector<AltCH> altCHs;
                    for (auto& n2 : nodes)
                        if (n2.isAlive && n2.isClusterHead &&
                            NodeDist(ns.x,ns.y,n2.x,n2.y)<=g_ctx.radioRange)
                            altCHs.push_back({n2.id, n2.NormEnergy()});
                    std::sort(altCHs.begin(),altCHs.end(),
                              [](auto& a,auto& b){return a.eN>b.eN;});
                    for (size_t k=0;k<std::min((size_t)2,altCHs.size());k++)
                        neighbors.push_back(altCHs[k].id);
                }
            }
            if (neighbors.empty()) continue;

            // Sélection Q-learning
            uint32_t nextHop = agents[ns.id].SelectAction(ns.id, neighbors, *g_ctx.pRng);
            if (nextHop == UINT32_MAX) continue;
            const NodeState& nhState = nodes[idIdx[nextHop]];

            // Drain énergétique
            double drain;
            if (ns.isClusterHead) {
                uint32_t nMem = 0;
                for (const auto& ci : g_ctx.pIfo->GetClusters())
                    if (ci.chId == ns.id) { nMem = ci.members.size(); break; }
                drain = LeachCHRound(nMem, ns.distToSink, FdqnCfg::DRAIN_BITS);
            } else {
                drain = LeachMemberRound(NodeDist(ns,nhState), FdqnCfg::DRAIN_BITS);
                // Drain RX sur le récepteur
                NodeState& relay = nodes[idIdx[nextHop]];
                if (relay.isAlive) relay.Consume(FdqnCfg::E_ELEC * FdqnCfg::DRAIN_BITS);
            }
            bool wasAlive = ns.isAlive;
            ns.Consume(drain);
            if (!ns.isAlive && wasAlive && !g_dead.count(ns.id)) {
                g_dead.insert(ns.id);
                if (!g_fndDone) {
                    g_pdrStable = g_ctx.pktEmitted > 0
                        ? 100.0*(double)g_ctx.pktDelivered/g_ctx.pktEmitted : 100.0;
                    g_fndDone=true; g_fndTime=now; g_summary.fnd_s=now;
                    NS_LOG_UNCOND("⭐ [FND] t="<<now<<"s | PDR_stable="
                        <<std::fixed<<std::setprecision(1)<<g_pdrStable<<"%");
                }
                if (!g_hndDone && g_dead.size()>=g_ctx.nNodes/2) {
                    g_hndDone=true; g_hndTime=now; g_summary.hnd_s=now;
                    NS_LOG_UNCOND("⭐ [HND] t="<<now<<"s");
                }
                if (!g_lndDone && g_dead.size()>=(uint32_t)(g_ctx.nNodes*0.9)) {
                    g_lndDone=true; g_lndTime=now; g_summary.lnd_s=now;
                    NS_LOG_UNCOND("⭐ [LND-90%] t="<<now<<"s");
                    Simulator::Stop();
                }
            }

            // PDR — règles de delivery cohérentes avec la littérature Q-Routing WSN
            // Référence : Boyan & Littman 1994 + adaptations WSN (Forster & Murphy 2007)
            g_ctx.pktEmitted++;
            bool delivered = false;
            if (ns.isAlive && nhState.isAlive) {
                if (ns.isClusterHead) {
                    // CH → prochain CH plus proche du sink → livraison si saut valide
                    // (progression vers sink, Forster & Murphy 2007, Section III)
                    delivered = (nhState.distToSink < ns.distToSink);
                } else {
                    // Membre → son CH : livraison si CH vivant après drain de ce step
                    bool chAlive = false;
                    for (auto& n2 : nodes)
                        if (n2.id == ns.clusterId && n2.isAlive) { chAlive = true; break; }
                    delivered = chAlive;
                }
            }
            if (delivered) g_ctx.pktDelivered++;

            // Mise à jour Q
            double reward = ComputeReward(ns, nhState, nextHop, g_ctx.radioRange);
            agents[ns.id].Update(ns.id, nextHop, reward, neighbors);
            ns.totalReward += reward;
            ns.txCount++;
        }

        if (now + FdqnCfg::RL_STEP_INTERVAL <= g_ctx.simDuration)
            Simulator::Schedule(Seconds(FdqnCfg::RL_STEP_INTERVAL), rlStep);
    };
}

static void InitDoMetrics() {
    doMetrics = [&](int round) {
        const double now = Simulator::Now().GetSeconds();
        for (auto& n : *g_ctx.pSt) if (g_dead.count(n.id)) n.isAlive=false;

        // Re-clustering IFO
        if (round % 2 == 0) {
            uint32_t nc = g_ctx.pIfo->ComputeNClusters(*g_ctx.pSt);
            g_ctx.pIfo->Run(*g_ctx.pSt, nc);
        }

        auto& nodes = *g_ctx.pSt;
        uint32_t alive=0, nCH=0; double sumE=0.0, drained=0.0;
        for (const auto& n : nodes) {
            drained += (g_ctx.initEnergy - n.energy);
            if (!n.isAlive) continue;
            alive++; sumE += n.energy;
            if (n.isClusterHead) nCH++;
        }

        double pdr = g_ctx.pktEmitted>0 ? 100.0*g_ctx.pktDelivered/g_ctx.pktEmitted : 100.0;

        // Convertir NodeState -> EvalNodeState pour ComputeRoundMetrics
        std::vector<EvalNodeState> evalStates;
        for (const auto& n : nodes) {
            EvalNodeState e(n.id, n.x, n.y, n.energy, n.distToSink);
            e.isAlive=n.isAlive; e.pepmRisk=n.pepmRisk;
            evalStates.push_back(e);
        }
        // Délai Q-Routing logique bout-en-bout
        // Remplace EvalDelay::ComputeDelay_ms(2,-1.0,alive) — valeur statique
        // basée sur 802.15.4 250kbps, incohérente avec la PHY 802.11b 1Mbps.
        // ComputeQRoutingLogicalDelay_ms() calcule depuis la topologie IFO réelle :
        //   membres (hops_membre_CH + hops_CH_sink), CH (hops_CH_sink),
        //   avec delayPerHop = 4.832ms (802.11b 1Mbps, 500 octets).
        const double delay_ms_round = ComputeQRoutingLogicalDelay_ms(nodes, g_ctx.radioRange);
        RoundMetrics m = ComputeRoundMetrics(round, now, evalStates,
            g_ctx.initEnergy, g_ctx.pktEmitted, g_ctx.pktDelivered, delay_ms_round, nCH);
        g_history.push_back(m);
        if (g_ctx.pMetricsCSV && g_ctx.pMetricsCSV->is_open())
            WriteMetricsRow(*g_ctx.pMetricsCSV, m);

        NS_LOG_UNCOND(std::fixed
            << "[QRouting R" << round << "] t=" << std::setprecision(0) << now << "s"
            << "|vivants=" << m.aliveNodes
            << "|E_moy=" << std::setprecision(3) << m.energyMean_J << "J"
            << "|PDR=" << std::setprecision(1) << pdr << "%"
            << "|Delay=" << std::setprecision(2) << delay_ms_round << " ms");

        if (now + FdqnCfg::METRICS_INTERVAL <= g_ctx.simDuration)
            Simulator::Schedule(Seconds(FdqnCfg::METRICS_INTERVAL),
                                [r=round+1](){ doMetrics(r); });
    };
}

int main(int argc, char* argv[]) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string resultsDir = "results/QRouting";
    uint32_t seed = EvalCfg::SEED;
    uint32_t    nNodes     = 300;
    CommandLine cmd;
    cmd.AddValue("resultsDir","Dossier",resultsDir);
    cmd.AddValue("seed","Seed",seed);
    cmd.AddValue("nNodes",     "Nombre de nœuds",   nNodes);
    cmd.Parse(argc, argv);
    fs::create_directories(resultsDir);

    RngSeedManager::SetSeed(seed); RngSeedManager::SetRun(1);
    std::mt19937 rng(seed);

    NS_LOG_UNCOND("╔═══════════════════════════════════╗");
    NS_LOG_UNCOND("║  Q-Routing Standard — Démarrage   ║");
    NS_LOG_UNCOND("╚═══════════════════════════════════╝");

    NodeContainer sensors, sinkNode;
    sensors.Create(nNodes); sinkNode.Create(1);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max="+std::to_string(EvalCfg::AREA_SIZE)+"]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max="+std::to_string(EvalCfg::AREA_SIZE)+"]"));
    mob.Install(sensors);
    mob.SetPositionAllocator("ns3::GridPositionAllocator",
        "MinX", DoubleValue(EvalCfg::SINK_X), "MinY", DoubleValue(EvalCfg::SINK_Y));
    mob.Install(sinkNode);
    WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",StringValue("DsssRate1Mbps"),"ControlMode",StringValue("DsssRate1Mbps"));
    WifiMacHelper mac; mac.SetType("ns3::AdhocWifiMac");
    YansWifiPhyHelper phy;
    YansWifiChannelHelper wch=YansWifiChannelHelper::Default();
    wch.AddPropagationLoss("ns3::RangePropagationLossModel",
        "MaxRange",DoubleValue(EvalCfg::RADIO_RANGE));
    phy.SetChannel(wch.Create());
    NodeContainer all; all.Add(sensors); all.Add(sinkNode);
    NetDeviceContainer devs=wifi.Install(phy,mac,all);
    OlsrHelper olsr; InternetStackHelper inet;
    inet.SetRoutingHelper(olsr); inet.Install(all);
    Ipv4AddressHelper ip; ip.SetBase("10.0.0.0","255.255.0.0");
    Ipv4InterfaceContainer ifaces=ip.Assign(devs);

    // États FDQN (réutilise NodeState de fdqn_te_plus)
    std::vector<NodeState> states(nNodes);
    std::map<uint32_t,uint32_t> idToIdx;
    for (uint32_t i=0; i<nNodes; i++) {
        auto mob2=sensors.Get(i)->GetObject<MobilityModel>();
        Vector v=mob2->GetPosition();
        uint32_t nid=sensors.Get(i)->GetId();
        double dSink=EvalDist(v.x,v.y,EvalCfg::SINK_X,EvalCfg::SINK_Y);
        states[i]=NodeState(nid,v.x,v.y,EvalCfg::E_INIT,dSink);
        idToIdx[nid]=i;
    }
    IFOClustering ifo;
    ifo.SetArea(EvalCfg::SINK_X,EvalCfg::SINK_Y,EvalCfg::AREA_SIZE,
                EvalCfg::RADIO_RANGE,EvalCfg::E_INIT,FdqnCfg::IFO_ITER);
    uint32_t nc0=ifo.ComputeNClusters(states); ifo.Run(states,nc0);

    std::map<uint32_t,QAgent> agents;
    for (auto& n:states) agents[n.id]=QAgent();

    uint16_t port=9;
    Address sinkAddr(InetSocketAddress(ifaces.GetAddress(nNodes),port));
    PacketSinkHelper sinkApp("ns3::UdpSocketFactory",sinkAddr);
    ApplicationContainer sApps=sinkApp.Install(sinkNode);
    sApps.Start(Seconds(1.0)); sApps.Stop(Seconds(EvalCfg::SIM_DURATION));
    for (uint32_t i=0;i<nNodes;i++) {
        OnOffHelper s("ns3::UdpSocketFactory",sinkAddr);
        s.SetAttribute("PacketSize",UintegerValue(EvalCfg::PKT_BITS/8));
        s.SetAttribute("DataRate",StringValue("10kbps"));
        s.SetAttribute("OnTime",StringValue("ns3::ConstantRandomVariable[Constant=2]"));
        s.SetAttribute("OffTime",StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
        ApplicationContainer a=s.Install(sensors.Get(i));
        a.Start(Seconds(15.0+i*0.01)); a.Stop(Seconds(EvalCfg::SIM_DURATION));
    }
    FlowMonitorHelper fmH; Ptr<FlowMonitor> fm=fmH.InstallAll();

    g_ctx.nNodes=nNodes; g_ctx.initEnergy=EvalCfg::E_INIT;
    g_ctx.sinkX=EvalCfg::SINK_X; g_ctx.sinkY=EvalCfg::SINK_Y;
    g_ctx.simDuration=EvalCfg::SIM_DURATION; g_ctx.radioRange=EvalCfg::RADIO_RANGE;
    g_ctx.pSt=&states; g_ctx.pIfo=&ifo; g_ctx.pIdIdx=&idToIdx;
    g_ctx.pAgents=&agents; g_ctx.pRng=&rng; g_ctx.pFM=fm;
    g_ctx.resultsDir=resultsDir; g_summary.modelName=EvalCfg::MODEL_QROUTING;

    std::ofstream metricsCSV;
    InitMetricsCSV(metricsCSV,resultsDir+"/metrics.csv",EvalCfg::MODEL_QROUTING);
    g_ctx.pMetricsCSV=&metricsCSV;

    InitRLStep(); InitDoMetrics();
    Simulator::Schedule(Seconds(2.0),rlStep);
    Simulator::Schedule(Seconds(FdqnCfg::METRICS_INTERVAL),[](){ doMetrics(1); });
    Simulator::Stop(Seconds(EvalCfg::SIM_DURATION));
    NS_LOG_UNCOND("▶ Simulation Q-Routing démarrée...");
    Simulator::Run();

    // Collecter les stats FlowMonitor NS-3
    fm->CheckForLostPackets();
    uint64_t txPkt=0, rxPkt=0;
    for (const auto& kv : fm->GetFlowStats()) {
        txPkt += kv.second.txPackets;
        rxPkt += kv.second.rxPackets;
    }

    uint32_t alive=0; double sumE=0.0;
    for (const auto& n:states) if(n.isAlive){alive++;sumE+=n.energy;}
    double pdrRL=g_ctx.pktEmitted>0?100.0*(double)g_ctx.pktDelivered/g_ctx.pktEmitted:100.0;
    g_summary.pdrStable_pct=(g_pdrStable>=0)?g_pdrStable:pdrRL;
    g_summary.pdrGlobal_pct=pdrRL;
    g_summary.avgDelay_ms=ComputeQRoutingLogicalDelay_ms(states, g_ctx.radioRange);
    g_summary.totalEnergy_J=nNodes*EvalCfg::E_INIT-sumE;
    g_summary.totalPktSent=g_ctx.pktEmitted;   // RL logique
    g_summary.totalPktRecv=g_ctx.pktDelivered;
    g_summary.totalRounds=(uint32_t)g_history.size();
    WriteSummaryCSV(resultsDir+"/summary.csv",g_summary);

    NS_LOG_UNCOND("\n╔════════════════════════════════════════╗");
    NS_LOG_UNCOND(  "║  RÉSULTATS Q-ROUTING                   ║");
    NS_LOG_UNCOND(  "╠════════════════════════════════════════╣");
    NS_LOG_UNCOND(  "║ Vivants     : " << alive << "/" << nNodes);
    NS_LOG_UNCOND(  "║ FND         : "
        << (g_fndDone ? std::to_string((int)g_summary.fnd_s) + "s" : "—"));
    NS_LOG_UNCOND(  "║ HND         : "
        << (g_hndDone ? std::to_string((int)g_summary.hnd_s) + "s" : "—"));
    NS_LOG_UNCOND(  "║ LND (90%)   : "
        << (g_lndDone ? std::to_string((int)g_summary.lnd_s) + "s" : "—"));
    NS_LOG_UNCOND(  "║ PDR stable  (pré-FND) : "
        << std::fixed << std::setprecision(1) << g_summary.pdrStable_pct << "%");
    NS_LOG_UNCOND(  "║ PDR global  (logique) : "
        << g_summary.pdrGlobal_pct << "%");
    NS_LOG_UNCOND(  "║ E consommée : "
        << std::setprecision(3) << g_summary.totalEnergy_J << " J");
    NS_LOG_UNCOND(  "║ Delay moyen : "
    << std::setprecision(2) << g_summary.avgDelay_ms << " ms");
    NS_LOG_UNCOND(  "╚════════════════════════════════════════╝");
    Simulator::Destroy();
    auto t1=std::chrono::high_resolution_clock::now();
    NS_LOG_UNCOND("⏱ "<<std::chrono::duration_cast<std::chrono::minutes>(t1-t0).count()<<" mn");
    return 0;
}
