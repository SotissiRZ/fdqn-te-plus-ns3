/* =============================================================================
 * eval_common.h — Structures et fonctions partagées par tous les modèles
 * ============================================================================= */

#ifndef EVAL_COMMON_H
#define EVAL_COMMON_H

#include "eval_config.h"
#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Utilitaire géométrique
// ─────────────────────────────────────────────────────────────────────────────

inline double EvalDist(double x1, double y1, double x2, double y2) {
    return std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Modèle de canal réaliste (log-distance + shadowing gaussien)
// Calibré pour IEEE 802.15.4 / ZigBee indoor-outdoor mixed environment
// Référence : Zuniga & Krishnamachari, SECON 2004 (modèle WSN standard)
// ─────────────────────────────────────────────────────────────────────────────

namespace EvalChannel {
    // Paramètres canal IEEE 802.15.4 @ 2.4GHz
    constexpr double PTX_DBM         = 0.0;     // dBm — puissance TX typique Zigbee
    constexpr double PL_D0_DB        = 55.0;    // Path loss à d0=1m (free space 2.4GHz)
    constexpr double PATH_EXP        = 2.7;     // Exposant path loss (indoor/outdoor)
    constexpr double SHADOW_STD_DB   = 4.0;     // Écart-type shadowing (dB)
    constexpr double NOISE_FIGURE_DB = 10.0;    // Figure de bruit récepteur (dB)
    constexpr double THERMAL_NOISE_DBM = -174.0 + 10.0 * std::log10(250000.0); // kTB @ 250kbps
    // = -174 + 53.98 ≈ -120 dBm
    constexpr double SNR_FLOOR_DB    = 3.0;     // SNR minimum pour décodage BPSK

    // Q-function approx (Borjesson & Sundberg 1979 — précision 0.27%)
    inline double Qfunc(double x) {
        if (x < 0) return 1.0 - Qfunc(-x);
        const double a1 = 0.4361836, a2 = -0.1201676, a3 = 0.9372980;
        const double p  = 0.3326700;
        double t = 1.0 / (1.0 + p * x);
        return (a1*t + a2*t*t + a3*t*t*t) * std::exp(-0.5*x*x);
    }

    /**
     * Calcule la Packet Reception Rate (PRR) pour une distance donnée.
     *
     * Modèle :
     *   1. PL(d) = PL0 + 10*n*log10(d/1) [dB] — path loss
     *   2. PRx = PTx - PL(d) [dBm]
     *   3. SNR = PRx - NoiseFigure - ThermalNoise [dB]
     *   4. BER = Q(sqrt(2 * 10^(SNR/10))) pour BPSK
     *   5. PRR = (1 - BER)^N_bits
     *
     * Le shadowing est intégré en moyenne (E[PRR] sans tirage stochastique)
     * pour garder le modèle déterministe et reproductible sans seed canal.
     * Pour une évaluation avec variabilité, utiliser EvalPRRStochastic.
     *
     * @param dist_m  Distance entre émetteur et récepteur en mètres
     * @param n_bits  Nombre de bits du paquet (défaut : PKT_BITS)
     * @return        PRR ∈ [0.0, 1.0]
     */
    inline double EvalPRR(double dist_m, uint32_t n_bits = EvalCfg::PKT_BITS) {
        if (dist_m <= 0.1) return 1.0;  // Distance négligeable
        if (dist_m > EvalCfg::RADIO_RANGE * 1.5) return 0.0;  // Hors portée

        const double pl_db = PL_D0_DB + 10.0 * PATH_EXP * std::log10(dist_m);
        const double prx_dbm = PTX_DBM - pl_db;
        const double noise_floor_dbm = THERMAL_NOISE_DBM + NOISE_FIGURE_DB;
        const double snr_db = prx_dbm - noise_floor_dbm;
        const double snr_lin = std::pow(10.0, snr_db / 10.0);

        if (snr_db < SNR_FLOOR_DB) return 0.0;

        // BER pour BPSK (Proakis, "Digital Communications", 5th ed.)
        const double ber = Qfunc(std::sqrt(2.0 * snr_lin));
        // PRR = (1-BER)^N — probabilité de décodage correct de tout le paquet
        const double prr = std::pow(1.0 - ber, static_cast<double>(n_bits));
        return std::max(0.0, std::min(1.0, prr));
    }

    /**
     * Version stochastique avec shadowing gaussien (pour multi-run).
     * Utiliser avec std::mt19937 et std::normal_distribution.
     */
    inline double EvalPRRStochastic(double dist_m, double shadow_sample_db,
                                    uint32_t n_bits = EvalCfg::PKT_BITS) {
        if (dist_m <= 0.1) return 1.0;
        const double pl_db = PL_D0_DB + 10.0 * PATH_EXP * std::log10(dist_m)
                             + shadow_sample_db;  // X_sigma ~ N(0, SHADOW_STD_DB²)
        const double prx_dbm = PTX_DBM - pl_db;
        const double noise_floor_dbm = THERMAL_NOISE_DBM + NOISE_FIGURE_DB;
        const double snr_db = prx_dbm - noise_floor_dbm;
        if (snr_db < SNR_FLOOR_DB) return 0.0;
        const double snr_lin = std::pow(10.0, snr_db / 10.0);
        const double ber = Qfunc(std::sqrt(2.0 * snr_lin));
        return std::max(0.0, std::min(1.0, std::pow(1.0 - ber, static_cast<double>(n_bits))));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modèle de délai réaliste (IEEE 802.15.4 CSMA/CA)
// Référence : Pollin et al., IEEE Trans. Mobile Computing 2008
// ─────────────────────────────────────────────────────────────────────────────

namespace EvalDelay {
    constexpr double DATA_RATE_BPS   = 250e3;   // 250 kbps (IEEE 802.15.4)
    constexpr double SLOT_TIME_S     = 320e-6;  // 320 µs (IEEE 802.15.4 slot)
    constexpr double CW_MIN         = 8.0;      // CWmin CSMA/CA
    constexpr double BACKOFF_MEAN_S  = (CW_MIN / 2.0) * SLOT_TIME_S;  // ~1.28 ms
    constexpr double ACK_TIME_S     = 512e-6;   // ACK frame time

    /**
     * Délai bout-en-bout pour n_hops sauts avec charge réseau rho ∈ [0,1].
     *
     * Composantes par saut :
     *   D_tx      = PKT_BITS / DATA_RATE   (temps de transmission)
     *   D_backoff = BACKOFF_MEAN / (1-rho) (CSMA/CA — modèle M/G/1 approximé)
     *   D_ack     = ACK_TIME
     *   D_proc    = 0.5ms (traitement nœud, mesure empirique TelosB)
     *
     * @param n_hops           Nombre de sauts (1 = membre→CH, 2 = membre→CH→sink)
     * @param rho              Charge canal ∈ [0, 0.85] ; -1 = estimation automatique
     * @param n_alive          Nombre de nœuds vivants (fallback si avg_members_per_ch<0)
     * @param avg_members_per_ch  Nombre moyen de membres par CH actif (pour rho précis).
     *                            Passer -1 si inconnu (fallback sur n_alive).
     *
     * Correction FIX-3v2 : rho était calculé via sqrt(n_alive) → valeur quasi-fixe
     * (sqrt(300)=17.3 → capped 15 → rho=0.6 toute la simulation → délai figé 40.4ms).
     * Désormais rho est proportionnel au nombre réel de membres par CH :
     *   rho = avg_members_per_ch × load_per_node / DATA_RATE
     * Ce rho varie dynamiquement : faible en début de vie (~5-8 membres/CH),
     * élevé en fin de vie quand les CHs accumulent plus de membres orphelins
     * suite aux morts de nœuds voisins.
     * Valeurs attendues : 5 mems→rho=0.20→37ms ; 12 mems→rho=0.48→39ms ;
     *                     20 mems→rho=0.80→47ms (saturation en fin de vie).
     */
    inline double ComputeDelay_ms(uint32_t n_hops = 2, double rho = -1.0,
                                   uint32_t n_alive = 300,
                                   double avg_members_per_ch = -1.0) {
        if (rho < 0.0) {
            const double load_per_node = 10e3;  // 10 kbps par nœud (OnOff app)
            if (avg_members_per_ch > 0.0) {
                // [FIX-3v2] rho basé sur la charge réelle du canal CH
                // Chaque membre envoie ses données au CH sur le même canal partagé.
                // La contention augmente linéairement avec le nombre de membres.
                rho = avg_members_per_ch * load_per_node / DATA_RATE_BPS;
            } else {
                // Fallback conservateur (utilisé si avg_members_per_ch non fourni)
                // Estimation : ~10 voisins actifs en moyenne dans la portée radio
                const double n_neighbors_avg = std::min(10.0,
                    static_cast<double>(n_alive) / 30.0);  // ~10 pour 300 nœuds
                rho = n_neighbors_avg * load_per_node / DATA_RATE_BPS;
            }
        }
        rho = std::max(0.01, std::min(0.85, rho));

        const double d_tx_s     = static_cast<double>(EvalCfg::PKT_BITS) / DATA_RATE_BPS;
        const double d_backoff_s = BACKOFF_MEAN_S / (1.0 - rho);  // M/G/1 approx
        const double d_proc_s   = 0.5e-3;  // 0.5 ms processing (TelosB empirical)
        const double d_per_hop  = d_tx_s + d_backoff_s + ACK_TIME_S + d_proc_s;

        return d_per_hop * n_hops * 1e3;  // Conversion en ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modèle énergétique LEACH (Heinzelman 2002) — inchangé
// ─────────────────────────────────────────────────────────────────────────────

inline double EvalEtx(uint32_t bits, double dist) {
    return bits * EvalCfg::E_ELEC + bits * EvalCfg::E_AMP * dist * dist;
}
inline double EvalErx(uint32_t bits) {
    return bits * EvalCfg::E_ELEC;
}
inline double EvalEda(uint32_t bits) {
    return bits * EvalCfg::E_DA;
}
inline double EvalDrainMember(double distToCH) {
    // Membre → CH : TX 1 paquet (DRAIN_BITS=8000 bits) par step RL.
    // Le simulateur envoie 1 paquet complet à chaque step de contrôle (5s),
    // cohérent avec le trafic OnOffApplication et avec qrouting_sim/fdqn_*.
    // Ref : Heinzelman 2002 eq. (4).
    return EvalEtx(EvalCfg::DRAIN_BITS, distToCH);
}
inline double EvalDrainCH(uint32_t nMembers, double distToSink) {
    // CH : RX de chaque membre + agrégation + TX agrégé vers sink
    // Formule exacte Heinzelman 2002, eq. (5).
    // DRAIN_BITS=8000 bits par step RL — cohérent avec tous les simulateurs.
    return nMembers * EvalErx(EvalCfg::DRAIN_BITS)   // RX des membres
         + nMembers * EvalEda(EvalCfg::DRAIN_BITS)   // agrégation
         + EvalEtx(EvalCfg::DRAIN_BITS, distToSink); // TX vers sink
}

/**
 *  Drain d'un nœud isolé dans LEACH — idle listening réel.
 *
 * Selon Heinzelman 2000/2002, un nœud sans CH dans sa portée radio
 * NE TRANSMET PAS directement vers le sink. Il reste en mode veille
 * avec idle listening, consommant I_idle × V × t_step.
 *

 * Référence : TelosB datasheet — I_idle = 0.5 µA @ 3V
 *             Heinzelman 2000 Section IV-B : pas de TX direct sink.
 *             Heinzelman 2002 eq. (4) : E_idle = E_elec * l (écoute d'un paquet
 *             broadcasté par le CH) — DRAIN_BITS_PER_STEP si écoute active,
 *             IDLE_DRAIN_J si pure veille entre rounds.
 *
 * @param distToSink  Non utilisé (conservé pour compatibilité signature)
 * @return            Énergie idle = I_idle × V × RL_STEP_INTERVAL = 8.25 µJ
 */
inline double EvalDrainIsolated(double /*distToSink*/) {
    // Idle courant de veille uniquement — PAS de TX, PAS de RX actif
    return EvalCfg::IDLE_DRAIN_J;  // = 8.25 µJ/step (vs 400 µJ avec EvalErx)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Coefficient de Gini énergétiqueIX-4
// Mesure l'inégalité de la distribution d'énergie résiduelle entre nœuds.
// Gini=0 : parfaite équité. Gini=1 : toute l'énergie sur 1 nœud.
// Référence : Wang et al., "Energy-efficient clustering protocol" IEEE Trans 2021
// ─────────────────────────────────────────────────────────────────────────────

inline double EvalEnergyGini(const std::vector<double>& energies_alive) {
    if (energies_alive.size() < 2) return 0.0;
    std::vector<double> e = energies_alive;
    std::sort(e.begin(), e.end());
    const size_t n = e.size();
    double sum = 0.0, weighted = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += e[i];
        weighted += (i + 1) * e[i];
    }
    if (sum < 1e-12) return 0.0;
    return (2.0 * weighted) / (static_cast<double>(n) * sum) -
           (static_cast<double>(n) + 1.0) / static_cast<double>(n);
}

// ─────────────────────────────────────────────────────────────────────────────
// État d'un nœud capteur
// ─────────────────────────────────────────────────────────────────────────────

struct EvalNodeState {
    uint32_t id        = 0;
    double   x         = 0.0, y = 0.0;
    double   energy    = EvalCfg::E_INIT;
    bool     isAlive   = true;
    uint32_t clusterId = 0;
    bool     isClusterHead = false;
    double   distToSink    = 0.0;
    double   pepmRisk      = 0.0;
    uint32_t txCount       = 0;
    uint32_t reclusterCount= 0;
    double   totalReward   = 0.0;

    EvalNodeState() = default;
    EvalNodeState(uint32_t id_, double x_, double y_,
                  double e, double dSink)
        : id(id_), x(x_), y(y_), energy(e), distToSink(dSink) {}

    double NormEnergy() const {
        return std::max(0.0, energy / EvalCfg::E_INIT);
    }

    bool Consume(double drain) {
        if (!isAlive) return false;
        energy -= std::max(0.0, drain);
        if (energy <= 0.0) { energy = 0.0; isAlive = false; }
        return isAlive;
    }
};

inline double NodeDist(const EvalNodeState& a, const EvalNodeState& b) {
    return EvalDist(a.x, a.y, b.x, b.y);
}

// ─────────────────────────────────────────────────────────────────────────────
// Métriques par round — structure uniforme pour tous les modèles
// ─────────────────────────────────────────────────────────────────────────────

struct RoundMetrics {
    uint32_t round        = 0;
    double   time_s       = 0.0;
    uint32_t aliveNodes   = 0;
    uint32_t deadNodes    = 0;
    double   energyMean_J = 0.0;
    double   energyTotalConsumed_J = 0.0;
    double   pdr_pct      = 0.0;
    double   delay_ms     = 0.0;
    uint64_t pktEmitted   = 0;
    uint64_t pktDelivered = 0;
    uint32_t nClusters    = 0;
    uint32_t pepmAtRisk   = 0;
    double   energyGini   = 0.0;
    uint32_t isolatedNodes = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Résumé final d'un modèle
// ─────────────────────────────────────────────────────────────────────────────

struct ModelSummary {
    std::string modelName;
    double fnd_s = 0.0;
    double hnd_s = 0.0;
    double lnd_s = 0.0;
    double pdrStable_pct = 0.0;
    double pdrGlobal_pct = 0.0;
    double avgDelay_ms   = 0.0;
    double totalEnergy_J = 0.0;
    double avgGini       = 0.0;
    uint64_t totalPktSent = 0;
    uint64_t totalPktRecv = 0;
    uint32_t totalRounds  = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Export CSV — format uniforme (étendu avec Gini et Isolated)
// ─────────────────────────────────────────────────────────────────────────────

inline void InitMetricsCSV(std::ofstream& f, const std::string& path,
                            const std::string& modelName) {
    fs::create_directories(fs::path(path).parent_path());
    f.open(path);
    f << "# Model: " << modelName << "\n"
      << "# Seed=" << EvalCfg::SEED
      << " N=" << EvalCfg::N_NODES
      << " E_init=" << EvalCfg::E_INIT << "J"
      << " Area=" << EvalCfg::AREA_SIZE << "m\n"
      << "# Channel: Log-distance (n=2.7) + Thermal noise (IEEE 802.15.4)\n"
      << "# Delay: IEEE 802.15.4 CSMA/CA M/G/1 model\n"
      << "Round,Time_s,AliveNodes,DeadNodes,"
         "EnergyMean_J,EnergyConsumed_J,"
         "PDR_pct,Delay_ms,"
         "PktEmitted,PktDelivered,"
         "NClusters,PEPM_AtRisk,"
         "EnergyGini,IsolatedNodes\n";
}

inline void WriteMetricsRow(std::ofstream& f, const RoundMetrics& m) {
    f << std::fixed
      << m.round << ","
      << std::setprecision(1) << m.time_s << ","
      << m.aliveNodes << "," << m.deadNodes << ","
      << std::setprecision(6) << m.energyMean_J << ","
      << m.energyTotalConsumed_J << ","
      << std::setprecision(2) << m.pdr_pct << ","
      << m.delay_ms << ","
      << m.pktEmitted << "," << m.pktDelivered << ","
      << m.nClusters << "," << m.pepmAtRisk << ","
      << std::setprecision(4) << m.energyGini << ","
      << m.isolatedNodes << "\n";
    f.flush();
}

inline void WriteSummaryCSV(const std::string& path, const ModelSummary& s) {
    std::ofstream f(path);
    f << "# Summary: " << s.modelName << "\n"
      << "Metric,Value\n"
      << "Model," << s.modelName << "\n"
      << "FND_s," << s.fnd_s << "\n"
      << "HND_s," << s.hnd_s << "\n"
      << "LND_s," << s.lnd_s << "\n"
      << "PDR_Stable_pct," << s.pdrStable_pct << "\n"
      << "PDR_Global_pct," << s.pdrGlobal_pct << "\n"
      << "AvgDelay_ms," << s.avgDelay_ms << "\n"
      << "TotalEnergyConsumed_J," << s.totalEnergy_J << "\n"
      << "AvgEnergyGini," << s.avgGini << "\n"
      << "TotalPktSent," << s.totalPktSent << "\n"
      << "TotalPktRecv," << s.totalPktRecv << "\n"
      << "TotalRounds," << s.totalRounds << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers communs : initialisation des nœuds depuis NS-3
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<EvalNodeState> InitNodeStates(
        ns3::NodeContainer& sensors, uint32_t nNodes,
        double sinkX, double sinkY, double initEnergy) {
    std::vector<EvalNodeState> states;
    states.reserve(nNodes);
    for (uint32_t i = 0; i < nNodes; i++) {
        auto mob = sensors.Get(i)->GetObject<ns3::MobilityModel>();
        ns3::Vector v = mob->GetPosition();
        double dSink = EvalDist(v.x, v.y, sinkX, sinkY);
        states.emplace_back(sensors.Get(i)->GetId(), v.x, v.y, initEnergy, dSink);
    }
    return states;
}

// ─────────────────────────────────────────────────────────────────────────────
// Métriques snapshot par round (étendu)
// ─────────────────────────────────────────────────────────────────────────────

inline RoundMetrics ComputeRoundMetrics(
        uint32_t round, double now,
        const std::vector<EvalNodeState>& nodes,
        double initEnergy, uint64_t pktEmitted, uint64_t pktDelivered,
        double delay_ms, uint32_t nClusters) {
    RoundMetrics m;
    m.round = round; m.time_s = now;
    double sumE = 0.0, drainTotal = 0.0;
    std::vector<double> energiesAlive;

    for (const auto& n : nodes) {
        drainTotal += (initEnergy - n.energy);
        if (!n.isAlive) { m.deadNodes++; continue; }
        m.aliveNodes++;
        sumE += n.energy;
        energiesAlive.push_back(n.energy);
        if (n.pepmRisk > EvalCfg::PEPM_RISK_THRESHOLD) m.pepmAtRisk++;
        if (!n.isClusterHead && n.clusterId == 0) m.isolatedNodes++;
    }
    m.energyMean_J           = m.aliveNodes > 0 ? sumE / m.aliveNodes : 0.0;
    m.energyTotalConsumed_J  = drainTotal;
    m.pktEmitted   = pktEmitted;
    m.pktDelivered = pktDelivered;
    m.pdr_pct      = pktEmitted > 0 ? 100.0 * pktDelivered / pktEmitted : 100.0;
    m.delay_ms     = delay_ms;
    m.nClusters    = nClusters;
    m.energyGini   = EvalEnergyGini(energiesAlive);
    return m;
}

#endif // EVAL_COMMON_H
