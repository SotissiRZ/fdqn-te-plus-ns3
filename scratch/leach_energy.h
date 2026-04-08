/* =============================================================================
 * leach_energy.h — Modèle énergétique LEACH pour FDQN-TE+
 *
 * Implémente le modèle radio de Heinzelman et al. (2002) :
 *   • E_tx(k, d) = k*E_elec + k*E_amp*d²   — Transmission libre espace
 *   • E_rx(k)    = k*E_elec                 — Réception
 *   • E_da(k)    = k*E_da                   — Agrégation (CH uniquement)
 *
 * DEUX NIVEAUX D'ÉNERGIE :
 *   1. Logique (ns.energy) : bilan LEACH analytique, utilisé pour
 *      les métriques (FND, HND, énergie moyenne) et la détection mort.
 *   2. NS-3 BasicEnergySource : consommation physique WiFi via courants.
 *      Utilisé uniquement pour EnergyCallback (cohérence avec NS-3).
 *
 * Placement NS-3 : scratch/ (même dossier que fdqn_te_plus.cc)
 * ============================================================================= */

#ifndef LEACH_ENERGY_H
#define LEACH_ENERGY_H

#include "fdqn_config.h"
#include <cstdint>

#include <cmath>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Fonctions énergie LEACH (inline — pas de coût d'appel)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Énergie de transmission (TX libre espace).
 * @param bits    Taille du paquet en bits
 * @param dist    Distance TX → RX en mètres
 * @return        Énergie consommée en Joules
 *
 * Formule : E_tx = bits * E_ELEC + bits * E_AMP * dist²
 *   - E_ELEC = 50 nJ/bit  (circuits émetteur)
 *   - E_AMP  = 10 pJ/bit/m² (amplificateur en espace libre)
 */
inline double LeachEtx(uint32_t bits, double dist) {
    return bits * FdqnCfg::E_ELEC
         + bits * FdqnCfg::E_AMP * dist * dist;
}

/**
 * Énergie de réception (RX).
 * @param bits    Taille du paquet en bits
 * @return        Énergie consommée en Joules
 *
 * Formule : E_rx = bits * E_ELEC
 *   (pas d'amplification en réception)
 */
inline double LeachErx(uint32_t bits) {
    return bits * FdqnCfg::E_ELEC;
}

/**
 * Énergie d'agrégation de données (CH uniquement).
 * @param bits    Taille d'un paquet agrégé en bits
 * @return        Énergie consommée en Joules
 *
 * Formule : E_da = bits * E_DA (5 nJ/bit)
 */
inline double LeachEda(uint32_t bits) {
    return bits * FdqnCfg::E_DA;
}

/**
 * Distance de croisement : au-delà de cette distance, le modèle
 * multipath est préférable au modèle libre espace.
 * d_crossover = sqrt(E_ELEC / E_AMP) ≈ 70.7 m
 * FDQN-TE+ utilise uniquement le modèle libre espace (d < RADIO_RANGE).
 */
inline double CrossoverDist() {
    return std::sqrt(FdqnCfg::E_ELEC / FdqnCfg::E_AMP);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilan énergie d'un round LEACH
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Énergie consommée par un MEMBRE en un step RL.
 * @param distToCH   Distance du membre à son CH
 * @param bits       Bits transmis (défaut = DRAIN_BITS)
 */
inline double LeachMemberRound(double distToCH,
                               uint32_t bits = FdqnCfg::DRAIN_BITS) {
    return LeachEtx(bits, distToCH);
}

/**
 * Énergie consommée par un CH en un step RL.
 * @param nMembers     Nombre de membres dans le cluster
 * @param distToSink   Distance du CH au sink
 * @param bits         Bits par paquet (défaut = DRAIN_BITS)
 */
inline double LeachCHRound(uint32_t nMembers, double distToSink,
                           uint32_t bits = FdqnCfg::DRAIN_BITS) {
    double eRx = nMembers * LeachErx(bits);
    double eDa = nMembers * LeachEda(bits);
    double eTx = LeachEtx(bits, distToSink);
    return eRx + eDa + eTx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Structure : état énergétique d'un nœud
// ─────────────────────────────────────────────────────────────────────────────

struct EnergyState {
    double  energy;       // Énergie résiduelle logique (J) — modèle LEACH
    bool    isAlive;      // Vivant si energy > 0
    double  totalDrain;   // Énergie totale drainée depuis le début (J)
    uint32_t txCount;     // Nombre de transmissions effectuées

    explicit EnergyState(double initE = FdqnCfg::E_INIT)
        : energy(initE), isAlive(true), totalDrain(0.0), txCount(0) {}

    /**
     * Déduire la consommation d'une transmission et mettre à jour l'état.
     * @param drain   Énergie à déduire (J) — calculée avec LeachEtx / LeachCHRound
     * @return        true si le nœud est toujours vivant après la déduction
     */
    bool Consume(double drain) {
        if (!isAlive) return false;
        drain = std::max(0.0, drain);
        energy -= drain;
        totalDrain += drain;
        txCount++;
        if (energy <= 0.0) {
            energy  = 0.0;
            isAlive = false;
        }
        return isAlive;
    }

    /** Fraction d'énergie résiduelle ∈ [0, 1] */
    double NormalizedEnergy(double eInit = FdqnCfg::E_INIT) const {
        return (eInit > 0.0) ? std::max(0.0, energy / eInit) : 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Formule LEACH optimale : nombre de clusters k_opt
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Nombre optimal de clusters selon la formule LEACH (Heinzelman 2002).
 *
 *   k_opt = sqrt(N / (2π)) * sqrt(ε_fs / ε_da) * (A / d_toBS²)
 *
 * Note : dans FDQN-TE+, nClusters est calculé depuis la topologie réelle
 * (degré moyen, portée radio) pour garantir 8–12 membres/cluster.
 * Cette formule sert de référence théorique (documentation).
 *
 * @param N       Nombre de nœuds vivants
 * @param area    Superficie de la zone (m²)
 * @param dToBS   Distance moyenne nœud → sink (m)
 */
inline uint32_t LeachKOpt(uint32_t N, double area, double dToBS) {
    if (dToBS < 1.0) dToBS = 1.0;
    double kOpt = std::sqrt((double)N / (2.0 * M_PI))
                * std::sqrt(FdqnCfg::E_AMP / FdqnCfg::E_DA)
                * (area / (dToBS * dToBS));
    return static_cast<uint32_t>(std::max(1.0, std::round(kOpt)));
}

#endif // LEACH_ENERGY_H
