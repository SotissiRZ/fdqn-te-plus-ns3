#!/bin/bash
# =============================================================================
# run_multiseed.sh — Campagne de simulation multi-seeds FDQN-TE+ vs baselines
#
# Objectif : valider la reproductibilité statistique des résultats.
# Exécute chaque protocole sur N_SEEDS graines distinctes et sauvegarde
# les résultats
#
# Protocoles simulés :
#   FDQN-TE+   (rl_server_eval.py  — port RL_PORT_BASE + seed_idx)
#   DQN-LEACH  (rl_server_dqnleach.py — port DQN_PORT_BASE + seed_idx)
#   LEACH      (scratch/leach_sim)
#   HEED       (scratch/heed_sim)
#   Q-Routing  (scratch/qrouting_sim)
#
# Usage :
#   chmod +x run_multiseed.sh
#   ./run_multiseed.sh              # 5 seeds par défaut
#   N_SEEDS=10 ./run_multiseed.sh   # 10 seeds
#
# Pré-requis : NS-3 compilé, rl_server_eval.py et rl_server_dqnleach.py
#              disponibles aux chemins définis ci-dessous.
# =============================================================================

set -euo pipefail

# ─── Paramètres ───────────────────────────────────────────────────────────────
NS3_DIR="${NS3_DIR:-$HOME/ns-allinone-3.39/ns-3.39}"

SEEDS="${SEEDS:-42 43 44 45 46}"    # 5 seeds par défaut
N_NODES=300
AREA_SIZE=1000
INIT_ENERGY=1.2
SIM_DURATION=3500

# Ports pour les serveurs RL
RL_PORT_BASE=5555        # FDQN-TE+  → 5555, 5556, 5557 ... (dynamique par seed)
DQN_PORT=5559            # DQN-LEACH → 5559 fixe (port natif de rl_server_dqnleach.py)

RESULTS_BASE="results_eval"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Serveur RL principal (FDQN-TE+)
RL_SERVER_SCRIPT="${RL_SERVER_SCRIPT:-$NS3_DIR/model/rl_server_eval.py}"
# Serveur RL DQN-LEACH (baseline DRL externe)
DQN_SERVER_SCRIPT="${DQN_SERVER_SCRIPT:-$NS3_DIR/model/rl_server_dqnleach.py}"

# Couleurs pour logs
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ─── Vérification prérequis ───────────────────────────────────────────────────
if [ ! -f "$NS3_DIR/ns3" ]; then
    log_error "NS-3 introuvable : $NS3_DIR/ns3"
    log_error "Définir NS3_DIR=<chemin_ns3> avant d'exécuter."
    exit 1
fi

if [ ! -f "$RL_SERVER_SCRIPT" ]; then
    log_error "Serveur RL (FDQN-TE+) introuvable : $RL_SERVER_SCRIPT"
    log_error "Définir RL_SERVER_SCRIPT=<chemin_complet> avant d'exécuter."
    exit 1
fi

if [ ! -f "$DQN_SERVER_SCRIPT" ]; then
    log_error "Serveur RL (DQN-LEACH) introuvable : $DQN_SERVER_SCRIPT"
    log_error "Définir DQN_SERVER_SCRIPT=<chemin_complet> avant d'exécuter."
    exit 1
fi

mkdir -p "$RESULTS_BASE"

# ─── Fonction : attendre que le port TCP soit prêt ────────────────────────────
wait_for_port() {
    local port=$1
    local label="${2:-serveur}"
    local max_wait=15
    local elapsed=0
    while ! nc -z 127.0.0.1 "$port" 2>/dev/null; do
        sleep 1
        elapsed=$(( elapsed + 1 ))
        if [ "$elapsed" -ge "$max_wait" ]; then
            log_warn "  [$label] Timeout attente port $port (${max_wait}s)"
            return 1
        fi
    done
    return 0
}

# ─── Fonction : démarrer un serveur RL et attendre qu'il soit prêt ───────────

start_rl_server() {
    local script="$1"
    local port="$2"
    local label="$3"
    local logfile="$4"

    log_info "  [$label] Démarrage serveur RL (port $port)..."
    $PYTHON_CMD "$script" --port "$port" > "$logfile" 2>&1 &
    RL_PID=$!

    if wait_for_port "$port" "$label"; then
        log_info "  [$label] Serveur RL prêt sur port $port (PID=$RL_PID)"
    else
        log_warn "  [$label] Serveur non prêt → mode Q-table fallback C++"
        kill "$RL_PID" 2>/dev/null || true
        RL_PID=0
    fi
}

# ─── Fonction : arrêter proprement un serveur RL ─────────────────────────────
stop_rl_server() {
    local pid="$1"
    local label="${2:-serveur}"
    if [ "${pid:-0}" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
        log_info "  [$label] Serveur RL arrêté (PID=$pid)"
    fi
}

# ─── Boucle principale ────────────────────────────────────────────────────────
seed_idx=0
total_seeds=$(echo $SEEDS | wc -w)

for SEED in $SEEDS; do
    log_info "═══════════════════════════════════════════════════════"
    log_info " Seed $SEED ($(( seed_idx + 1 )) / $total_seeds)"
    log_info "═══════════════════════════════════════════════════════"

    RL_PORT=$(( RL_PORT_BASE + seed_idx ))    # port FDQN-TE+ (dynamique)

    # ── 1. FDQN-TE+ ──────────────────────────────────────────────────────────
    log_step "  [FDQN-TE+] Seed=$SEED"
    FDQN_DIR="$RESULTS_BASE/FDQN_TEplus/seed_${SEED}"
    mkdir -p "$FDQN_DIR"

    start_rl_server "$RL_SERVER_SCRIPT" "$RL_PORT" "FDQN-TE+" \
                    "$FDQN_DIR/rl_server.log"
    FDQN_RL_PID=$RL_PID

    "$NS3_DIR/ns3" run \
        "scratch/fdqn_te_plus_eval \
        --nNodes=$N_NODES \
        --areaSize=$AREA_SIZE \
        --initEnergy=$INIT_ENERGY \
        --simDuration=$SIM_DURATION \
        --seed=$SEED \
        --rlPort=$RL_PORT \
        --resultsDir=$FDQN_DIR" \
        > "$FDQN_DIR/ns3.log" 2>&1

    stop_rl_server "$FDQN_RL_PID" "FDQN-TE+"
    log_info "  [FDQN-TE+] ✓ Terminé → $FDQN_DIR"

    # ── 2. DQN-LEACH (baseline DRL externe) ──────────────────────────────────
    log_step "  [DQN-LEACH] Seed=$SEED"
    DQNL_DIR="$RESULTS_BASE/DQN_LEACH/seed_${SEED}"
    mkdir -p "$DQNL_DIR"

    start_rl_server "$DQN_SERVER_SCRIPT" "$DQN_PORT" "DQN-LEACH" \
                    "$DQNL_DIR/rl_server.log"
    DQNL_RL_PID=$RL_PID

    # fdqn_leach.cc accepte : --nNodes --areaSize --initEnergy --simDuration
    #                         --seed --rlPort --resultsDir
    "$NS3_DIR/ns3" run \
        "scratch/fdqn_leach \
        --nNodes=$N_NODES \
        --areaSize=$AREA_SIZE \
        --initEnergy=$INIT_ENERGY \
        --simDuration=$SIM_DURATION \
        --seed=$SEED \
        --rlPort=$DQN_PORT \
        --resultsDir=$DQNL_DIR" \
        > "$DQNL_DIR/ns3.log" 2>&1

    stop_rl_server "$DQNL_RL_PID" "DQN-LEACH"
    log_info "  [DQN-LEACH] ✓ Terminé → $DQNL_DIR"

    # ── 3. HEED (baseline sans RL) ────────────────────────────────────────────
    log_step "  [HEED] Seed=$SEED"
    HEED_DIR="$RESULTS_BASE/HEED/seed_${SEED}"
    mkdir -p "$HEED_DIR"

    # heed_sim.cc accepte : --seed --resultsDir
    "$NS3_DIR/ns3" run \
        "scratch/heed_sim \
        --seed=$SEED \
        --resultsDir=$HEED_DIR" \
        > "$HEED_DIR/ns3.log" 2>&1
    log_info "  [HEED] ✓ Terminé → $HEED_DIR"

    # ── 4. LEACH baseline ─────────────────────────────────────────────────────
    log_step "  [LEACH] Seed=$SEED"
    LEACH_DIR="$RESULTS_BASE/LEACH/seed_${SEED}"
    mkdir -p "$LEACH_DIR"

    "$NS3_DIR/ns3" run \
        "scratch/leach_sim \
        --seed=$SEED \
        --resultsDir=$LEACH_DIR" \
        > "$LEACH_DIR/ns3.log" 2>&1
    log_info "  [LEACH] ✓ Terminé → $LEACH_DIR"

    # ── 5. Q-Routing baseline ─────────────────────────────────────────────────
    log_step "  [Q-Routing] Seed=$SEED"
    QROUT_DIR="$RESULTS_BASE/QRouting/seed_${SEED}"
    mkdir -p "$QROUT_DIR"

    "$NS3_DIR/ns3" run \
        "scratch/qrouting_sim \
        --seed=$SEED \
        --resultsDir=$QROUT_DIR" \
        > "$QROUT_DIR/ns3.log" 2>&1
    log_info "  [Q-Routing] ✓ Terminé → $QROUT_DIR"

    seed_idx=$(( seed_idx + 1 ))
done

log_info ""
log_info "═══════════════════════════════════════════════════════"
log_info " Toutes les simulations terminées."
log_info " Protocoles : FDQN-TE+, DQN-LEACH, HEED, LEACH, Q-Routing"
log_info " Lancer l'agrégation : python3 aggregate_results.py"
log_info "═══════════════════════════════════════════════════════"
