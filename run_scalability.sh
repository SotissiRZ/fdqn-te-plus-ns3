#!/bin/bash
# =============================================================================
# run_scalability.sh — Campagne de scalabilité N=50/100/200/300
#
# Exécute tous les protocoles × toutes les seeds × toutes les tailles de réseau
# et sauvegarde dans results_eval/scale_N<N>/<PROTOCOL>/seed_<SEED>/
#
# Protocoles : FDQN-TE+, DQN-LEACH, LEACH, HEED, Q-Routing
# Tailles    : N = 50, 100, 200, 300  (modifiable via NODE_SIZES)
# Seeds      : 42 43 44 45 46         (modifiable via SEEDS)
#
# Usage :
#   chmod +x run_scalability.sh && ./run_scalability.sh
#   NODE_SIZES="50 100" SEEDS="42 43" ./run_scalability.sh   # sous-ensemble
# =============================================================================

# PAS de set -e : on gere les erreurs manuellement pour ne pas stopper la boucle
set -uo pipefail

# ─── Paramètres ───────────────────────────────────────────────────────────────
NS3_DIR="${NS3_DIR:-$HOME/ns-allinone-3.39/ns-3.39}"

NODE_SIZES="${NODE_SIZES:-50 100 200 300}"
SEEDS="${SEEDS:-42 43 44 45 46}"

AREA_SIZE=1000
INIT_ENERGY=1.2
SIM_DURATION=3500

RL_PORT_BASE=5555        # FDQN-TE+ : port = RL_PORT_BASE + seed_idx
DQN_PORT=5559            # DQN-LEACH : port fixe

RESULTS_BASE="results_eval"
PYTHON_CMD="${PYTHON_CMD:-python3}"

RL_SERVER_SCRIPT="${RL_SERVER_SCRIPT:-$NS3_DIR/model/rl_server_eval.py}"
DQN_SERVER_SCRIPT="${DQN_SERVER_SCRIPT:-$NS3_DIR/model/rl_server_dqnleach.py}"

# Couleurs
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# Compteurs globaux pour le bilan final
N_OK=0
N_FAIL=0
FAILURES=()

# ─── Vérification prérequis ───────────────────────────────────────────────────
if [ ! -f "$NS3_DIR/ns3" ]; then
    log_error "NS-3 introuvable : $NS3_DIR/ns3"
    log_error "Definir NS3_DIR=<chemin_ns3> avant d'executer."
    exit 1
fi

if [ ! -f "$RL_SERVER_SCRIPT" ]; then
    log_error "Serveur RL (FDQN-TE+) introuvable : $RL_SERVER_SCRIPT"
    exit 1
fi

if [ ! -f "$DQN_SERVER_SCRIPT" ]; then
    log_error "Serveur RL (DQN-LEACH) introuvable : $DQN_SERVER_SCRIPT"
    exit 1
fi

mkdir -p "$RESULTS_BASE"

# ─── Attendre qu'un port TCP soit pret (max 15 s) ────────────────────────────
wait_for_port() {
    local port=$1
    local label="${2:-serveur}"
    local elapsed=0
    while ! nc -z 127.0.0.1 "$port" 2>/dev/null; do
        sleep 1
        elapsed=$(( elapsed + 1 ))
        if [ "$elapsed" -ge 15 ]; then
            log_warn "  [$label] Timeout port $port -> mode Q-table fallback C++"
            return 1
        fi
    done
    return 0
}

# ─── Demarrer un serveur RL (retourne PID dans RL_PID) ───────────────────────
start_rl_server() {
    local script="$1" port="$2" label="$3" logfile="$4"
    log_info "  [$label] Demarrage serveur RL (port $port)..."
    $PYTHON_CMD "$script" --port "$port" > "$logfile" 2>&1 &
    RL_PID=$!
    if wait_for_port "$port" "$label"; then
        log_info "  [$label] Serveur pret (PID=$RL_PID)"
    else
        kill "$RL_PID" 2>/dev/null || true
        RL_PID=0
    fi
}

# ─── Arreter proprement un serveur RL ────────────────────────────────────────
stop_rl_server() {
    local pid="${1:-0}" label="${2:-serveur}"
    if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
        log_info "  [$label] Serveur arrete (PID=$pid)"
    fi
}

# ─── Lancer une simulation NS3 avec gestion d'erreur ─────────────────────────
# Ne stoppe PAS la boucle en cas d'echec : log l'erreur et continue.
run_sim() {
    local label="$1"
    local out_dir="$2"
    local sim_args="$3"
    local logfile="$out_dir/ns3.log"

    if "$NS3_DIR/ns3" run "$sim_args" > "$logfile" 2>&1; then
        log_ok "  [$label] Termine -> $out_dir"
        N_OK=$(( N_OK + 1 ))
    else
        local rc=$?
        log_fail "  [$label] Echec exit=$rc  (voir $logfile)"
        FAILURES+=("$label -> $out_dir")
        N_FAIL=$(( N_FAIL + 1 ))
    fi
}

# ─── Boucle principale : N x seeds x protocoles ──────────────────────────────
total_n=$(echo $NODE_SIZES | wc -w)
total_s=$(echo $SEEDS     | wc -w)
n_idx=0

for N in $NODE_SIZES; do
    n_idx=$(( n_idx + 1 ))
    echo ""
    log_info "╔══════════════════════════════════════════════════════╗"
    log_info "  N=$N noeuds  ($n_idx / $total_n)"
    log_info "╚══════════════════════════════════════════════════════╝"

    SCALE_BASE="$RESULTS_BASE/scale_N${N}"
    seed_idx=0

    for SEED in $SEEDS; do
        seed_idx=$(( seed_idx + 1 ))
        RL_PORT=$(( RL_PORT_BASE + seed_idx - 1 ))

        echo ""
        log_info "  ── N=$N  seed=$SEED  ($seed_idx / $total_s) ────────────────────"

        # ── 1. FDQN-TE+ ──────────────────────────────────────────────────────
        log_step "  [FDQN-TE+] N=$N seed=$SEED"
        OUT_DIR="$SCALE_BASE/FDQN_TEplus/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        start_rl_server "$RL_SERVER_SCRIPT" "$RL_PORT" "FDQN-TE+" \
                        "$OUT_DIR/rl_server.log"
        FDQN_PID=$RL_PID

        run_sim "FDQN-TE+ N=$N s=$SEED" "$OUT_DIR" \
            "scratch/fdqn_te_plus_eval --nNodes=$N --areaSize=$AREA_SIZE --initEnergy=$INIT_ENERGY --simDuration=$SIM_DURATION --seed=$SEED --rlPort=$RL_PORT --resultsDir=$OUT_DIR"

        stop_rl_server "$FDQN_PID" "FDQN-TE+"

        # ── 2. DQN-LEACH ─────────────────────────────────────────────────────
        log_step "  [DQN-LEACH] N=$N seed=$SEED"
        OUT_DIR="$SCALE_BASE/DQN_LEACH/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        start_rl_server "$DQN_SERVER_SCRIPT" "$DQN_PORT" "DQN-LEACH" \
                        "$OUT_DIR/rl_server.log"
        DQNL_PID=$RL_PID

        run_sim "DQN-LEACH N=$N s=$SEED" "$OUT_DIR" \
            "scratch/fdqn_leach --nNodes=$N --areaSize=$AREA_SIZE --initEnergy=$INIT_ENERGY --simDuration=$SIM_DURATION --seed=$SEED --rlPort=$DQN_PORT --resultsDir=$OUT_DIR"

        stop_rl_server "$DQNL_PID" "DQN-LEACH"

        # ── 3. LEACH ─────────────────────────────────────────────────────────
        log_step "  [LEACH] N=$N seed=$SEED"
        OUT_DIR="$SCALE_BASE/LEACH/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        run_sim "LEACH N=$N s=$SEED" "$OUT_DIR" \
            "scratch/leach_sim --nNodes=$N --seed=$SEED --resultsDir=$OUT_DIR"

        # ── 4. HEED ──────────────────────────────────────────────────────────
        log_step "  [HEED] N=$N seed=$SEED"
        OUT_DIR="$SCALE_BASE/HEED/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        run_sim "HEED N=$N s=$SEED" "$OUT_DIR" \
            "scratch/heed_sim --nNodes=$N --seed=$SEED --resultsDir=$OUT_DIR"

        # ── 5. Q-Routing ─────────────────────────────────────────────────────
        log_step "  [Q-Routing] N=$N seed=$SEED"
        OUT_DIR="$SCALE_BASE/QRouting/seed_${SEED}"
        mkdir -p "$OUT_DIR"

        run_sim "Q-Routing N=$N s=$SEED" "$OUT_DIR" \
            "scratch/qrouting_sim --nNodes=$N --seed=$SEED --resultsDir=$OUT_DIR"

    done  # seeds
done  # node sizes

# ─── Bilan final ──────────────────────────────────────────────────────────────
echo ""
log_info "╔══════════════════════════════════════════════════════╗"
log_info "  Campagne scalabilite terminee"
log_info "  Node sizes : $NODE_SIZES"
log_info "  Seeds      : $SEEDS"
log_info "  Succes     : $N_OK / $(( N_OK + N_FAIL ))"
if [ "$N_FAIL" -gt 0 ]; then
    log_warn "  Echecs     : $N_FAIL"
    for f in "${FAILURES[@]}"; do
        log_warn "    x $f"
    done
else
    log_info "  Echecs     : 0  -- tout s'est bien passe"
fi
echo ""
log_info "  Lancer le trace :"
log_info "    python3 aggregate_results.py --n_nodes_scale $NODE_SIZES"
log_info "╚══════════════════════════════════════════════════════╝"
