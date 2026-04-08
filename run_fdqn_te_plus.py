#!/usr/bin/env python3
# run_fdqn_te_plus.py

import subprocess
import sys
import os
import time
import signal

def main():
    print("="*70)
    print("FDQN-TE+ - Lancement de la simulation complète")
    print("="*70)

    # Vérifier que tous les modules existent
    required_files = [
        "fdqn_config.py",
        "addqn_agent.py",
        "pepm_lstm.py",
        "fedmeta_drl.py",
        "rl_server.py"
    ]

    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Fichier manquant: {f}")
            sys.exit(1)

    # Lancer le serveur RL
    print("\n🚀 Démarrage du serveur RL...")
    server_process = subprocess.Popen(
        [sys.executable, "rl_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Attendre que le serveur démarre
    time.sleep(2)

    # Vérifier que le serveur tourne
    if server_process.poll() is not None:
        print("❌ Le serveur RL n'a pas démarré correctement")
        sys.exit(1)

    print("✅ Serveur RL démarré")

    # Lancer NS-3
    print("\n📡 Lancement de NS-3...")
    ns3_process = subprocess.run([
        "./ns3", "run", "scratch/fdqn_te_plus",
        "--", "--nNodes=300", "--simDuration=1000"
    ])

    # Arrêter le serveur
    print("\n🛑 Arrêt du serveur RL...")
    server_process.terminate()
    server_process.wait()

    print("\n✅ Simulation terminée")
    print("   Résultats disponibles dans:")
    print("   - fdqnte_results.csv")
    print("   - fdqnte_rl_history.json")
    print("   - fdqnte_rounds.csv")

if __name__ == "__main__":
    main()
