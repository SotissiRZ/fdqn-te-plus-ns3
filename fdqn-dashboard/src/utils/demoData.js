// src/utils/demoData.js
// Données de démonstration calées sur les vraies valeurs de la simulation FDQN-TE+
// (N=300, E_INIT=2J, simDuration=4000s, FND=2447s, HND=3207s, PDR_RL moy=92.4%)

export const DEMO_SUMMARY = {
  N:                      300,
  AliveNodes:             48,
  DeadNodes:              252,
  EnergyMean_J:           0.1638,
  EnergyTotalConsumed_J:  592.14,
  PDR_RL_pct:             87.18,
  PDR_NS3_pct:            100.00,
  PDR_pct:                87.18,
  AvgDelay_ms:            4.77,
  TxPackets:              31867,
  RxPackets:              31867,
  RL_PktEmitted:          194299,
  RL_PktDelivered:        169395,
  IFO_Rounds:             40,
  FND_t:                  2447,
  HND_t:                  3207,
  LND_t:                  0,
  RL_Steps:               194299,
  FedRounds:              3885,
  FedRound:               3885,
  Seed:                   42,
  SimDuration_s:          4000,
  RadioRange_m:           100,
  AreaSize_m:             1000,
  InitEnergy_J:           2.0,
};

/**
 * Génère des données énergie de démo qui imitent la vraie simulation.
 * 79 rounds, FND à t=2447s, HND à t=3207s, PDR décroissant.
 */
export function demoEnergy() {
  const rows = [];
  const N = 300;
  let alive = 300, energy = 2.0, drained = 0;

  for (let r = 1; r <= 79; r++) {
    const t = r * 50;

    // Drain énergétique réaliste (calé sur les vraies données)
    const drainStep = t < 2000 ? 8.0 : t < 2500 ? 9.5 : 5.5;
    energy = Math.max(0, energy - drainStep / N);
    drained += drainStep;

    // Mortalité des nœuds (commence à t=2447s)
    if (t > 2450 && alive > 49) {
      if (t < 2700)       alive = Math.max(280, alive - Math.floor(1 + Math.random() * 3));
      else if (t < 2900)  alive = Math.max(200, alive - Math.floor(10 + Math.random() * 15));
      else if (t < 3200)  alive = Math.max(120, alive - Math.floor(15 + Math.random() * 20));
      else if (t < 3500)  alive = Math.max(60,  alive - Math.floor(8  + Math.random() * 12));
      else                alive = Math.max(49,  alive - Math.floor(2  + Math.random() * 5));
    }
    const dead = N - alive;

    // PDR RL: commence ~99.5%, décroît progressivement
    let pdrRL;
    if (t < 1000)       pdrRL = 99.5  - r * 0.05  + (Math.random() - 0.5) * 0.3;
    else if (t < 1500)  pdrRL = 97.5  - (r-20)*0.2 + (Math.random() - 0.5) * 0.5;
    else if (t < 2000)  pdrRL = 92.5  - (r-30)*0.15+ (Math.random() - 0.5) * 0.4;
    else if (t < 2450)  pdrRL = 89.2  + (Math.random() - 0.5) * 0.5;
    else                pdrRL = 89.2  - (r-49)*0.04 + (Math.random() - 0.5) * 0.3;
    pdrRL = Math.max(87, Math.min(100, pdrRL));

    // PEPM@risque: monte à partir de t=1500s
    const atRisk = t < 1500 ? 0 : Math.min(296, Math.floor((t - 1500) / 10 + Math.random() * 5));

    // Clusters actifs
    const nClusters = alive > 200 ? 23 : alive > 100 ? 18 : alive > 50 ? 8 : 1;

    rows.push({
      round: r, time: t,
      alive, dead,
      energy: +energy.toFixed(4),
      energyMin: +(energy * (0.35 + Math.random() * 0.1)).toFixed(4),
      energyMax: +(Math.min(2, energy * (1.02 + Math.random() * 0.05))).toFixed(4),
      energyStd: +(energy * (0.1 + Math.random() * 0.05)).toFixed(4),
      drained: +drained.toFixed(2),
      totalEnergy: +(energy * alive).toFixed(2),
      pdrRL: +pdrRL.toFixed(2),
      pdrNS3: 100,
      delay: t < 500 ? 4.85 : 4.76,
      atRisk,
      pepmMean: +(0.22 + r * 0.006).toFixed(4),
      fnd: t >= 2447 ? 2447 : 0,
      hnd: t >= 3207 ? 3207 : 0,
      lnd: 0,
      rlSteps: r * 3000 - Math.floor(Math.random() * 100),
      fedRound: r * 60 - Math.floor(Math.random() * 5),
      ifoRound: Math.floor(r / 2),
      nClusters,
      rlEmit:  r * 3000,
      rlDeliv: Math.floor(r * 3000 * pdrRL / 100),
    });
  }
  return rows;
}

/**
 * Génère des données RL de démo imitant la structure réelle de fdqnte_rl_history.json.
 */
export function demoRL() {
  const history = [];
  let alive = 300, energy = 2.0, drained = 0;

  for (let r = 1; r <= 79; r++) {
    const t = r * 50;
    energy = Math.max(0, energy - 8.0 / 300);
    drained += 8.0;
    if (t > 2450 && alive > 49) alive = Math.max(49, alive - (t > 2800 ? 15 : t > 2650 ? 5 : 1));

    const pdrRL = t < 1000 ? 99.5 - r * 0.05 : Math.max(87, 92 - (r - 20) * 0.06);
    const rewardMean = 100 + r * 3.2 + (Math.random() - 0.5) * 20;

    history.push({
      round: r, time: t,
      alive, dead: 300 - alive,
      energy: +energy.toFixed(4),
      drained: +drained.toFixed(2),
      pdrRL: +pdrRL.toFixed(2), pdrNS3: 100,
      delay: t < 500 ? 4.85 : 4.76,
      rlSteps: r * 3000,
      fedRound: r * 60,
      nClusters: alive > 200 ? 23 : alive > 100 ? 15 : 5,
      atRisk: t < 1500 ? 0 : Math.min(296, Math.floor((t - 1500) / 10)),
      rewardMin:  +(rewardMean * 0.7).toFixed(2),
      rewardMax:  +(rewardMean * 1.3).toFixed(2),
      rewardMean: +rewardMean.toFixed(2),
      qMean: 0.5,
    });
  }

  return {
    info: { nNodes:300, initEnergy_J:2, simDuration_s:4000, areaSize_m:1000, radioRange_m:100, seed:42 },
    metrics: DEMO_SUMMARY,
    history,
    series: history.map(h => ({
      step: h.rlSteps, round: h.round,
      reward: h.rewardMean / 500,
      loss: Math.max(0, 0.4 * Math.exp(-h.round * 0.04) + Math.random() * 0.05),
      epsilon: Math.max(0.05, Math.pow(0.995, h.rlSteps)),
    })),
    config: { gamma: 0.99, lr: "3e-4", epsilon_decay: 0.995, fed_period: 50 },
    stats: { global_step: 194299, actions_requested: 194299, rewards_received: 194195 },
  };
}

/**
 * Génère une topologie de démo avec 300 nœuds, 24 CH, clusters réalistes.
 */
export function demoTopo() {
  const nodes = [];
  const N = 300, N_CH = 24;
  const SINK_X = 500, SINK_Y = 500;

  // Positionner les CH uniformément
  const chPositions = Array.from({ length: N_CH }, (_, i) => {
    const angle = (i / N_CH) * 2 * Math.PI;
    const r = 200 + Math.random() * 200;
    return { x: SINK_X + r * Math.cos(angle), y: SINK_Y + r * Math.sin(angle) };
  });

  let nodeId = 0;
  chPositions.forEach((chPos, ci) => {
    const membersCount = 8 + Math.floor(Math.random() * 4);
    // CH
    const dist = Math.hypot(chPos.x - SINK_X, chPos.y - SINK_Y);
    nodes.push({
      id: nodeId++, x: +chPos.x.toFixed(1), y: +chPos.y.toFixed(1),
      clusterId: nodeId - 1, isCH: true,
      energy: 2.0, energyNorm: 1.0,
      distToSink: +dist.toFixed(1),
      pepmRisk: 0.0, isAlive: true,
      txCount: 0, reclusterCount: 0,
      fitness: +(0.9 + Math.random() * 0.28).toFixed(4),
    });
    // Membres
    for (let m = 0; m < membersCount; m++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 20 + Math.random() * 110;
      const mx = Math.max(0, Math.min(1000, chPos.x + r * Math.cos(angle)));
      const my = Math.max(0, Math.min(1000, chPos.y + r * Math.sin(angle)));
      const mDist = Math.hypot(mx - SINK_X, my - SINK_Y);
      nodes.push({
        id: nodeId++, x: +mx.toFixed(1), y: +my.toFixed(1),
        clusterId: nodes[nodes.length - 1 - m]?.clusterId ?? ci,
        isCH: false,
        energy: 2.0, energyNorm: 1.0,
        distToSink: +mDist.toFixed(1),
        pepmRisk: 0.0, isAlive: true,
        txCount: 0, reclusterCount: 0,
        fitness: +(0.83 + Math.random() * 0.35).toFixed(4),
      });
    }
    if (nodeId >= N) return;
  });

  // Compléter si besoin
  while (nodes.length < N) {
    const x = Math.random() * 1000, y = Math.random() * 1000;
    nodes.push({
      id: nodeId++, x: +x.toFixed(1), y: +y.toFixed(1),
      clusterId: 0, isCH: false,
      energy: 2.0, energyNorm: 1.0,
      distToSink: +Math.hypot(x - SINK_X, y - SINK_Y).toFixed(1),
      pepmRisk: 0.0, isAlive: true,
      txCount: 0, reclusterCount: 0,
      fitness: +(0.83 + Math.random() * 0.35).toFixed(4),
    });
  }

  return nodes.slice(0, N);
}