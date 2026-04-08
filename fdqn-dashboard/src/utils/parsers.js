// src/utils/parsers.js
/**
 * fdqnte_energy.csv
 * Round,Time_s,AliveNodes,DeadNodes,EnergyMean_J,EnergyStdDev_J,
 * EnergyMin_J,EnergyMax_J,TotalDrained_J,PDR_pct,FND_t,HND_t,
 * RLSteps,FedRound,IFORound,AtRiskPEPM
 *
 * NOTE: PDR_pct est toujours 0.0 pendant la simulation (calculé post-sim
 * par FlowMonitor). Charger fdqnte_summary.csv pour le PDR réel.
 */
export function parseEnergyCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  if (lines.length < 2) return [];
  const hdr = lines[0].split(",").map(h => h.trim());
  const g = (row, ...keys) => {
    for (const k of keys) { const i = hdr.indexOf(k); if (i !== -1) return parseFloat(row[i]) || 0; }
    return 0;
  };
  return lines.slice(1).map(line => {
    const v = line.split(",");
    return {
      round:      g(v, "Round"),
      time:       g(v, "Time_s"),
      alive:      g(v, "AliveNodes"),
      dead:       g(v, "DeadNodes"),
      energy:     g(v, "EnergyMean_J"),
      energyMin:  g(v, "EnergyMin_J"),
      energyMax:  g(v, "EnergyMax_J"),
      energyStd:  g(v, "EnergyStdDev_J"),
      drained:    g(v, "TotalDrained_J"),
      totalEnergy:g(v, "TotalEnergy_J"),
      pdrRL:      g(v, "PDR_RL_pct"),
      pdrNS3:     g(v, "PDR_NS3_pct"),
      delay:      g(v, "AvgDelay_ms"),
      atRisk:     g(v, "AtRiskPEPM"),
      pepmMean:   g(v, "PEPMRiskMean"),
      fnd:        g(v, "FND_s"),
      hnd:        g(v, "HND_s"),
      lnd:        g(v, "LND_s"),
      rlSteps:    g(v, "RLSteps"),
      fedRound:   g(v, "FedRound"),
      ifoRound:   g(v, "IFORound"),
      nClusters:  g(v, "NClusters"),
      rlEmit:     g(v, "RL_PktEmitted"),
      rlDeliv:    g(v, "RL_PktDelivered"),
    };
  }).filter(r => r.round > 0 || r.time > 0);
}

export function parseSummaryCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  const result = {};
  lines.forEach(line => {
    const parts = line.split(",");
    const k = parts[0]?.trim();
    const v = parts[1]?.trim();
    // Sauter la ligne d'en-tête "Param,Value"
    if (!k || k === "Param" || v === undefined) return;
    const num = parseFloat(v);
    result[k] = isNaN(num) ? v : num;
  });
  // Alias FedRounds → FedRound
  if (result.FedRounds !== undefined && result.FedRound === undefined)
    result.FedRound = result.FedRounds;
  return result;
}

/**
 * fdqnte_topology_final.csv  (ou _initial.csv ou _rXXXX.csv)
 * NodeId,X,Y,ClusterId,IsClusterHead,Energy,EnergyNorm,
 * DistToSink,PEPMRisk,IsAlive,TxCount,ReclusterCount,Fitness
 *
 * IMPORTANT: IsClusterHead et IsAlive sont des entiers 0/1 dans le CSV.
 *            EnergyNorm est dans [0,1] — c'est cette colonne qu'on utilise
 *            pour colorier les nœuds (Energy est en Joules, pas normalisée).
 */
export function parseTopologyCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  if (lines.length < 2) return [];
  const hdr = lines[0].split(",").map(h => h.trim());
  const gi = (row, k) => { const i = hdr.indexOf(k); return i !== -1 ? row[i]?.trim() : undefined; };
  const gf = (row, k, fb=0) => { const v = gi(row,k); return v !== undefined ? (parseFloat(v) ?? fb) : fb; };
  const gb = (row, k) => gi(row,k) === "1";   // entier 0/1 → booléen

  return lines.slice(1).map(line => {
    const v = line.split(",");
    return {
      id:          gf(v,"NodeId"),
      x:           gf(v,"X"),
      y:           gf(v,"Y"),
      clusterId:   gf(v,"ClusterId"),
      isCH:        gb(v,"IsClusterHead"),   // ← entier 0/1
      energy:      gf(v,"Energy"),
      energyNorm:  gf(v,"EnergyNorm"),      // ← [0,1] utilisé pour couleurs
      distToSink:  gf(v,"DistToSink"),
      pepmRisk:    gf(v,"PEPMRisk"),
      isAlive:     gb(v,"IsAlive"),          // ← entier 0/1
      txCount:     gf(v,"TxCount"),
      reclusterCount: gf(v,"ReclusterCount"),
      fitness:     gf(v,"Fitness"),
    };
  }).filter(n => n.x >= 0 && n.y >= 0);
}

/**
 * fdqnte_rl_history.json
 * { config, stats:{global_step,...}, history:{loss[], reward[]},
 *   pepm_summary, fed_stats }
 */
export function parseRLJson(text) {
  const raw  = JSON.parse(text);
  const hist = Array.isArray(raw.history) ? raw.history : [];

  // Structure réelle fdqnte_rl_history.json:
  // history[i] = { round, timestamp_s, alive_nodes, dead_nodes, avg_energy_J,
  //   total_energy_consumed_J, pdr_RL_pct, pdr_NS3_pct, avg_delay_ms,
  //   rl_steps, fed_round, n_clusters, at_risk_pepm,
  //   rewards:{min,max,mean}, q_values:{min,max,mean} }
  // simulation_info = { nNodes, initEnergy_J, simDuration_s, areaSize_m, radioRange_m, seed }
  // metrics = { fnd_time_s, hnd_time_s, avg_pdr_RL_pct, avg_delay_ms, total_energy_consumed_J }
  const history = hist.map(h => ({
    round:      h.round,
    time:       h.timestamp_s,
    alive:      h.alive_nodes,
    dead:       h.dead_nodes,
    energy:     h.avg_energy_J,
    drained:    h.total_energy_consumed_J,
    pdrRL:      h.pdr_RL_pct,
    pdrNS3:     h.pdr_NS3_pct,
    delay:      h.avg_delay_ms,
    rlSteps:    h.rl_steps,
    fedRound:   h.fed_round,
    nClusters:  h.n_clusters,
    atRisk:     h.at_risk_pepm,
    rewardMin:  h.rewards?.min  ?? 0,
    rewardMax:  h.rewards?.max  ?? 0,
    rewardMean: h.rewards?.mean ?? 0,
    qMin:       h.q_values?.min  ?? 0,
    qMax:       h.q_values?.max  ?? 0,
    qMean:      h.q_values?.mean ?? 0,
  }));

  return {
    history,
    info:    raw.simulation_info || {},
    metrics: raw.metrics         || {},
  };
}

export const parseEnergy    = parseEnergyCSV;
export const parseSummary   = parseSummaryCSV;
export const parseTopo      = parseTopologyCSV;
export const parseRL        = parseRLJson;

// ── parseComparison — lit comparison_metrics.csv (données réelles simulation) ─
// Structure: [SUMMARY], [TABLE_3_ENERGY_CONSUMPTION_J], [TABLE_4_NETWORK_LIFETIME],
//            [TABLE_5_PDR_PERCENT], [TABLE_6_END_TO_END_DELAY_ms],
//            [TABLE_7_ALIVE_NODES], [TABLE_8_DEAD_NODES], [TABLE_RL_METRICS]
export function parseComparison(txt) {
  const lines = txt.split("\n");
  const result = {
    summary:      {},
    energyData:   [],   // { time, energy }
    lifetimeData: [],   // { time, alive, dead }
    pdrData:      [],   // { time, pdrRL, pdrNS3 }
    delayData:    [],   // { time, delay }
    aliveData:    [],   // { time, alive }
    deadData:     [],   // { time, dead }
    rlData:       [],   // { time, rlSteps, fedRounds }
  };

  let section = null;
  let headerSkipped = false;

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;

    // Détection de section
    if (line.startsWith("[")) {
      section = line;
      headerSkipped = false;
      continue;
    }

    // Sauter la ligne d'en-tête de chaque section
    if (!headerSkipped) {
      headerSkipped = true;
      continue;
    }

    const p = line.split(",");
    const n = (i) => parseFloat(p[i]) || 0;

    if (section === "[SUMMARY]" && p.length >= 2) {
      const key = p[0].trim();
      const val = p[1]?.trim();
      // Sauter la ligne d'en-tête "Metric,Value,Unit"
      if (key === "Metric" || key === "" || val === undefined) continue;
      const num = parseFloat(val);
      // Normaliser les clés
      const keyMap = {
        "FND (First Node Death)":    "FND",
        "HND (Half Node Death)":     "HND",
        "LND (90% Node Death)":      "LND",
        "Total Rounds":              "TotalRounds",
        "Average PDR RL":            "AveragePDRRL",
        "Average PDR NS-3":          "AveragePDRNS3",
        "Average End-to-End Delay":  "AverageEndToEndDelay",
        "Total Energy Consumed":     "TotalEnergyConsumed",
      };
      const k = keyMap[key] || key;
      result.summary[k] = isNaN(num) ? val : num;
    }
    else if (section === "[TABLE_3_ENERGY_CONSUMPTION_J]" && p.length >= 2) {
      result.energyData.push({ time: n(0), energy: n(1) });
    }
    else if (section === "[TABLE_4_NETWORK_LIFETIME]" && p.length >= 4) {
      result.lifetimeData.push({ time: n(3), alive: n(1), dead: n(2) });
    }
    else if (section === "[TABLE_5_PDR_PERCENT]" && p.length >= 3) {
      result.pdrData.push({ time: n(0), pdrRL: n(1), pdrNS3: n(2) });
    }
    else if (section === "[TABLE_6_END_TO_END_DELAY_ms]" && p.length >= 2) {
      result.delayData.push({ time: n(0), delay: n(1) });
    }
    else if (section === "[TABLE_7_ALIVE_NODES]" && p.length >= 2) {
      result.aliveData.push({ time: n(0), alive: n(1) });
    }
    else if (section === "[TABLE_8_DEAD_NODES]" && p.length >= 2) {
      result.deadData.push({ time: n(0), dead: n(1) });
    }
    else if (section === "[TABLE_RL_METRICS]" && p.length >= 3) {
      result.rlData.push({ time: n(0), rlSteps: n(1), fedRounds: n(2) });
    }
  }

  return result;
}

// ── parseRouting — lit fdqnte_routing.csv ────────────────────────────────────
// Colonnes: Time_s, SrcId, SrcX, SrcY, ClusterId, IsCH, NextHop, HopCount,
//           Delivered, Delay_ms, EnergyDrain_J, PEPMRisk
export function parseRouting(txt) {
  const lines = txt.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  if (lines.length < 2) return null;
  const hdr = lines[0].split(",").map(h => h.trim());
  const g  = (row, k) => { const i = hdr.indexOf(k); return i !== -1 ? row[i]?.trim() : ""; };
  const gf = (row, k) => parseFloat(g(row, k)) || 0;

  const rows = lines.slice(1).map(line => {
    const v = line.split(",");
    return {
      time:      gf(v, "Time_s"),
      srcId:     g(v,  "SrcId"),
      isCH:      g(v,  "IsCH") === "1",
      cluster:   g(v,  "ClusterId"),
      nextHop:   g(v,  "NextHop"),
      delivered: g(v,  "Delivered") === "1",
      delay:     gf(v, "Delay_ms"),
      drain:     gf(v, "EnergyDrain_J"),
      pepm:      gf(v, "PEPMRisk"),
    };
  }).filter(r => r.time >= 0);

  // Série temporelle (buckets 50s)
  const bkts = {};
  rows.forEach(r => {
    const b = Math.floor(r.time / 50) * 50;
    if (!bkts[b]) bkts[b] = { count:0, delivered:0, drain:0, pepm:[], ch:new Set(), src:new Set() };
    bkts[b].count++;
    if (r.delivered) bkts[b].delivered++;
    bkts[b].drain += r.drain;
    bkts[b].pepm.push(r.pepm);
    bkts[b].src.add(r.srcId);
    if (r.isCH) bkts[b].ch.add(r.srcId);
  });

  const timeSeries = Object.keys(bkts).map(t => {
    const b = bkts[t];
    const pm = b.pepm.length ? b.pepm.reduce((s,v)=>s+v,0)/b.pepm.length : 0;
    return {
      time:     +t + 25,
      packets:  b.count,
      pdr:      b.count ? +(b.delivered / b.count * 100).toFixed(2) : 100,
      drain_mJ: +(b.drain * 1000).toFixed(3),
      pepm:     +pm.toFixed(4),
      nCH:      b.ch.size,
      nSrc:     b.src.size,
    };
  }).sort((a, b) => a.time - b.time);

  // Drain par nœud
  const nd = {};
  rows.forEach(r => {
    if (!nd[r.srcId]) nd[r.srcId] = { drain:0, count:0, pepm:[], isCH:false };
    nd[r.srcId].drain += r.drain;
    nd[r.srcId].count++;
    nd[r.srcId].pepm.push(r.pepm);
    if (r.isCH) nd[r.srcId].isCH = true;
  });

  const nodeList = Object.entries(nd).map(([id, n]) => ({
    id,
    drain:  +n.drain.toFixed(4),
    count:  n.count,
    pepm:   n.pepm.length ? +(n.pepm.reduce((s,v)=>s+v,0)/n.pepm.length).toFixed(4) : 0,
    isCH:   n.isCH,
  })).sort((a, b) => b.drain - a.drain);

  const nonDelivered = rows.filter(r => !r.delivered).map(r => ({
    time: r.time, src: r.srcId, cluster: r.cluster, pepm: r.pepm,
  }));

  const total = rows.length;
  const deliv = rows.filter(r => r.delivered).length;
  const dv    = nodeList.map(n => n.drain);

  return {
    timeSeries,
    nodeList,
    nonDelivered,
    top10Drain: nodeList.slice(0, 10),
    summary: {
      totalPackets: total,
      delivered:    deliv,
      pdrGlobal:    +(deliv / total * 100).toFixed(2),
      nonDelivered: total - deliv,
      drainMin:     +Math.min(...dv).toFixed(4),
      drainMax:     +Math.max(...dv).toFixed(4),
      drainMoy:     +(dv.reduce((s,v)=>s+v,0)/dv.length).toFixed(4),
      timeRange:    `${rows[0]?.time}s → ${rows[rows.length-1]?.time}s`,
      nNodes:       nodeList.length,
      nCHActive:    nodeList.filter(n => n.isCH).length,
    },
  };
}
