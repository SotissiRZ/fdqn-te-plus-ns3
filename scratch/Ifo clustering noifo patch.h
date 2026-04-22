/* =============================================================================
 * ifo_clustering_noIFO_patch.h
 *
 * PATCH D'ABLATION — DQN_noIFO
 *
 * Ajouter cette méthode publique à la classe IFOClustering dans ifo_clustering.h.
 *
 * EMPLACEMENT : dans la section publique de IFOClustering, après GetClusters().
 *
 * RÔLE :
 *   Permet à fdqn_te_plus_noIFO.cc d'injecter un clustering LEACH probabiliste
 *   (calculé en C++ sans passer par ifo.Run()) dans le conteneur interne de
 *   l'objet IFO, afin que les méthodes suivantes continuent de fonctionner
 *   normalement sans modification :
 *     - GetClusters()              → SendTopology() vers Python
 *     - GetStats()                 → logs + CSV énergie (NClusters, membersMin/Max/Mean)
 *     - GetRound()                 → colonne IFORound dans energy CSV
 *     - TriggerProactiveRecluster → recluster proactif PEPM (conservé en noIFO)
 *
 * COHÉRENCE :
 *   GetRound() retourne 0 en noIFO (aucun appel à Run()) — c'est correct et
 *   mesurable : la colonne IFORound sera 0 dans tous les CSV, ce qui distingue
 *   clairement cette variante du modèle complet.
 * ============================================================================= */

// ── À COPIER dans ifo_clustering.h, section public de IFOClustering ──────────

    /**
     * Injecte un clustering calculé externalement (ex. LEACH probabiliste)
     * dans le conteneur interne de l'objet IFO.
     *
     * Après cet appel, GetClusters(), GetStats() et TriggerProactiveRecluster()
     * opèrent sur ce clustering — exactement comme après un appel à Run().
     * GetRound() reste inchangé (ne s'incrémente pas ici, contrairement à Run()).
     *
     * @param clusters  Vecteur de ClusterInfo construit par l'appelant.
     *                  Chaque ClusterInfo doit avoir chId + members remplis.
     */
    void SetClustersFromExternal(const std::vector<ClusterInfo>& clusters) {
        m_clusters = clusters;   // remplace l'état interne
        // m_round intentionnellement NON incrémenté → IFORound=0 en noIFO
    }

// ── FIN DU PATCH ──────────────────────────────────────────────────────────────

/*
 * VÉRIFICATION RAPIDE après ajout :
 *   grep "SetClustersFromExternal\|m_clusters" ifo_clustering.h
 *   → doit trouver les deux lignes ci-dessus + la déclaration de m_clusters
 *
 * Si m_clusters est déclaré sous un autre nom (ex. m_clusterList, clusters_),
 * remplacer "m_clusters" par le nom exact dans la ligne "m_clusters = clusters;"
 *
 * EXEMPLE de déclaration attendue dans la section private :
 *   std::vector<ClusterInfo> m_clusters;
 */
