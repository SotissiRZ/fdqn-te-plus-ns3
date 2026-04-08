# =============================================================================
# CMakeLists.txt — FDQN-TE+ Module NS-3
#
# Intégration dans NS-3 :
#   1. Copier ce dossier dans <ns3-root>/contrib/fdqn-te-plus/
#   2. Lancer : ./ns3 configure --enable-examples
#   3. Construire : ./ns3 build fdqn-te-plus
#   4. Exécuter  : ./ns3 run "fdqn_te_plus --nNodes=300 --seed=42"
# =============================================================================

cmake_minimum_required(VERSION 3.13)

# ── Déclaration du module NS-3 ─────────────────────────────────────────────────
build_lib(
    LIBNAME fdqn-te-plus
    SOURCE_FILES
        model/ifo/ifo-clustering.cc
        model/addqn/addqn-routing.cc
        model/pepm/pepm-energy.cc
        model/federated/fedmeta-drl.cc
        utils/fdqnte-stats.cc
    HEADER_FILES
        model/ifo/ifo-clustering.h
        model/addqn/addqn-routing.h
        model/pepm/pepm-energy.h
        model/federated/fedmeta-drl.h
        utils/fdqnte-stats.h
    LIBRARIES_TO_LINK
        ${libcore}
        ${libnetwork}
        ${libinternet}
        ${libmobility}
        ${libenergy}
        ${libwifi}
        ${libapplications}
        ${libflow-monitor}
        ${libnetanim}
)

# ── Script de simulation principal ────────────────────────────────────────────
build_exec(
    EXECNAME    fdqn_te_plus
    SOURCE_FILES scratch/fdqn_te_plus.cc
    LIBRARIES_TO_LINK
        ${libfdqn-te-plus}
        ${libcore}
        ${libnetwork}
        ${libinternet}
        ${libmobility}
        ${libenergy}
        ${libwifi}
        ${libapplications}
        ${libflow-monitor}
        ${libnetanim}
)

# ── Liaison Python (ns3-ai ou socket bridge) ──────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(Python3_FOUND)
    message(STATUS "Python3 détecté : ${Python3_EXECUTABLE}")
    target_include_directories(${libfdqn-te-plus} PRIVATE
        ${Python3_INCLUDE_DIRS}
    )
    target_link_libraries(${libfdqn-te-plus}
        ${Python3_LIBRARIES}
    )
else()
    message(WARNING "Python3 non trouvé — bridge Python désactivé")
endif()
