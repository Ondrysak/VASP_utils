#ifndef ELASTIC_PROPERTIES_H_INCLUDED
#define ELASTIC_PROPERTIES_H_INCLUDED

#include <string>
#include <vector>

struct ElasticTensor;

/**
 * @brief Crystal system inferred from the spglib space group number.
 */
enum class CrystalSystem {
    Triclinic,     ///< Space groups   1–2
    Monoclinic,    ///< Space groups   3–15
    Orthorhombic,  ///< Space groups  16–74
    Tetragonal,    ///< Space groups  75–142
    Trigonal,      ///< Space groups 143–167
    Hexagonal,     ///< Space groups 168–194
    Cubic,         ///< Space groups 195–230
    Unknown        ///< Space group number out of range
};

/**
 * @brief Voigt, Reuss, and Hill (VRH) elastic moduli.
 *
 * Voigt is an upper bound; Reuss is a lower bound; Hill is their arithmetic
 * mean and the most widely quoted estimate.
 */
struct ElasticModuli {
    double B_voigt{};    ///< Voigt bulk modulus (GPa).
    double B_reuss{};    ///< Reuss bulk modulus (GPa).
    double B_vrh{};      ///< Hill bulk modulus (GPa).
    double G_voigt{};    ///< Voigt shear modulus (GPa).
    double G_reuss{};    ///< Reuss shear modulus (GPa).
    double G_vrh{};      ///< Hill shear modulus (GPa).
    double E_vrh{};      ///< Young's modulus (Hill), GPa.  E = 9BG/(3B+G).
    double nu_vrh{};     ///< Poisson's ratio (Hill).       ν = (3B−2G)/(2(3B+G)).
    double pugh_ratio{}; ///< Pugh's ratio B_vrh/G_vrh.  >1.75 → ductile.
    bool is_ductile{};   ///< True when Pugh's ratio exceeds 1.75.
};

/**
 * @brief Result of a Born mechanical stability check.
 */
struct StabilityResult {
    bool is_stable{false};                     ///< True if all criteria are satisfied.
    std::vector<std::string> failed_criteria;  ///< Human-readable list of violated conditions.
};

/**
 * @brief Map a spglib space group number (1–230) to its crystal system.
 */
CrystalSystem crystalSystemFromSpaceGroup(int spacegroup_number);

/** @brief Return the name of a crystal system as a human-readable string. */
const char* crystalSystemName(CrystalSystem cs);

/**
 * @brief Compute Voigt, Reuss, and Hill elastic moduli from a 6×6 elastic tensor.
 *
 * Reuss averages require inverting the compliance matrix; LAPACKE is used
 * internally.  If inversion fails the Reuss values are set to zero and a
 * warning is printed.
 *
 * Index ordering of @p C matches VASP OUTCAR: 0=XX,1=YY,2=ZZ,3=XY,4=YZ,5=ZX.
 *
 * @param C  Elastic tensor in GPa.
 * @return   Populated ElasticModuli struct.
 */
ElasticModuli computeElasticModuli(const ElasticTensor& C);

/**
 * @brief Check Born mechanical stability criteria for the given crystal system.
 *
 * VASP OUTCAR index ordering: 0=XX,1=YY,2=ZZ,3=XY,4=YZ,5=ZX.
 * Crystallographic mapping: C44→C[4][4], C55→C[5][5], C66→C[3][3].
 *
 * For triclinic and monoclinic, positive-definiteness of the full 6×6 matrix
 * is checked via eigenvalue analysis.  For higher-symmetry systems the
 * standard Born–Huang criteria are applied explicitly.
 *
 * @param C   Elastic tensor in GPa.
 * @param cs  Crystal system.
 * @return    StabilityResult.
 */
StabilityResult checkMechanicalStability(const ElasticTensor& C, CrystalSystem cs);

/**
 * @brief Print a formatted elastic analysis report to stdout.
 *
 * @param C         Elastic tensor (printed as a 6×6 table).
 * @param moduli    VRH averages.
 * @param stability Born stability result.
 * @param cs        Crystal system (used for labelling).
 */
void printElasticReport(const ElasticTensor& C, const ElasticModuli& moduli, const StabilityResult& stability,
                        CrystalSystem cs);

#endif  // ELASTIC_PROPERTIES_H_INCLUDED
