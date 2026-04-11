#include "elastic_properties.h"

#include <lapacke.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "outcar_file.h"

// ─────────────────────────────────────────────────────────────────────────────
// Crystal system helpers
// ─────────────────────────────────────────────────────────────────────────────

CrystalSystem crystalSystemFromSpaceGroup(int sg) {
    if (sg >= 1 && sg <= 2) return CrystalSystem::Triclinic;
    if (sg >= 3 && sg <= 15) return CrystalSystem::Monoclinic;
    if (sg >= 16 && sg <= 74) return CrystalSystem::Orthorhombic;
    if (sg >= 75 && sg <= 142) return CrystalSystem::Tetragonal;
    if (sg >= 143 && sg <= 167) return CrystalSystem::Trigonal;
    if (sg >= 168 && sg <= 194) return CrystalSystem::Hexagonal;
    if (sg >= 195 && sg <= 230) return CrystalSystem::Cubic;
    return CrystalSystem::Unknown;
}

const char* crystalSystemName(CrystalSystem cs) {
    switch (cs) {
        case CrystalSystem::Triclinic: return "Triclinic";
        case CrystalSystem::Monoclinic: return "Monoclinic";
        case CrystalSystem::Orthorhombic: return "Orthorhombic";
        case CrystalSystem::Tetragonal: return "Tetragonal";
        case CrystalSystem::Trigonal: return "Trigonal";
        case CrystalSystem::Hexagonal: return "Hexagonal";
        case CrystalSystem::Cubic: return "Cubic";
        default: return "Unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Invert a 6×6 matrix in-place using LAPACKE (LU factorisation + inversion).
// Returns true on success; false if the matrix is singular.
bool invert6x6(double M[6][6]) {
    // LAPACKE expects column-major storage; the elastic tensor is symmetric so
    // row-major == column-major (M^T == M for a symmetric matrix).
    lapack_int n = 6;
    lapack_int lda = 6;
    std::array<lapack_int, 6> ipiv;

    // LU factorisation
    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, &M[0][0], lda, ipiv.data());
    if (info != 0)
        return false;

    // Inversion
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, &M[0][0], lda, ipiv.data());
    return (info == 0);
}

// Compute all eigenvalues of a 6×6 symmetric matrix using LAPACKE dsyev.
// Returns false if the driver fails.
bool eigenvalues6x6(const double M[6][6], double eigs[6]) {
    double A[6][6];
    std::memcpy(A, M, sizeof(A));

    lapack_int n = 6;
    lapack_int lda = 6;
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, &A[0][0], lda, eigs);
    return (info == 0);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Voigt / Reuss / Hill averages
// ─────────────────────────────────────────────────────────────────────────────

// VASP OUTCAR index ordering: 0=XX, 1=YY, 2=ZZ, 3=XY, 4=YZ, 5=ZX.
// Mapping to standard Voigt crystallographic labels:
//   C11=C[0][0]  C22=C[1][1]  C33=C[2][2]
//   C12=C[0][1]  C13=C[0][2]  C23=C[1][2]
//   C44=C[4][4] (YZ,YZ)  C55=C[5][5] (ZX,ZX)  C66=C[3][3] (XY,XY)

ElasticModuli computeElasticModuli(const ElasticTensor& C) {
    const auto& c = C.C;

    // ── Voigt bounds ──────────────────────────────────────────────────────────
    // Hill (1952), Eq. (3) and (6).
    double B_V = (c[0][0] + c[1][1] + c[2][2] + 2.0 * (c[0][1] + c[0][2] + c[1][2])) / 9.0;

    // Shear diagonal elements in standard-Voigt sense:
    //   C44(YZ,YZ)=c[4][4], C55(ZX,ZX)=c[5][5], C66(XY,XY)=c[3][3]
    double G_V = (c[0][0] + c[1][1] + c[2][2] - c[0][1] - c[0][2] - c[1][2] +
                  3.0 * (c[3][3] + c[4][4] + c[5][5])) /
                 15.0;

    // ── Reuss bounds ──────────────────────────────────────────────────────────
    // Invert C to get compliance matrix S.
    double S[6][6];
    std::memcpy(S, c, sizeof(S));

    double B_R = 0.0, G_R = 0.0;
    bool reuss_ok = invert6x6(S);

    if (reuss_ok) {
        B_R = 1.0 / (S[0][0] + S[1][1] + S[2][2] + 2.0 * (S[0][1] + S[0][2] + S[1][2]));
        // Shear compliances: S44(YZ,YZ)=S[4][4], S55(ZX,ZX)=S[5][5], S66(XY,XY)=S[3][3]
        G_R = 15.0 / (4.0 * (S[0][0] + S[1][1] + S[2][2]) - 4.0 * (S[0][1] + S[0][2] + S[1][2]) +
                      3.0 * (S[3][3] + S[4][4] + S[5][5]));
    } else {
        std::cerr << "Warning: elastic tensor is singular; Reuss moduli set to zero.\n";
    }

    // ── Hill averages ─────────────────────────────────────────────────────────
    double B_vrh = (B_V + B_R) / 2.0;
    double G_vrh = (G_V + G_R) / 2.0;

    ElasticModuli m;
    m.B_voigt = B_V;
    m.B_reuss = B_R;
    m.B_vrh = B_vrh;
    m.G_voigt = G_V;
    m.G_reuss = G_R;
    m.G_vrh = G_vrh;

    if (G_vrh > 1e-12 && (3.0 * B_vrh + G_vrh) > 1e-12) {
        m.E_vrh = 9.0 * B_vrh * G_vrh / (3.0 * B_vrh + G_vrh);
        m.nu_vrh = (3.0 * B_vrh - 2.0 * G_vrh) / (2.0 * (3.0 * B_vrh + G_vrh));
        m.pugh_ratio = B_vrh / G_vrh;
        m.is_ductile = (m.pugh_ratio > 1.75);
    }

    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Born mechanical stability criteria
// ─────────────────────────────────────────────────────────────────────────────

// Convenience: test a single scalar condition and record failure.
static void addCheck(StabilityResult& r, bool ok, const char* label) {
    if (!ok)
        r.failed_criteria.push_back(label);
}

StabilityResult checkMechanicalStability(const ElasticTensor& C, CrystalSystem cs) {
    StabilityResult result;
    const auto& c = C.C;

    // Short aliases for the most frequently used elements (VASP ordering).
    const double C11 = c[0][0], C22 = c[1][1], C33 = c[2][2];
    const double C12 = c[0][1], C13 = c[0][2], C23 = c[1][2];
    // Standard-Voigt shear diagonal: C44=YZ, C55=ZX, C66=XY
    const double C44 = c[4][4], C55 = c[5][5], C66 = c[3][3];

    switch (cs) {
        // ── Cubic ─────────────────────────────────────────────────────────────
        // Born & Huang (1954); Mouhat & Coudert, PRB 90, 224104 (2014) Table I.
        case CrystalSystem::Cubic:
            addCheck(result, C11 - C12 > 0.0, "C11 - C12 > 0");
            addCheck(result, C11 + 2.0 * C12 > 0.0, "C11 + 2*C12 > 0");
            addCheck(result, C44 > 0.0, "C44 > 0");
            break;

        // ── Hexagonal ────────────────────────────────────────────────────────
        // Mouhat & Coudert Table I.
        case CrystalSystem::Hexagonal:
        case CrystalSystem::Trigonal:
            addCheck(result, C11 - C12 > 0.0, "C11 - C12 > 0");
            addCheck(result, C44 > 0.0, "C44 > 0");
            addCheck(result, 2.0 * C13 * C13 < C33 * (C11 + C12), "2*C13^2 < C33*(C11+C12)");
            break;

        // ── Tetragonal ───────────────────────────────────────────────────────
        // Mouhat & Coudert Table I (tetragonal classes I & II).
        case CrystalSystem::Tetragonal:
            addCheck(result, C11 - C12 > 0.0, "C11 - C12 > 0");
            addCheck(result, C11 + C33 - 2.0 * C13 > 0.0, "C11 + C33 - 2*C13 > 0");
            addCheck(result, 2.0 * C13 * C13 < C33 * (C11 + C12), "2*C13^2 < C33*(C11+C12)");
            addCheck(result, C44 > 0.0, "C44 > 0");
            addCheck(result, C66 > 0.0, "C66 > 0");
            break;

        // ── Orthorhombic ─────────────────────────────────────────────────────
        // Mouhat & Coudert Table I.
        case CrystalSystem::Orthorhombic: {
            addCheck(result, C11 > 0.0, "C11 > 0");
            addCheck(result, C22 > 0.0, "C22 > 0");
            addCheck(result, C33 > 0.0, "C33 > 0");
            addCheck(result, C44 > 0.0, "C44 > 0");
            addCheck(result, C55 > 0.0, "C55 > 0");
            addCheck(result, C66 > 0.0, "C66 > 0");
            addCheck(result, C11 + C22 - 2.0 * C12 > 0.0, "C11 + C22 - 2*C12 > 0");
            addCheck(result, C11 + C33 - 2.0 * C13 > 0.0, "C11 + C33 - 2*C13 > 0");
            addCheck(result, C22 + C33 - 2.0 * C23 > 0.0, "C22 + C33 - 2*C23 > 0");
            // det of upper-left 3×3 submatrix > 0
            double det3 = C11 * (C22 * C33 - C23 * C23) - C12 * (C12 * C33 - C23 * C13) +
                          C13 * (C12 * C23 - C22 * C13);
            addCheck(result, det3 > 0.0, "det(C_3x3) > 0");
            break;
        }

        // ── Monoclinic / Triclinic: positive-definiteness via eigenvalues ─────
        // The simplest universally correct criterion: all eigenvalues must be > 0.
        case CrystalSystem::Monoclinic:
        case CrystalSystem::Triclinic:
        case CrystalSystem::Unknown:
        default: {
            double eigs[6];
            if (eigenvalues6x6(c, eigs)) {
                for (int i = 0; i < 6; ++i) {
                    if (eigs[i] <= 0.0) {
                        std::ostringstream oss;
                        oss << "eigenvalue[" << i << "] = " << eigs[i] << " <= 0";
                        result.failed_criteria.push_back(oss.str());
                    }
                }
            } else {
                result.failed_criteria.push_back("eigenvalue computation failed");
            }
            break;
        }
    }

    result.is_stable = result.failed_criteria.empty();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Report formatting
// ─────────────────────────────────────────────────────────────────────────────

void printElasticReport(const ElasticTensor& C, const ElasticModuli& m, const StabilityResult& stab,
                        CrystalSystem cs) {
    const auto& c = C.C;

    // ── Elastic tensor ────────────────────────────────────────────────────────
    std::cout << "=== Elastic Tensor (GPa) ===\n";
    std::cout << "         XX        YY        ZZ        XY        YZ        ZX\n";
    const char* row_labels[] = {"XX", "YY", "ZZ", "XY", "YZ", "ZX"};
    for (int i = 0; i < 6; ++i) {
        std::cout << " " << row_labels[i];
        for (int j = 0; j < 6; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << c[i][j];
        std::cout << "\n";
    }
    std::cout << "\n";

    // ── Crystal system ────────────────────────────────────────────────────────
    std::cout << "Crystal system: " << crystalSystemName(cs) << "\n\n";

    // ── Mechanical stability ──────────────────────────────────────────────────
    std::cout << "=== Mechanical Stability (Born–Huang criteria) ===\n";
    if (stab.is_stable) {
        std::cout << "Status: STABLE — all criteria satisfied.\n";
    } else {
        std::cout << "Status: UNSTABLE — the following criteria are violated:\n";
        for (const auto& crit : stab.failed_criteria)
            std::cout << "  ✗  " << crit << "\n";
    }
    std::cout << "\n";

    // ── Elastic moduli table ──────────────────────────────────────────────────
    std::cout << "=== Elastic Moduli (GPa) ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "                     Voigt        Reuss         Hill\n";
    std::cout << " Bulk modulus    " << std::setw(12) << m.B_voigt << std::setw(13) << m.B_reuss << std::setw(13)
              << m.B_vrh << "\n";
    std::cout << " Shear modulus   " << std::setw(12) << m.G_voigt << std::setw(13) << m.G_reuss << std::setw(13)
              << m.G_vrh << "\n";
    std::cout << "\n";

    std::cout << "=== Derived Properties (Hill averages) ===\n";
    std::cout << " Young's modulus E   = " << std::setw(10) << m.E_vrh << " GPa\n";
    std::cout << " Poisson's ratio ν   = " << std::setw(10) << m.nu_vrh << "\n";
    std::cout << " Pugh's ratio  B/G   = " << std::setw(10) << m.pugh_ratio;
    std::cout << "  (" << (m.is_ductile ? "ductile" : "brittle") << ", threshold 1.75)\n";
}
