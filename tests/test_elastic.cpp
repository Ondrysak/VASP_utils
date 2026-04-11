#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

#include "elastic_properties.h"
#include "outcar_file.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build an ElasticTensor from flat values in VASP ordering
// (XX,YY,ZZ,XY,YZ,ZX) for a cubic crystal: only C11, C12, C44 are independent.
// ─────────────────────────────────────────────────────────────────────────────
static ElasticTensor makeCubicTensor(double C11, double C12, double C44) {
    ElasticTensor C{};
    // Diagonal: C11
    C.C[0][0] = C.C[1][1] = C.C[2][2] = C11;
    // Off-diagonal upper 3×3: C12
    C.C[0][1] = C.C[1][0] = C12;
    C.C[0][2] = C.C[2][0] = C12;
    C.C[1][2] = C.C[2][1] = C12;
    // Shear diagonal (VASP ordering: C66=C[3][3], C44=C[4][4], C55=C[5][5])
    C.C[3][3] = C.C[4][4] = C.C[5][5] = C44;
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
// Crystal system mapping
// ─────────────────────────────────────────────────────────────────────────────
TEST(CrystalSystem, SpaceGroupRanges) {
    EXPECT_EQ(crystalSystemFromSpaceGroup(1), CrystalSystem::Triclinic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(2), CrystalSystem::Triclinic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(3), CrystalSystem::Monoclinic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(15), CrystalSystem::Monoclinic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(16), CrystalSystem::Orthorhombic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(74), CrystalSystem::Orthorhombic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(75), CrystalSystem::Tetragonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(142), CrystalSystem::Tetragonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(143), CrystalSystem::Trigonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(167), CrystalSystem::Trigonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(168), CrystalSystem::Hexagonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(194), CrystalSystem::Hexagonal);
    EXPECT_EQ(crystalSystemFromSpaceGroup(195), CrystalSystem::Cubic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(230), CrystalSystem::Cubic);
    EXPECT_EQ(crystalSystemFromSpaceGroup(0), CrystalSystem::Unknown);
    EXPECT_EQ(crystalSystemFromSpaceGroup(231), CrystalSystem::Unknown);
}

// ─────────────────────────────────────────────────────────────────────────────
// Elastic moduli for a cubic crystal
//
// For cubic (C11, C12, C44):
//   B_Voigt  = (C11 + 2*C12) / 3
//   G_Voigt  = (C11 - C12 + 3*C44) / 5
//   B_Reuss  = B_Voigt  (for cubic, Voigt == Reuss for bulk modulus)
//   G_Reuss  = 5*C44*(C11-C12) / (4*C44 + 3*(C11-C12))
// ─────────────────────────────────────────────────────────────────────────────
TEST(ElasticModuli, CubicVoigt) {
    // Hypothetical cubic crystal: C11=500, C12=200, C44=150 GPa
    double C11 = 500.0, C12 = 200.0, C44 = 150.0;
    auto C = makeCubicTensor(C11, C12, C44);
    auto m = computeElasticModuli(C);

    double expected_B_V = (C11 + 2.0 * C12) / 3.0;
    double expected_G_V = (C11 - C12 + 3.0 * C44) / 5.0;

    EXPECT_NEAR(m.B_voigt, expected_B_V, 1e-6);
    EXPECT_NEAR(m.G_voigt, expected_G_V, 1e-6);
}

TEST(ElasticModuli, CubicReussBulk) {
    // For cubic crystals B_Reuss == B_Voigt (exact).
    double C11 = 500.0, C12 = 200.0, C44 = 150.0;
    auto C = makeCubicTensor(C11, C12, C44);
    auto m = computeElasticModuli(C);

    double expected_B = (C11 + 2.0 * C12) / 3.0;
    EXPECT_NEAR(m.B_reuss, expected_B, 1e-4);
}

TEST(ElasticModuli, CubicReussShear) {
    double C11 = 500.0, C12 = 200.0, C44 = 150.0;
    auto C = makeCubicTensor(C11, C12, C44);
    auto m = computeElasticModuli(C);

    double Cs = C11 - C12;
    double expected_G_R = 5.0 * C44 * Cs / (4.0 * C44 + 3.0 * Cs);
    EXPECT_NEAR(m.G_reuss, expected_G_R, 1e-4);
}

TEST(ElasticModuli, DerivedQuantities) {
    double C11 = 500.0, C12 = 200.0, C44 = 150.0;
    auto C = makeCubicTensor(C11, C12, C44);
    auto m = computeElasticModuli(C);

    // E = 9BG/(3B+G)
    double E_expected = 9.0 * m.B_vrh * m.G_vrh / (3.0 * m.B_vrh + m.G_vrh);
    EXPECT_NEAR(m.E_vrh, E_expected, 1e-4);

    // ν = (3B-2G)/(2(3B+G))
    double nu_expected = (3.0 * m.B_vrh - 2.0 * m.G_vrh) / (2.0 * (3.0 * m.B_vrh + m.G_vrh));
    EXPECT_NEAR(m.nu_vrh, nu_expected, 1e-4);

    // Pugh = B/G
    EXPECT_NEAR(m.pugh_ratio, m.B_vrh / m.G_vrh, 1e-4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Born stability — cubic
// ─────────────────────────────────────────────────────────────────────────────
TEST(Stability, CubicStable) {
    // C11=500, C12=200, C44=150: all Born criteria satisfied.
    auto C = makeCubicTensor(500.0, 200.0, 150.0);
    auto r = checkMechanicalStability(C, CrystalSystem::Cubic);
    EXPECT_TRUE(r.is_stable);
    EXPECT_TRUE(r.failed_criteria.empty());
}

TEST(Stability, CubicUnstable_NegativeC44) {
    auto C = makeCubicTensor(500.0, 200.0, -10.0);
    auto r = checkMechanicalStability(C, CrystalSystem::Cubic);
    EXPECT_FALSE(r.is_stable);
    // Exactly the C44 criterion should fail.
    ASSERT_EQ(r.failed_criteria.size(), 1u);
    EXPECT_NE(r.failed_criteria[0].find("C44"), std::string::npos);
}

TEST(Stability, CubicUnstable_C11minusC12) {
    // C11 < C12 → violates C11-C12 > 0
    auto C = makeCubicTensor(100.0, 200.0, 150.0);
    auto r = checkMechanicalStability(C, CrystalSystem::Cubic);
    EXPECT_FALSE(r.is_stable);
    bool found = false;
    for (const auto& crit : r.failed_criteria)
        if (crit.find("C11 - C12") != std::string::npos)
            found = true;
    EXPECT_TRUE(found);
}

// ─────────────────────────────────────────────────────────────────────────────
// OUTCAR parser
// ─────────────────────────────────────────────────────────────────────────────

// Write a minimal fake OUTCAR and verify the parser extracts the tensor.
TEST(OUTCARParser, ParsesSingleBlock) {
    // Write a temporary file that looks like a minimal VASP OUTCAR snippet.
    const std::string tmpfile = "/tmp/test_outcar_elastic.tmp";
    {
        std::ofstream f(tmpfile);
        f << " some preamble\n";
        f << " SYMMETRIZED ELASTIC MODULI (kBar)\n";
        f << " Direction    XX          YY          ZZ          XY          YZ          ZX\n";
        f << " -----------------------------------------------------------------------\n";
        // 6 rows; values in kBar (will be divided by 10 to get GPa)
        f << " XX     5000.000    2000.000    2000.000       0.000       0.000       0.000\n";
        f << " YY     2000.000    5000.000    2000.000       0.000       0.000       0.000\n";
        f << " ZZ     2000.000    2000.000    5000.000       0.000       0.000       0.000\n";
        f << " XY        0.000       0.000       0.000    1500.000       0.000       0.000\n";
        f << " YZ        0.000       0.000       0.000       0.000    1500.000       0.000\n";
        f << " ZX        0.000       0.000       0.000       0.000       0.000    1500.000\n";
        f << " -----------------------------------------------------------------------\n";
    }

    OUTCAR outcar;
    ASSERT_TRUE(outcar.readOUTCAR(tmpfile));
    EXPECT_TRUE(outcar.has_elastic_tensor);

    // kBar → GPa conversion: 5000 kBar = 500 GPa
    EXPECT_NEAR(outcar.elastic_tensor.C[0][0], 500.0, 1e-6);
    EXPECT_NEAR(outcar.elastic_tensor.C[0][1], 200.0, 1e-6);
    // C44 in standard Voigt = C[4][4] (YZ,YZ row in VASP)
    EXPECT_NEAR(outcar.elastic_tensor.C[4][4], 150.0, 1e-6);
}

TEST(OUTCARParser, UsesLastBlock) {
    // Two blocks: second one has different values; parser must return the last.
    const std::string tmpfile = "/tmp/test_outcar_elastic2.tmp";
    {
        std::ofstream f(tmpfile);
        auto writeBlock = [&](double diag, double offdiag, double shear) {
            f << " SYMMETRIZED ELASTIC MODULI (kBar)\n";
            f << " Direction    XX          YY          ZZ          XY          YZ          ZX\n";
            f << " -----------------------------------------------------------------------\n";
            f << " XX " << diag << " " << offdiag << " " << offdiag << " 0.0 0.0 0.0\n";
            f << " YY " << offdiag << " " << diag << " " << offdiag << " 0.0 0.0 0.0\n";
            f << " ZZ " << offdiag << " " << offdiag << " " << diag << " 0.0 0.0 0.0\n";
            f << " XY 0.0 0.0 0.0 " << shear << " 0.0 0.0\n";
            f << " YZ 0.0 0.0 0.0 0.0 " << shear << " 0.0\n";
            f << " ZX 0.0 0.0 0.0 0.0 0.0 " << shear << "\n";
        };
        writeBlock(1000.0, 500.0, 300.0);  // first (stale)
        writeBlock(9000.0, 3000.0, 2000.0);  // second (final)
    }

    OUTCAR outcar;
    ASSERT_TRUE(outcar.readOUTCAR(tmpfile));
    // Should reflect the second block: 9000 kBar = 900 GPa
    EXPECT_NEAR(outcar.elastic_tensor.C[0][0], 900.0, 1e-4);
}

TEST(OUTCARParser, FailsOnMissingFile) {
    OUTCAR outcar;
    EXPECT_FALSE(outcar.readOUTCAR("/nonexistent/path/OUTCAR"));
    EXPECT_FALSE(outcar.has_elastic_tensor);
}

TEST(OUTCARParser, FailsWithNoElasticBlock) {
    const std::string tmpfile = "/tmp/test_outcar_noblock.tmp";
    {
        std::ofstream f(tmpfile);
        f << " This OUTCAR has no elastic tensor.\n";
        f << " TOTEN =  -123.456 eV\n";
    }
    OUTCAR outcar;
    EXPECT_FALSE(outcar.readOUTCAR(tmpfile));
    EXPECT_FALSE(outcar.has_elastic_tensor);
}
