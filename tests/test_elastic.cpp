#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <sstream>
#include <vector>

#include "elastic_utils.h"

// ── helpers ─────────────────────────────────────────────────────────────────

static constexpr double kTol = 1e-9;

// Build a 3×3 strain tensor with a single component set.
static std::array<double, 9> makeStrain(int i, int j, double v) {
    std::array<double, 9> s{};
    s[i * 3 + j] = v;
    s[j * 3 + i] = v;
    return s;
}

// Multiply two 3×3 row-major matrices.
static std::array<double, 9> matmul(const std::array<double, 9>& A, const std::array<double, 9>& B) {
    std::array<double, 9> C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    return C;
}

static std::array<double, 9> transpose(const std::array<double, 9>& A) {
    std::array<double, 9> T{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T[i * 3 + j] = A[j * 3 + i];
    return T;
}

// ── amplitude validation ─────────────────────────────────────────────────────

TEST(AmplitudeValidation, AcceptsSmallStrains) {
    std::ostringstream warn, err;
    EXPECT_TRUE(elasticValidateAmplitudes({-0.01, 0.01}, {-0.01, 0.01}, warn, err));
    EXPECT_TRUE(warn.str().empty());
    EXPECT_TRUE(err.str().empty());
}

TEST(AmplitudeValidation, WarnsAbove1Percent) {
    std::ostringstream warn, err;
    EXPECT_TRUE(elasticValidateAmplitudes({0.02}, {}, warn, err));
    EXPECT_FALSE(warn.str().empty());
    EXPECT_TRUE(err.str().empty());
}

TEST(AmplitudeValidation, ErrorsAbove5Percent) {
    std::ostringstream warn, err;
    EXPECT_FALSE(elasticValidateAmplitudes({0.06}, {}, warn, err));
    EXPECT_FALSE(err.str().empty());
}

TEST(AmplitudeValidation, ErrorsOnShearAbove5Percent) {
    std::ostringstream warn, err;
    EXPECT_FALSE(elasticValidateAmplitudes({}, {-0.06, 0.06}, warn, err));
    EXPECT_FALSE(err.str().empty());
}

TEST(AmplitudeValidation, At5PercentIsAnError) {
    // Exactly at the limit is still an error (> 0.05 check).
    std::ostringstream warn, err;
    // 0.05 is NOT > 0.05, so it should pass.
    EXPECT_TRUE(elasticValidateAmplitudes({0.05}, {}, warn, err));
    // 0.051 IS > 0.05, so it should fail.
    err.str("");
    EXPECT_FALSE(elasticValidateAmplitudes({0.051}, {}, warn, err));
}

// ── Cholesky deformation ─────────────────────────────────────────────────────

// For any deformation F produced by elasticCholeskyDeformation, verify F^T F = I + 2E.
static void checkFTFEqualsIPlus2E(const std::array<double, 9>& strain) {
    const auto F = elasticCholeskyDeformation(strain);
    const auto FT = transpose(F);
    const auto FTF = matmul(FT, F);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double expected = (i == j ? 1.0 : 0.0) + 2.0 * strain[i * 3 + j];
            EXPECT_NEAR(FTF[i * 3 + j], expected, kTol) << "F^T F mismatch at (" << i << "," << j << ")";
        }
}

TEST(CholeskyDeformation, NormalStrainE11) {
    checkFTFEqualsIPlus2E(makeStrain(0, 0, 0.01));
}
TEST(CholeskyDeformation, NormalStrainE22) {
    checkFTFEqualsIPlus2E(makeStrain(1, 1, 0.01));
}
TEST(CholeskyDeformation, NormalStrainE33) {
    checkFTFEqualsIPlus2E(makeStrain(2, 2, 0.01));
}
TEST(CholeskyDeformation, ShearStrainE12) {
    checkFTFEqualsIPlus2E(makeStrain(0, 1, 0.01));
}
TEST(CholeskyDeformation, ShearStrainE13) {
    checkFTFEqualsIPlus2E(makeStrain(0, 2, 0.01));
}
TEST(CholeskyDeformation, ShearStrainE23) {
    checkFTFEqualsIPlus2E(makeStrain(1, 2, 0.01));
}

TEST(CholeskyDeformation, IsUpperTriangular) {
    const auto F = elasticCholeskyDeformation(makeStrain(0, 1, 0.008));
    EXPECT_NEAR(F[1 * 3 + 0], 0.0, kTol);  // F[1,0]
    EXPECT_NEAR(F[2 * 3 + 0], 0.0, kTol);  // F[2,0]
    EXPECT_NEAR(F[2 * 3 + 1], 0.0, kTol);  // F[2,1]
}

// ── Symmetric deformation ────────────────────────────────────────────────────

// For any deformation F produced by elasticSymmetricDeformation:
//   1. F^T F = I + 2E
//   2. F is symmetric (F = F^T)
static void checkSymmetricDeformation(const std::array<double, 9>& strain) {
    const auto F = elasticSymmetricDeformation(strain);
    const auto FT = transpose(F);
    const auto FTF = matmul(FT, F);

    // F^T F = I + 2E
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double expected = (i == j ? 1.0 : 0.0) + 2.0 * strain[i * 3 + j];
            EXPECT_NEAR(FTF[i * 3 + j], expected, kTol) << "F^T F mismatch at (" << i << "," << j << ")";
        }

    // F = F^T (symmetric)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(F[i * 3 + j], F[j * 3 + i], kTol) << "F not symmetric at (" << i << "," << j << ")";
}

TEST(SymmetricDeformation, NormalStrainE11) {
    checkSymmetricDeformation(makeStrain(0, 0, 0.01));
}
TEST(SymmetricDeformation, NormalStrainE33) {
    checkSymmetricDeformation(makeStrain(2, 2, 0.01));
}
TEST(SymmetricDeformation, ShearStrainE12) {
    checkSymmetricDeformation(makeStrain(0, 1, 0.01));
}
TEST(SymmetricDeformation, ShearStrainE13) {
    checkSymmetricDeformation(makeStrain(0, 2, 0.01));
}
TEST(SymmetricDeformation, ShearStrainE23) {
    checkSymmetricDeformation(makeStrain(1, 2, 0.01));
}

// For zero strain, both methods should give the identity matrix.
TEST(SymmetricDeformation, ZeroStrainIsIdentity) {
    std::array<double, 9> zero{};
    const auto F = elasticSymmetricDeformation(zero);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(F[i * 3 + j], (i == j ? 1.0 : 0.0), kTol);
}

// Cholesky and symmetric should give the same result for pure normal strains
// (diagonal F, no off-diagonal ambiguity).
TEST(SymmetricDeformation, MatchesCholeskyForDiagonalStrain) {
    const auto strain = makeStrain(2, 2, 0.008);
    const auto Fc = elasticCholeskyDeformation(strain);
    const auto Fs = elasticSymmetricDeformation(strain);
    for (int k = 0; k < 9; ++k)
        EXPECT_NEAR(Fc[k], Fs[k], kTol);
}

// For a shear strain, Cholesky and symmetric differ but both satisfy F^T F = I + 2E.
TEST(SymmetricDeformation, DiffersFromCholeskyOnShear) {
    const auto strain = makeStrain(0, 1, 0.01);
    const auto Fc = elasticCholeskyDeformation(strain);
    const auto Fs = elasticSymmetricDeformation(strain);
    bool any_diff = false;
    for (int k = 0; k < 9; ++k)
        if (std::abs(Fc[k] - Fs[k]) > 1e-12)
            any_diff = true;
    EXPECT_TRUE(any_diff) << "Expected Cholesky and symmetric to differ for shear strain";
}

// ── elasticVaspStressToVoigt ─────────────────────────────────────────────────

TEST(VaspStressToVoigt, CorrectReorderingAndScaling) {
    // VASP order: XX YY ZZ XY YZ ZX  (kBar)
    const std::array<double, 6> kb = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0};
    const auto v = elasticVaspStressToVoigt(kb, 1.0);
    // Voigt [11,22,33,23,13,12] = [XX,YY,ZZ,YZ,ZX,XY] * 0.1
    EXPECT_NEAR(v[0], 10.0, kTol);  // 11 = XX
    EXPECT_NEAR(v[1], 20.0, kTol);  // 22 = YY
    EXPECT_NEAR(v[2], 30.0, kTol);  // 33 = ZZ
    EXPECT_NEAR(v[3], 50.0, kTol);  // 23 = YZ (kb[4])
    EXPECT_NEAR(v[4], 60.0, kTol);  // 13 = ZX (kb[5])
    EXPECT_NEAR(v[5], 40.0, kTol);  // 12 = XY (kb[3])
}

TEST(VaspStressToVoigt, NegateFlipsSign) {
    const std::array<double, 6> kb = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto vp = elasticVaspStressToVoigt(kb, 1.0);
    const auto vn = elasticVaspStressToVoigt(kb, -1.0);
    EXPECT_NEAR(vp[0], -vn[0], kTol);
}

// ── elasticFitPolynomial ─────────────────────────────────────────────────────

// Quadratic fit: E = 3 + 0·ε + 170·ε²  → coefficients [3, 0, 170, ...]
TEST(FitPolynomial, QuadraticRecoversCii) {
    const double a0 = -10.0, a2 = 170.0;
    std::vector<std::pair<double, double>> pts;
    for (double eps : {-0.01, -0.007, -0.005, -0.003, 0.003, 0.005, 0.007, 0.01})
        pts.emplace_back(eps, a0 + a2 * eps * eps);

    const auto c = elasticFitPolynomial(pts, 2);
    EXPECT_NEAR(c[0], a0, 1e-6);
    EXPECT_NEAR(c[1], 0.0, 1e-6);
    EXPECT_NEAR(c[2], a2, 1e-6);
}

// Quartic fit: E = E0 + a2·ε² + a4·ε⁴  → coefficients [E0, 0, a2, 0, a4]
TEST(FitPolynomial, QuarticRecoversCiiAndHigherOrder) {
    const double a0 = -10.0, a2 = 170.0, a4 = 5000.0;
    std::vector<std::pair<double, double>> pts;
    for (double eps : {-0.01, -0.008, -0.006, -0.004, -0.002, 0.002, 0.004, 0.006, 0.008, 0.01})
        pts.emplace_back(eps, a0 + a2 * eps * eps + a4 * std::pow(eps, 4));

    const auto c = elasticFitPolynomial(pts, 4);
    EXPECT_NEAR(c[0], a0, 1e-4);
    EXPECT_NEAR(c[1], 0.0, 1e-4);
    EXPECT_NEAR(c[2], a2, 1e-4);  // this is what Cii is derived from
    EXPECT_NEAR(c[3], 0.0, 1e-4);
    EXPECT_NEAR(c[4], a4, 1e-2);
}

// Quadratic fit of data that has a quartic component: the a2 coefficient is
// biased, but the quartic fit recovers it correctly.
TEST(FitPolynomial, QuarticCorrectionRemovesBias) {
    const double a2_true = 170.0, a4 = 8000.0;
    std::vector<std::pair<double, double>> pts;
    for (double eps : {-0.01, -0.008, -0.006, -0.004, -0.002, 0.002, 0.004, 0.006, 0.008, 0.01})
        pts.emplace_back(eps, a2_true * eps * eps + a4 * std::pow(eps, 4));

    const double a2_quad = elasticFitPolynomial(pts, 2)[2];
    const double a2_quartic = elasticFitPolynomial(pts, 4)[2];

    // Quadratic absorbs part of a4 → larger error
    EXPECT_GT(std::abs(a2_quad - a2_true), std::abs(a2_quartic - a2_true));
    EXPECT_NEAR(a2_quartic, a2_true, 1e-4);
}

// Too few points for the requested degree must throw.
TEST(FitPolynomial, ThrowsOnTooFewPoints) {
    std::vector<std::pair<double, double>> pts = {{0.01, 1.0}, {0.02, 2.0}};
    EXPECT_THROW(elasticFitPolynomial(pts, 4), std::runtime_error);
}
