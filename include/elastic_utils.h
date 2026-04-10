#ifndef ELASTIC_UTILS_H_INCLUDED
#define ELASTIC_UTILS_H_INCLUDED

#include <lapacke.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief Warn if any amplitude exceeds 1 %, error if any exceeds 5 %.
 *
 * @param normal   Normal-strain amplitudes (e11, e22, e33).
 * @param shear    Shear-strain amplitudes  (e12, e13, e23).
 * @param warn_os  Stream for warning messages (default: std::cerr).
 * @param err_os   Stream for error  messages (default: std::cerr).
 * @return true if all amplitudes are within the 5 % hard limit.
 */
inline bool elasticValidateAmplitudes(const std::vector<double>& normal, const std::vector<double>& shear,
                                      std::ostream& warn_os = std::cerr, std::ostream& err_os = std::cerr) {
    static constexpr double kWarn = 0.01;
    static constexpr double kMax = 0.05;
    bool ok = true;
    auto check = [&](double v, const char* tag) {
        if (std::abs(v) > kMax) {
            err_os << "Error: " << tag << " strain amplitude " << v << " exceeds maximum allowed " << kMax << "\n";
            ok = false;
        } else if (std::abs(v) > kWarn) {
            warn_os << "Warning: " << tag << " strain amplitude " << v << " > " << kWarn
                    << " — consider staying at or below 1% for accuracy.\n";
        }
    };
    for (double v : normal)
        check(v, "normal");
    for (double v : shear)
        check(v, "shear");
    return ok;
}

/**
 * @brief Upper-triangular (Cholesky) deformation gradient satisfying F^T F = I + 2E.
 *
 * The first lattice-vector direction is left unchanged. This is the pymatgen-style
 * deformation used by default.
 */
inline std::array<double, 9> elasticCholeskyDeformation(const std::array<double, 9>& strain) {
    std::array<double, 9> ftf{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            ftf[r * 3 + c] = (r == c ? 1.0 : 0.0) + 2.0 * strain[r * 3 + c];

    if (LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 3, ftf.data(), 3) != 0)
        throw std::runtime_error("Cholesky decomposition failed (I + 2E not SPD)");

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < r; ++c)
            ftf[r * 3 + c] = 0.0;
    return ftf;
}

/**
 * @brief Symmetric deformation gradient F = sqrt(I + 2E) via eigendecomposition.
 *
 * Applies the deformation isotropically — important when the lattice orientation
 * relative to an external quantisation axis (e.g. spin) must be preserved.
 */
inline std::array<double, 9> elasticSymmetricDeformation(const std::array<double, 9>& strain) {
    std::array<double, 9> A{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[r * 3 + c] = (r == c ? 1.0 : 0.0) + 2.0 * strain[r * 3 + c];

    std::array<double, 3> w{};
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', 3, A.data(), 3, w.data()) != 0)
        throw std::runtime_error("Eigendecomposition failed for symmetric deformation gradient");

    // After DSYEV (ROW_MAJOR): A[i*3+k] = i-th component of k-th eigenvector.
    for (int k = 0; k < 3; ++k) {
        if (w[k] < 0.0)
            throw std::runtime_error("I + 2E is not positive definite (strain too large)");
        w[k] = std::sqrt(w[k]);
    }

    std::array<double, 9> F{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                F[i * 3 + j] += A[i * 3 + k] * w[k] * A[j * 3 + k];
    return F;
}

/**
 * @brief Convert 3×3 strain tensor (row-major) to Voigt notation [e11,e22,e33,e23,e13,e12].
 */
inline std::array<double, 6> elasticStrainVoigt(const std::array<double, 9>& strain) {
    return {strain[0], strain[4], strain[8], strain[5], strain[2], strain[1]};
}

/**
 * @brief Convert VASP "in kB" stress [XX,YY,ZZ,XY,YZ,ZX] to Voigt [11,22,33,23,13,12] in GPa.
 */
inline std::array<double, 6> elasticVaspStressToVoigt(const std::array<double, 6>& kb, double sign) {
    static constexpr double kBar_to_GPa = 0.1;
    return {sign * kb[0] * kBar_to_GPa, sign * kb[1] * kBar_to_GPa, sign * kb[2] * kBar_to_GPa,
            sign * kb[4] * kBar_to_GPa, sign * kb[5] * kBar_to_GPa, sign * kb[3] * kBar_to_GPa};
}

/**
 * @brief Least-squares polynomial fit of degree @p degree to (x, y) data.
 *
 * @return Coefficients [a0, a1, ..., a_degree] of the fitted polynomial.
 * @throws std::runtime_error on degenerate input or LAPACK failure.
 */
inline std::vector<double> elasticFitPolynomial(const std::vector<std::pair<double, double>>& pts, int degree) {
    const int N = static_cast<int>(pts.size());
    const int nc = degree + 1;
    if (N < nc)
        throw std::runtime_error("Too few data points for polynomial degree");

    std::vector<double> A(N * nc), b(N);
    for (int r = 0; r < N; ++r) {
        double pw = 1.0;
        for (int c = 0; c < nc; ++c) {
            A[r * nc + c] = pw;
            pw *= pts[r].first;
        }
        b[r] = pts[r].second;
    }
    if (LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', N, nc, 1, A.data(), nc, b.data(), 1) != 0)
        throw std::runtime_error("LAPACKE_dgels failed in elasticFitPolynomial");

    b.resize(nc);
    return b;
}

#endif  // ELASTIC_UTILS_H_INCLUDED
