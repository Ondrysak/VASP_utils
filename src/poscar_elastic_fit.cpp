#include <lapacke.h>

#include <array>
#include <CLI/CLI.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

static constexpr double kBar_to_GPa = 0.1;
static constexpr double eV_per_A3_to_GPa = 160.21766208;

struct DeformRecord {
    std::string file;
    std::string mode;
    double amount;
    std::array<double, 6> strain_voigt;  // [ε₁₁, ε₂₂, ε₃₃, 2ε₂₃, 2ε₁₃, 2ε₁₂]
};

std::vector<DeformRecord> parseManifest(const std::string& path) {
    std::ifstream f(path);
    std::vector<DeformRecord> records;
    if (!f)
        return records;
    std::string line;
    std::getline(f, line);  // header: file,mode,amount,e11,e22,e33,e23,e13,e12,...
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string tok;
        DeformRecord rec;
        std::getline(ss, rec.file, ',');
        std::getline(ss, rec.mode, ',');
        std::getline(ss, tok, ',');
        rec.amount = std::stod(tok);
        double e[6]{};
        for (int i = 0; i < 6; ++i) {
            std::getline(ss, tok, ',');
            e[i] = std::stod(tok);
        }
        // manifest order: e11, e22, e33, e23, e13, e12
        // Voigt shear strain = 2 × tensor shear strain
        rec.strain_voigt = {e[0], e[1], e[2], 2.0 * e[3], 2.0 * e[4], 2.0 * e[5]};
        records.push_back(std::move(rec));
    }
    return records;
}

// Look for OUTCAR at <dir>/<basename>/OUTCAR or <dir>/<basename>.OUTCAR
std::string findOutcar(const std::string& outcar_dir, const std::string& file_col) {
    namespace fs = std::filesystem;
    const fs::path base = fs::path(file_col).filename();
    const fs::path dir(outcar_dir);
    if (const auto p = dir / base / "OUTCAR"; fs::exists(p))
        return p.string();
    if (const auto p = dir / (base.string() + ".OUTCAR"); fs::exists(p))
        return p.string();
    return {};
}

// Parse LAST "in kB" line → [XX, YY, ZZ, XY, YZ, ZX] in kBar
bool parseOutcarStress(const std::string& path, std::array<double, 6>& out) {
    std::ifstream f(path);
    if (!f)
        return false;
    std::string line;
    bool found = false;
    while (std::getline(f, line)) {
        const auto pos = line.find("in kB");
        if (pos == std::string::npos)
            continue;
        std::istringstream ss(line.substr(pos + 5));
        double v[6];
        bool ok = true;
        for (int i = 0; i < 6; ++i)
            if (!(ss >> v[i])) {
                ok = false;
                break;
            }
        if (ok) {
            for (int i = 0; i < 6; ++i)
                out[i] = v[i];
            found = true;
        }
    }
    return found;
}

// Parse LAST "free  energy   TOTEN" line → energy in eV
bool parseOutcarEnergy(const std::string& path, double& energy_ev) {
    std::ifstream f(path);
    if (!f)
        return false;
    std::string line;
    bool found = false;
    while (std::getline(f, line)) {
        if (line.find("free  energy   TOTEN") == std::string::npos)
            continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::istringstream ss(line.substr(eq + 1));
        double e;
        if (ss >> e) {
            energy_ev = e;
            found = true;
        }
    }
    return found;
}

// VASP [XX, YY, ZZ, XY, YZ, ZX] kBar → Voigt [11, 22, 33, 23, 13, 12] GPa
// VASP index:  0    1    2    3    4    5
// Voigt index: 0    1    2    5    3    4   (reorder 23←4, 13←5, 12←3)
std::array<double, 6> vaspStressToVoigt(const std::array<double, 6>& kb, double sign) {
    return {sign * kb[0] * kBar_to_GPa,   // 11 = XX
            sign * kb[1] * kBar_to_GPa,   // 22 = YY
            sign * kb[2] * kBar_to_GPa,   // 33 = ZZ
            sign * kb[4] * kBar_to_GPa,   // 23 = YZ
            sign * kb[5] * kBar_to_GPa,   // 13 = ZX
            sign * kb[3] * kBar_to_GPa};  // 12 = XY
}

void printMatrix(const std::array<std::array<double, 6>, 6>& C, std::ostream& os) {
    static const char* lbl[] = {"11", "22", "33", "23", "13", "12"};
    os << std::fixed << std::setprecision(3);
    os << "\nElastic constants Cij (GPa):\n";
    os << "         ";
    for (const auto* s : lbl)
        os << std::setw(9) << s;
    os << "\n";
    for (int i = 0; i < 6; ++i) {
        os << "  C" << lbl[i] << "  ";
        for (int j = 0; j < 6; ++j)
            os << std::setw(9) << C[i][j];
        os << "\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Fit elastic constants Cij from VASP OUTCAR files"};

    std::string manifest_file{"elastic_deformations.csv"};
    std::string outcar_dir{"."};
    std::string method{"stress"};
    std::string output_file{"elastic_constants.txt"};
    double volume{0.0};
    bool negate_stress{false};

    app.add_option("--manifest,-m", manifest_file, "Manifest CSV from poscar_elastic_deformations")
        ->capture_default_str()
        ->check(CLI::ExistingFile);
    app.add_option("--outcar-dir,-d", outcar_dir, "Root dir: OUTCARs at <dir>/<name>/OUTCAR or <dir>/<name>.OUTCAR")
        ->capture_default_str();
    app.add_option("--method", method,
                   "stress: linear stress-strain fit (full Cij); "
                   "energy: parabolic energy-strain fit (diagonal Cii only)")
        ->capture_default_str()
        ->check(CLI::IsMember({"stress", "energy"}));
    app.add_option("--volume,-V", volume, "Reference cell volume in Å³ (required for energy method)");
    app.add_option("--output,-o", output_file, "Output text file")->capture_default_str();
    app.add_flag("--negate-stress", negate_stress, "Negate VASP stress values (use if fitted Cij have wrong sign)");

    CLI11_PARSE(app, argc, argv);

    const auto records = parseManifest(manifest_file);
    if (records.empty()) {
        std::cerr << "Error: empty or unreadable manifest " << manifest_file << "\n";
        return 1;
    }
    std::cout << "Loaded " << records.size() << " deformation records from " << manifest_file << ".\n";

    std::array<std::array<double, 6>, 6> C{};

    // ---- stress method: linear regression σ = C·ε ----
    if (method == "stress") {
        const int M = static_cast<int>(records.size());
        // A: M×6 strain matrix;  B: M×6 stress matrix (GPa)
        std::vector<double> A(M * 6, 0.0);
        std::vector<double> B(M * 6, 0.0);
        const double sign = negate_stress ? -1.0 : 1.0;

        for (int r = 0; r < M; ++r) {
            const std::string outcar = findOutcar(outcar_dir, records[r].file);
            if (outcar.empty()) {
                std::cerr << "Error: OUTCAR not found for '" << records[r].file << "'\n"
                          << "  Searched in: " << outcar_dir << "\n";
                return 1;
            }
            std::array<double, 6> stress_kb{};
            if (!parseOutcarStress(outcar, stress_kb)) {
                std::cerr << "Error: no 'in kB' stress line found in " << outcar << "\n"
                          << "  Make sure ISIF >= 2 is set in INCAR.\n";
                return 1;
            }
            const auto sv = vaspStressToVoigt(stress_kb, sign);
            for (int c = 0; c < 6; ++c) {
                A[r * 6 + c] = records[r].strain_voigt[c];
                B[r * 6 + c] = sv[c];
            }
        }

        // Solve min‖A·X − B‖² where X = Cᵀ (A: M×6, B: M×6)
        // After dgels, B[j*6+i] = Xᵀ[i][j]  →  C[i][j] = B[j*6+i]
        if (LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', M, 6, 6, A.data(), 6, B.data(), 6) != 0) {
            std::cerr << "Error: LAPACKE_dgels least-squares solve failed.\n";
            return 1;
        }
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                C[i][j] = B[j * 6 + i];
        // Symmetrize
        for (int i = 0; i < 6; ++i)
            for (int j = i + 1; j < 6; ++j) {
                const double s = 0.5 * (C[i][j] + C[j][i]);
                C[i][j] = C[j][i] = s;
            }

        // ---- energy method: parabolic fit E = a₀ + a₁ε + a₂ε² per mode ----
    } else {
        if (volume <= 0.0) {
            std::cerr << "Error: --volume <Å³> is required for the energy method.\n";
            return 1;
        }
        static const std::map<std::string, int> modeVi = {{"e11", 0}, {"e22", 1}, {"e33", 2},
                                                          {"e23", 3}, {"e13", 4}, {"e12", 5}};

        std::map<std::string, std::vector<std::pair<double, double>>> data;
        for (const auto& rec : records) {
            const auto it = modeVi.find(rec.mode);
            if (it == modeVi.end()) {
                std::cerr << "Warning: unknown mode '" << rec.mode << "'\n";
                continue;
            }
            const std::string outcar = findOutcar(outcar_dir, rec.file);
            if (outcar.empty()) {
                std::cerr << "Error: OUTCAR not found for '" << rec.file << "'\n";
                return 1;
            }
            double energy;
            if (!parseOutcarEnergy(outcar, energy)) {
                std::cerr << "Error: no TOTEN line in " << outcar << "\n";
                return 1;
            }
            data[rec.mode].emplace_back(rec.strain_voigt[it->second], energy);
        }

        for (const auto& [mode, pts] : data) {
            const int vi = modeVi.at(mode);
            const int N = static_cast<int>(pts.size());
            std::vector<double> Av(N * 3), bv(N);
            for (int r = 0; r < N; ++r) {
                const double eps = pts[r].first;
                Av[r * 3 + 0] = 1.0;
                Av[r * 3 + 1] = eps;
                Av[r * 3 + 2] = eps * eps;
                bv[r] = pts[r].second;
            }
            if (LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', N, 3, 1, Av.data(), 3, bv.data(), 1) != 0) {
                std::cerr << "Error: polynomial fit failed for mode " << mode << "\n";
                return 1;
            }
            // Cii = 2·a₂ / V₀  (eV/Å³ → GPa)
            C[vi][vi] = 2.0 * bv[2] / volume * eV_per_A3_to_GPa;
        }
        std::cout << "Note: energy method gives diagonal Cii only.\n"
                  << "      Off-diagonal Cij require mixed-strain deformations.\n";
    }

    printMatrix(C, std::cout);

    std::ofstream of(output_file);
    if (!of) {
        std::cerr << "Warning: cannot write " << output_file << "\n";
    } else {
        of << "# Elastic constants Cij in GPa  (Voigt rows/cols: 11 22 33 23 13 12)\n"
           << "# Method: " << method << "\n"
           << std::fixed << std::setprecision(6);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
                of << std::setw(14) << C[i][j];
            of << "\n";
        }
        std::cout << "Written: " << output_file << "\n";
    }

    return 0;
}
