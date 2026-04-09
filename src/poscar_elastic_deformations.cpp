#include <lapacke.h>

#include <array>
#include <CLI/CLI.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "poscar_file.h"

namespace {

struct DeformationSpec {
    int i;
    int j;
    const char* label;
    bool shear;
};

std::array<double, 9> deformationFromStrain(const std::array<double, 9>& strain) {
    // Solve F^T F = I + 2E using an upper-triangular Cholesky factor (pymatgen style).
    std::array<double, 9> ftf{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            ftf[r * 3 + c] = (r == c ? 1.0 : 0.0) + 2.0 * strain[r * 3 + c];
        }
    }

    const lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 3, ftf.data(), 3);
    if (info != 0) {
        throw std::runtime_error("Cholesky decomposition failed for strain state (I + 2E not SPD)");
    }

    // Zero lower-triangular part to keep a clean upper-triangular deformation matrix.
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < r; ++c) {
            ftf[r * 3 + c] = 0.0;
        }
    }

    return ftf;
}

void applyDeformationToLattice(POSCAR& poscar, const std::array<double, 9>& deformation) {
    // POSCAR lattice is row-vector form. Apply: new = (F * old^T)^T.
    double old_lattice[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            old_lattice[i][j] = poscar.lattice[i][j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) {
                value += deformation[i * 3 + k] * old_lattice[j][k];
            }
            poscar.lattice[j][i] = value;
        }
    }
}

std::array<double, 6> strainVoigt(const std::array<double, 9>& strain) {
    return {strain[0], strain[4], strain[8], strain[5], strain[2], strain[1]};
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Generate POSCAR deformations for elastic constant calculations (pymatgen-like stencil)"};

    std::string input_file{"POSCAR"};
    std::string output_prefix{"POSCAR_def"};
    std::string manifest_file{"elastic_deformations.csv"};
    std::vector<double> norm_strains{-0.01, -0.005, 0.005, 0.01};
    std::vector<double> shear_strains{-0.06, -0.03, 0.03, 0.06};

    app.add_option("--input,-i", input_file, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output-prefix,-o", output_prefix, "Prefix for generated POSCAR files")->capture_default_str();
    app.add_option("--manifest,-m", manifest_file, "CSV output with strain/deformation metadata")
        ->capture_default_str();
    app.add_option("--norm-strains", norm_strains,
                   "Normal strain amplitudes (e11, e22, e33). Example: --norm-strains -0.01,-0.005,0.005,0.01")
        ->delimiter(',');
    app.add_option("--shear-strains", shear_strains,
                   "Shear strain amplitudes (e12, e13, e23). Example: --shear-strains -0.06,-0.03,0.03,0.06")
        ->delimiter(',');

    CLI11_PARSE(app, argc, argv);

    POSCAR reference;
    if (!reference.readPOSCAR(input_file)) {
        std::cerr << "Error: failed to parse input POSCAR " << input_file << "\n";
        return 1;
    }
    reference.toDirect();

    std::ofstream manifest(manifest_file);
    if (!manifest) {
        std::cerr << "Error: failed to open manifest file " << manifest_file << " for writing\n";
        return 1;
    }

    manifest << "file,mode,amount,e11,e22,e33,e23,e13,e12,d11,d12,d13,d21,d22,d23,d31,d32,d33\n";

    const std::array<DeformationSpec, 6> modes = {{{0, 0, "e11", false},
                                                   {1, 1, "e22", false},
                                                   {2, 2, "e33", false},
                                                   {0, 1, "e12", true},
                                                   {0, 2, "e13", true},
                                                   {1, 2, "e23", true}}};

    int index = 0;
    for (const auto& mode : modes) {
        const auto& amplitudes = mode.shear ? shear_strains : norm_strains;
        for (double amount : amplitudes) {
            std::array<double, 9> strain{};
            strain[mode.i * 3 + mode.j] = amount;
            strain[mode.j * 3 + mode.i] = amount;

            std::array<double, 9> deformation;
            try {
                deformation = deformationFromStrain(strain);
            } catch (const std::exception& ex) {
                std::cerr << "Error: " << ex.what() << " for mode " << mode.label << " amount=" << amount << "\n";
                return 1;
            }

            POSCAR deformed = reference;
            applyDeformationToLattice(deformed, deformation);

            const std::string output_file =
                output_prefix + "_" + std::to_string(index) + "_" + mode.label + "_" + std::to_string(amount);

            if (!deformed.writePOSCAR(output_file)) {
                std::cerr << "Error: failed to write deformed POSCAR " << output_file << "\n";
                return 1;
            }

            const auto voigt = strainVoigt(strain);
            manifest << output_file << "," << mode.label << "," << std::setprecision(12) << amount;
            for (double v : voigt) {
                manifest << "," << v;
            }
            for (double d : deformation) {
                manifest << "," << d;
            }
            manifest << "\n";

            ++index;
        }
    }

    std::cout << "Generated " << index << " deformed structures using pymatgen-style independent strain modes.\n";
    std::cout << "Manifest written to " << manifest_file << "\n";

    return 0;
}
