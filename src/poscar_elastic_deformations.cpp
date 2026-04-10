#include <array>
#include <CLI/CLI.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "elastic_utils.h"
#include "poscar_file.h"

namespace {

struct DeformationSpec {
    int i;
    int j;
    const char* label;
    bool shear;
};

void applyDeformationToLattice(POSCAR& poscar, const std::array<double, 9>& deformation) {
    double old_lattice[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            old_lattice[i][j] = poscar.lattice[i][j];

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k)
                value += deformation[i * 3 + k] * old_lattice[j][k];
            poscar.lattice[j][i] = value;
        }
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Generate POSCAR deformations for elastic constant calculations"};

    std::string input_file{"POSCAR"};
    std::string output_prefix{"POSCAR_def"};
    std::string manifest_file{"elastic_deformations.csv"};
    // Default amplitudes: ±0.5% and ±1.0% (4 points — sufficient for the stress
    // method).  For the energy method (parabolic fit) at least 7 points per mode
    // are recommended; use --norm-strains / --shear-strains to supply more.
    std::vector<double> norm_strains{-0.01, -0.005, 0.005, 0.01};
    std::vector<double> shear_strains{-0.01, -0.005, 0.005, 0.01};
    bool symmetric_deform{false};

    app.add_option("--input,-i", input_file, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output-prefix,-o", output_prefix, "Prefix for generated POSCAR files")->capture_default_str();
    app.add_option("--manifest,-m", manifest_file, "CSV output with strain/deformation metadata")
        ->capture_default_str();
    app.add_option("--norm-strains", norm_strains,
                   "Normal strain amplitudes (e11, e22, e33). Default: -0.01,-0.005,0.005,0.01")
        ->delimiter(',');
    app.add_option("--shear-strains", shear_strains,
                   "Shear strain amplitudes (e12, e13, e23). Default: -0.01,-0.005,0.005,0.01")
        ->delimiter(',');
    app.add_flag("--symmetric", symmetric_deform,
                 "Use symmetric deformation gradient F = sqrt(I+2E) instead of the default "
                 "upper-triangular (Cholesky) form. Recommended when lattice orientation "
                 "relative to spin or other external axes must be preserved.");

    CLI11_PARSE(app, argc, argv);

    if (!elasticValidateAmplitudes(norm_strains, shear_strains))
        return 1;

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
                deformation =
                    symmetric_deform ? elasticSymmetricDeformation(strain) : elasticCholeskyDeformation(strain);
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

            const auto voigt = elasticStrainVoigt(strain);
            manifest << output_file << "," << mode.label << "," << std::setprecision(12) << amount;
            for (double v : voigt)
                manifest << "," << v;
            for (double d : deformation)
                manifest << "," << d;
            manifest << "\n";

            ++index;
        }
    }

    const char* style = symmetric_deform ? "symmetric" : "upper-triangular (Cholesky)";
    std::cout << "Generated " << index << " deformed structures (" << style << " deformation gradient).\n";
    std::cout << "Manifest written to " << manifest_file << "\n";

    return 0;
}
