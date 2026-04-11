#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "elastic_properties.h"
#include "outcar_file.h"
#include "poscar_file.h"
#include "symmetry.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Compute elastic moduli and mechanical stability from a VASP OUTCAR.\n"
                 "Crystal system is detected automatically from the POSCAR symmetry."};

    std::string poscar_file{"POSCAR"};
    std::string outcar_file{"OUTCAR"};
    double symprec{1e-5};

    app.add_option("--poscar,-p", poscar_file, "Input POSCAR file (used for symmetry detection)")
        ->capture_default_str()
        ->check(CLI::ExistingFile);
    app.add_option("--outcar,-o", outcar_file, "Input OUTCAR file (must contain IBRION=6 elastic tensor)")
        ->capture_default_str()
        ->check(CLI::ExistingFile);
    app.add_option("--symprec", symprec, "Symmetry tolerance (spglib symprec)")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);

    // ── Parse POSCAR ─────────────────────────────────────────────────────────
    POSCAR poscar;
    if (!poscar.readPOSCAR(poscar_file)) {
        std::cerr << "Error: failed to parse POSCAR file " << poscar_file << "\n";
        return 1;
    }

    // ── Parse OUTCAR ─────────────────────────────────────────────────────────
    OUTCAR outcar;
    if (!outcar.readOUTCAR(outcar_file)) {
        return 1;
    }

    // ── Symmetry analysis (space group → crystal system) ─────────────────────
    auto dataset = analyzeSymmetry(poscar, symprec);
    CrystalSystem cs = CrystalSystem::Unknown;

    if (dataset) {
        cs = crystalSystemFromSpaceGroup(dataset->spacegroup_number);
        std::cout << "Space group: " << dataset->spacegroup_number << "  (" << dataset->international_symbol << ")\n";
    } else {
        std::cerr << "Warning: symmetry analysis failed; stability criteria may be overly general.\n";
    }

    // ── Compute moduli and stability ─────────────────────────────────────────
    ElasticModuli moduli = computeElasticModuli(outcar.elastic_tensor);
    StabilityResult stability = checkMechanicalStability(outcar.elastic_tensor, cs);

    // ── Print report ─────────────────────────────────────────────────────────
    std::cout << "\n";
    printElasticReport(outcar.elastic_tensor, moduli, stability, cs);

    return 0;
}
