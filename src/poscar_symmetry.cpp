#include <spglib.h>

#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "poscar_file.h"
#include "symmetry.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Analyze symmetry of a POSCAR structure using spglib"};

    std::string inputFile{"POSCAR"};
    std::string outputFile{"POSCAR_primitive"};
    double symprec{1e-5};
    bool primitive{false};
    bool wyckoff{false};
    bool symoperation{false};

    app.add_option("--input,-i", inputFile, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output,-o", outputFile, "Output POSCAR file (used with --primitive)")->capture_default_str();
    app.add_option("--symprec", symprec, "Symmetry tolerance (spglib symprec)")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_flag("--primitive,-p", primitive, "Generate primitive cell POSCAR");
    app.add_flag("--wyckoff,-w", wyckoff, "Print Wyckoff positions");
    app.add_flag("--symoper,-s", symoperation, "Print symmetry operations");

    CLI11_PARSE(app, argc, argv);

    if (symprec > 1e-3) {
        std::cerr << "Warning: symprec is too high! Consider using default value 1e-5.\n";
    }

    POSCAR poscar;
    poscar.readPOSCAR(inputFile);

    auto dataset = analyzeSymmetry(poscar, symprec);

    if (dataset) {
        printSymmetryInfo(*dataset, wyckoff, symoperation);
    } else {
        std::cerr << "Error: failed to analyze symmetry.\n";
        return 1;
    }

    if (primitive) {
        std::cout << "Creating the primitive cell file.\n";
        auto maybePrimitive = standardizeCell(poscar, symprec, 1, 0);

        if (!maybePrimitive) {
            std::cerr << "Error: failed to create primitive POSCAR.\n";
            return 1;
        }

        if (!maybePrimitive->writePOSCAR(outputFile)) {
            std::cerr << "Error: failed to write primitive POSCAR to " << outputFile << "\n";
            return 1;
        }
    }

    return 0;
}
