#include <spglib.h>

#include <CLI/CLI.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "poscar_file.h"
#include "symmetry.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Transform POSCAR structure to primitive cell using spglib"};

    std::string inputFile{"POSCAR"};
    std::string outputFile{"POSCAR_primitive"};
    double symprec{1e-5};
    const int primitive{1};
    int idealize{0};

    app.add_option("--input,-i", inputFile, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output,-o", outputFile, "Output POSCAR file")->capture_default_str();
    app.add_option("--symprec", symprec, "Symmetry tolerance (spglib symprec)")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_flag("--idealize", idealize, "Idealize the primitive cell");

    CLI11_PARSE(app, argc, argv);

    if (symprec > 1e-3) {
        std::cerr << "Warning: symprec is too high! Consider using default value 1e-5.\n";
    }

    POSCAR poscar;

    if (!poscar.readPOSCAR(inputFile)) {
        std::cerr << "Error: failed to parse input POSCAR " << inputFile << "\n";
        return 1;
    }

    auto poscar_stand = standardizeCell(poscar, symprec, primitive, idealize);
    if (!poscar_stand) {
        std::cerr << "Error: failed to create primitive cell POSCAR.\n";
        return 1;
    }

    if (!poscar_stand->writePOSCAR(outputFile)) {
        std::cerr << "Error: failed to write primitie cell POSCAR file to " << outputFile << "\n";
        return 1;
    }

    return 0;
}
