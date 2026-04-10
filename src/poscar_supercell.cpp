#include <CLI/CLI.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "poscar_file.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Create supercell"};

    std::string inputFile{"POSCAR"};
    std::string outputFile{"POSCAR_"};
    bool diagonal{true};
    double S[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    std::vector<int> dims = {1, 1, 1};

    app.add_option("--input,-i", inputFile, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output,-o", outputFile, "Output POSCAR file")->capture_default_str();
    app.add_option("--dim", dims, "Dimensions of supercell (nx ny nz)")
        ->expected(3)
        ->check(CLI::PositiveNumber)
        ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (std::any_of(dims.begin(), dims.end(), [](int d) { return d > 100; })) {
        std::cerr << "Error: one of the dimensions exceeds 100! That's too much!\n";
        return 1;
    }

    if (std::any_of(dims.begin(), dims.end(), [](int d) { return d > 20; })) {
        std::cerr << "Warning: one of the dimensions exeeds 20. Consider if you need that!\n";
    }

    S[0] = dims[0];
    S[4] = dims[1];
    S[8] = dims[2];

    if (outputFile == "POSCAR_") {
        outputFile += std::to_string((int)S[0]) + "_" + std::to_string((int)S[4]) + "_" + std::to_string((int)S[8]);
    }

    POSCAR poscar;

    if (!poscar.readPOSCAR(inputFile)) {
        std::cerr << "Error: failed to parse input POSCAR " << inputFile << "\n";
        return 1;
    }
    poscar.toDirect();

    POSCAR poscar_supercell;
    poscar_supercell = poscar.makeSupercell(S);
    if (poscar_supercell.total_atoms == 0) {
        std::cerr << "Error: failed to make supercell!!!\n";
        return 1;
    }

    if (!poscar_supercell.writePOSCAR(outputFile)) {
        std::cerr << "Error: failed to write primitie cell POSCAR file to " << outputFile << "\n";
        return 1;
    }

    return 0;
}
