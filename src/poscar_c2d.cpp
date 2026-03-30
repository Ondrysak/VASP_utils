#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "poscar_file.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Convert POSCAR coordinates from Cartesian to Direct"};

    std::string inputFile{"POSCAR"};
    std::string outputFile{"POSCAR_direct"};

    app.add_option("--input,-i", inputFile, "Input POSCAR file")->capture_default_str();
    app.add_option("--output,-o", outputFile, "Output POSCAR file")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    POSCAR poscar;
    if (!poscar.readPOSCAR(inputFile)) {
        std::cerr << "Error reading POSCAR file: " << inputFile << "\n";
        return 1;
    }

    if (!poscar.is_direct) {
        poscar.toDirect();
        std::cout << "Converted Cartesian -> Direct\n";
    } else {
        std::cout << "POSCAR is already in Direct coordinates\n";
    }

    if (!poscar.writePOSCAR(outputFile)) {
        std::cerr << "Error writing output POSCAR file: " << outputFile << "\n";
        return 1;
    }

    std::cout << "Output written to: " << outputFile << "\n";
    return 0;
}
