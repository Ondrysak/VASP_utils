#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "poscar_file.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Transform POSCAR structure format to ctrls format used by ecalj/Qestaal"};

    std::string inputFile{"POSCAR"};
    std::string outputFile{"ctrls.system"};

    app.add_option("--input,-i", inputFile, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--output,-o", outputFile, "Output ctrls file")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    POSCAR poscar;
    if (!poscar.readPOSCAR(inputFile)) {
        std::cerr << "Error: failed to parse input POSCAR " << inputFile << "\n";
        return 1;
    }

    if (poscar.is_direct) {
        poscar.toCartesian();
        std::cout << "Converted Direct -> Cartesian\n";
    }

    if (!poscar.writeCtrlsFile(outputFile)) {
        std::cerr << "Error: writing output ctrls file " << outputFile << "\n";
        return 1;
    }

    std::cout << "Output written to: " << outputFile << "\n";

    return 0;
}
