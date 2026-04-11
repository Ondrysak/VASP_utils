#include "outcar_file.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Conversion factor: VASP OUTCAR reports elastic moduli in kBar.
// 1 kBar = 0.1 GPa.
static constexpr double kBar_to_GPa = 0.1;

// Label used in VASP OUTCAR to mark the symmetrized elastic tensor block.
static constexpr const char* kElasticHeader = "SYMMETRIZED ELASTIC MODULI (kBar)";

bool OUTCAR::readOUTCAR(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open OUTCAR file " << filename << "\n";
        return false;
    }

    // Scan the entire file, keeping the last occurrence of the elastic tensor
    // block (restarts produce duplicate blocks; we want the final converged one).

    bool found_block = false;
    ElasticTensor candidate{};

    std::string line;
    while (std::getline(file, line)) {
        // Detect the header line.
        if (line.find(kElasticHeader) == std::string::npos)
            continue;

        // Next line: column labels "Direction   XX   YY ..." — skip.
        if (!std::getline(file, line))
            break;

        // Next line: separator "------..." — skip.
        if (!std::getline(file, line))
            break;

        // Read 6 data rows (XX, YY, ZZ, XY, YZ, ZX).
        bool row_ok = true;
        for (int row = 0; row < 6 && row_ok; ++row) {
            if (!std::getline(file, line)) {
                row_ok = false;
                break;
            }
            std::istringstream iss(line);

            // First token is the row label ("XX", "YY", ...); discard it.
            std::string label;
            iss >> label;

            for (int col = 0; col < 6 && row_ok; ++col) {
                double val;
                if (!(iss >> val)) {
                    std::cerr << "Warning: malformed elastic tensor row in " << filename << " (row " << row << ")\n";
                    row_ok = false;
                    break;
                }
                candidate.C[row][col] = val * kBar_to_GPa;
            }
        }

        if (row_ok) {
            found_block = true;
            elastic_tensor = candidate;
        }
    }

    has_elastic_tensor = found_block;

    if (!found_block) {
        std::cerr << "Error: no elastic tensor found in " << filename << "\n"
                  << "       Make sure the VASP run used IBRION=6 and completed successfully.\n";
        return false;
    }

    return true;
}
