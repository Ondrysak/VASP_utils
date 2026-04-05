#include <algorithm>
#include <cctype>
#include <CLI/CLI.hpp>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "poscar_file.h"
#include "random_utility.h"

namespace {

std::vector<std::string> splitCsv(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }),
                   item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

bool parseIndicesSpec(const std::string& spec, int total_atoms, std::set<size_t>& indices_out) {
    if (spec.empty()) {
        return true;
    }

    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c); }),
                    token.end());
        if (token.empty()) {
            continue;
        }

        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            int idx = 0;
            try {
                idx = std::stoi(token);
            } catch (...) {
                return false;
            }
            if (idx < 1 || idx > total_atoms) {
                return false;
            }
            indices_out.insert(static_cast<size_t>(idx - 1));
            continue;
        }

        int lo = 0;
        int hi = 0;
        try {
            lo = std::stoi(token.substr(0, dash));
            hi = std::stoi(token.substr(dash + 1));
        } catch (...) {
            return false;
        }

        if (lo > hi || lo < 1 || hi > total_atoms) {
            return false;
        }
        for (int i = lo; i <= hi; ++i) {
            indices_out.insert(static_cast<size_t>(i - 1));
        }
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Randomly displace atoms in a POSCAR file to generate perturbed structures"};

    std::string filename{"POSCAR"};
    int n_atoms{1};
    int n_files{1};
    double amplitude{0.01};
    double ampx{0.01};
    double ampy{0.01};
    double ampz{0.01};
    bool allAtoms{false};
    bool wrap{false};
    bool zero_net{false};
    unsigned int seed{0};
    bool seed_provided{false};
    std::string species_filter;
    std::string indices_spec;

    app.add_option("--input,-i", filename, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--nfiles,-n", n_files, "Number of displaced structure files to create")
        ->capture_default_str()
        ->check(CLI::Range(1, 1000));
    app.add_option("--natoms", n_atoms, "Number of atoms to displace")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_flag("--allatoms", allAtoms, "Displace all eligible atoms (overrides --natoms)");
    app.add_option("--amp,-a", amplitude, "Maximum isotropic displacement amplitude in Angstroms")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    app.add_option("--ampx", ampx, "Maximum displacement along x in Angstroms")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    app.add_option("--ampy", ampy, "Maximum displacement along y in Angstroms")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    app.add_option("--ampz", ampz, "Maximum displacement along z in Angstroms")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
    app.add_option("--seed", seed, "Random seed for deterministic displacements")
        ->capture_default_str()
        ->each([&seed_provided](const std::string&) { seed_provided = true; });
    app.add_option("--species", species_filter, "Comma-separated list of element symbols to displace");
    app.add_option("--indices", indices_spec, "1-based atom indices and ranges (e.g. 1,3-5,8)");
    app.add_flag("--wrap", wrap, "Wrap resulting fractional coordinates into [0,1)");
    app.add_flag("--zero-net", zero_net, "Remove net Cartesian translation over displaced atoms");

    CLI11_PARSE(app, argc, argv);

    if (ampx == 0.01 && ampy == 0.01 && ampz == 0.01) {
        ampx = amplitude;
        ampy = amplitude;
        ampz = amplitude;
    }

    if (std::max({ampx, ampy, ampz}) > 0.75) {
        std::cerr << "Warning: amplitude is above 0.75 Angstroms! That can cause errors running VASP!\n";
    }

    if (seed_provided) {
        seedRandom(seed);
    }

    POSCAR original;
    if (!original.readPOSCAR(filename)) {
        std::cerr << "Error: reading POSCAR file " << filename << "\n";
        return 1;
    }

    const auto atom_species = original.atomSpecies();
    std::set<size_t> selected;
    for (size_t i = 0; i < atom_species.size(); ++i) {
        selected.insert(i);
    }

    if (!species_filter.empty()) {
        const auto wanted_species = splitCsv(species_filter);
        std::set<size_t> by_species;
        for (size_t i = 0; i < atom_species.size(); ++i) {
            if (std::find(wanted_species.begin(), wanted_species.end(), atom_species[i]) != wanted_species.end()) {
                by_species.insert(i);
            }
        }

        std::set<size_t> filtered;
        std::set_intersection(selected.begin(), selected.end(), by_species.begin(), by_species.end(),
                              std::inserter(filtered, filtered.begin()));
        selected = std::move(filtered);
    }

    if (!indices_spec.empty()) {
        std::set<size_t> by_indices;
        if (!parseIndicesSpec(indices_spec, original.total_atoms, by_indices)) {
            std::cerr << "Error: invalid --indices specification. Use e.g. 1,3-5,8\n";
            return 1;
        }

        std::set<size_t> filtered;
        std::set_intersection(selected.begin(), selected.end(), by_indices.begin(), by_indices.end(),
                              std::inserter(filtered, filtered.begin()));
        selected = std::move(filtered);
    }

    std::vector<size_t> candidate_indices(selected.begin(), selected.end());
    if (candidate_indices.empty()) {
        std::cerr << "Warning: no atoms matched filters; no files generated.\n";
        return 0;
    }

    if (allAtoms) {
        if (n_atoms != 1) {
            std::cerr << "Note: --allatoms overrides --natoms\n";
        }
        n_atoms = static_cast<int>(candidate_indices.size());
        std::cout << "Displacing all eligible atoms in input file.\n";
    } else if (n_atoms > static_cast<int>(candidate_indices.size())) {
        std::cerr << "Warning: requested more atoms than eligible. Displacing all " << candidate_indices.size()
                  << " eligible atoms.\n";
        n_atoms = static_cast<int>(candidate_indices.size());
    }

    std::cout << "Number of atoms to displace: " << n_atoms << "\n";

    DisplacementOptions options;
    options.candidate_indices = std::move(candidate_indices);
    options.wrap_direct = wrap;
    options.zero_net = zero_net;
    options.ampx = ampx;
    options.ampy = ampy;
    options.ampz = ampz;

    for (int j = 0; j < n_files; j++) {
        POSCAR output(original);
        std::string filenameOut = "POSCAR_modified" + std::to_string(j + 1);
        output.displaceAtoms(n_atoms, options);
        if (!output.writePOSCAR(filenameOut)) {
            std::cerr << "Error: failed to write displaced POSCAR to " << filenameOut << "\n";
            return 1;
        }
    }

    return 0;
}
