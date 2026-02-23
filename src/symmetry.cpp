#include "symmetry.h"

#include <spglib.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "poscar_file.h"

void initializeSpglibInput(POSCAR& poscarDirect, double lattice[3][3], std::vector<std::array<double, 3>>& positions,
                           std::vector<int>& types, std::map<std::string, int>& element_map) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lattice[i][j] = poscarDirect.lattice[i][j];

    // Prepare atomic positions as flat vector (num_atoms*3)
    for (int i = 0; i < poscarDirect.total_atoms; ++i) {
        positions[i][0] = poscarDirect.coordinates[i].x;
        positions[i][1] = poscarDirect.coordinates[i].y;
        positions[i][2] = poscarDirect.coordinates[i].z;
    }

    // Prepare atomic types
    int type_counter{1};
    int idx{0};

    for (size_t i = 0; i < poscarDirect.elements.size(); ++i) {
        const std::string& el = poscarDirect.elements[i];
        int n = poscarDirect.num_atoms[i];
        if (element_map.find(el) == element_map.end())
            element_map[el] = type_counter++;
        for (int j = 0; j < n; ++j)
            types[idx++] = element_map[el];
    }
}

SpglibDatasetPtr analyzeSymmetry(const POSCAR& poscar, const double& symprec) {
    // Make a copy in fractional coordinates
    POSCAR poscarDirect = poscar;
    if (!poscarDirect.is_direct) {
        poscarDirect.toDirect();
    }

    int num_atoms = static_cast<int>(poscarDirect.total_atoms);
    double lattice[3][3];
    std::vector<std::array<double, 3>> positions(num_atoms);
    std::vector<int> types(num_atoms);
    std::map<std::string, int> element_map;

    initializeSpglibInput(poscarDirect, lattice, positions, types, element_map);

    SpglibDataset* dataset = spg_get_dataset(lattice, (double(*)[3])positions.data(), types.data(), num_atoms, symprec);

    return SpglibDatasetPtr(dataset, &spg_free_dataset);
}

std::optional<POSCAR> standardizeCell(const POSCAR& poscar, const double& symprec, const int& primitive,
                                      const int& idealize) {
    // 1) Copy and ensure fractional coordinates
    POSCAR poscarDirect = poscar;
    if (!poscarDirect.is_direct) {
        poscarDirect.toDirect();
    }

    int num_atoms_in = static_cast<int>(poscarDirect.total_atoms);
    int num_atoms{num_atoms_in * 64};

    double lattice[3][3];
    std::vector<std::array<double, 3>> positions(num_atoms);
    std::vector<int> types(num_atoms);
    std::map<std::string, int> element_map;

    initializeSpglibInput(poscarDirect, lattice, positions, types, element_map);

    // Checking if empty spheres are present in input
    for (const auto& el : poscar.elements) {
        if (el == "X" || el == "E" || el == "V" || el == "Vac") {
            std::cerr << "Warning: Empty sphere detected.\n"
                      << "SPGLIB will treat them as real atoms and symmetry may change.\n";
        }
    }

    // 3) Call spglib standardization
    int num_std = spg_standardize_cell(lattice, (double(*)[3])positions.data(), types.data(), num_atoms_in, primitive,
                                       idealize, symprec);

    if (num_std <= 0) {
        return std::nullopt;
    }

    // Determine maximum type index
    int max_type = 0;
    for (int i = 0; i < num_std; ++i)
        if (types[i] > max_type)
            max_type = types[i];

    // Create lookup vector (type -> element)
    std::vector<std::string> type_to_element(max_type + 1);

    // Fill from original element_map
    for (const auto& [el, t] : element_map) {
        if (t <= max_type)
            type_to_element[t] = el;
    }

    // 4) Build POSCAR from arrays now holding primitive cell
    POSCAR std_poscar;
    std_poscar.is_direct = true;
    std_poscar.total_atoms = num_std;
    std_poscar.comment = poscar.comment + (primitive == 1 ? " primitive cell" : " conventional cell");

    // Copy new lattice
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            std_poscar.lattice[i][j] = lattice[i][j];

    // Copy positions
    std_poscar.elements.clear();
    std_poscar.num_atoms.clear();
    std_poscar.coordinates.clear();

    // Adding atomic coordinates to kepp the elemet order as well as correct assignment of atomic position to each
    // element
    for (const auto& original_el : poscar.elements) {
        int count_for_this_el = 0;

        for (int i = 0; i < num_std; ++i) {
            if (type_to_element[types[i]] == original_el) {
                Atom a;
                a.x = positions[i][0];
                a.y = positions[i][1];
                a.z = positions[i][2];
                std_poscar.coordinates.push_back(a);
                count_for_this_el++;
            }
        }

        // Pokud prvek v nové struktuře existuje, přidáme ho do hlavičky
        if (count_for_this_el > 0) {
            std_poscar.elements.push_back(original_el);
            std_poscar.num_atoms.push_back(count_for_this_el);
        }
    }
    return std_poscar;
}

void printSymmetryInfo(const SpglibDataset& dataset, const bool& wyckoff, const bool& symoperation) {
    std::cout << "=== Symmetry Information ===\n";

    // Space group
    std::cout << "Space group number: " << dataset.spacegroup_number << "\n";
    std::cout << "International symbol: " << dataset.international_symbol << "\n";

    // Hall symbol
    std::cout << "Hall symbol: " << dataset.hall_symbol << "\n";

    // Point group
    std::cout << "Point group: " << dataset.pointgroup_symbol << "\n";

    // Symmetry operations
    std::cout << "\nNumber of symmetry operations: " << dataset.n_operations << "\n";

    if (symoperation)
        printSymmetryOperations(dataset);

    // Compute number of irreducible atoms
    std::set<int> irreducible_atoms;
    for (int i = 0; i < dataset.n_atoms; ++i)
        irreducible_atoms.insert(dataset.equivalent_atoms[i]);

    std::cout << "\nNumber of Wyckoff positions (irreducible atoms): " << irreducible_atoms.size() << "\n";

    // Print Wyckoff letters for each atom
    if (wyckoff) {
        std::cout << "Wyckoff letters: ";
        for (int i = 0; i < dataset.n_atoms; ++i) {
            char wyckoff_letter = 'a' + dataset.wyckoffs[i];  // convert 0->'a', 1->'b', ...
            std::cout << wyckoff_letter << " ";
        }
    }
    std::cout << "\n";
}

void printSymmetryOperations(const SpglibDataset& dataset) {
    for (int i = 0; i < dataset.n_operations; ++i) {
        std::cout << "Operation " << i + 1 << ":\n";
        std::cout << "  Rotation matrix:\n";
        for (int j = 0; j < 3; ++j)
            std::cout << "   " << std::setw(2) << dataset.rotations[i][j][0] << " " << std::setw(2)
                      << dataset.rotations[i][j][1] << " " << std::setw(2) << dataset.rotations[i][j][2] << "\n";
        std::cout << "  Translation vector: " << dataset.translations[i][0] << " " << dataset.translations[i][1] << " "
                  << dataset.translations[i][2] << "\n";
    }
}
