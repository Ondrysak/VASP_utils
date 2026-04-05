// k-path tables based on Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010).
// K-point coordinates are in fractional units of the PRIMITIVE reciprocal lattice.
// Use the output POSCAR_prim (primitive=1 from spglib) as input for VASP.

#include "kpath.h"

#include <cmath>
#include <string>

#include "poscar_file.h"

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Bravais lattice classification
// ─────────────────────────────────────────────────────────────────────────────

enum class BravaisType { cP, cF, cI, tP, tI, oP, hP, hR, Unknown };

static BravaisType detectBravais(int sg, const char* intl_sym) {
    // Extract centering letter: first non-space character of international symbol.
    char centering = '\0';
    for (int i = 0; intl_sym[i] != '\0'; ++i) {
        if (intl_sym[i] != ' ') {
            centering = intl_sym[i];
            break;
        }
    }

    if (sg >= 195 && sg <= 230) {
        if (centering == 'P')
            return BravaisType::cP;
        if (centering == 'F')
            return BravaisType::cF;
        if (centering == 'I')
            return BravaisType::cI;
    } else if (sg >= 75 && sg <= 142) {
        if (centering == 'P')
            return BravaisType::tP;
        if (centering == 'I')
            return BravaisType::tI;
    } else if (sg >= 16 && sg <= 74) {
        if (centering == 'P')
            return BravaisType::oP;
    } else if (sg >= 168 && sg <= 194) {
        return BravaisType::hP;
    } else if (sg >= 143 && sg <= 167) {
        if (centering == 'R')
            return BravaisType::hR;
    }
    return BravaisType::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lattice parameter helpers
// ─────────────────────────────────────────────────────────────────────────────

static double vecnorm(const double v[3]) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// K-path builders  (Setyawan-Curtarolo Tables 2-9, 12)
// ─────────────────────────────────────────────────────────────────────────────

static KPath kpath_cP() {
    KPath p;
    p.bravais_label = "cP";
    p.points = {{"G", 0, 0, 0}, {"M", 0.5, 0.5, 0}, {"R", 0.5, 0.5, 0.5}, {"X", 0, 0.5, 0}};
    p.segments = {{"G", "X"}, {"X", "M"}, {"M", "G"}, {"G", "R"}, {"R", "X"}, {"X", "M"}, {"M", "R"}};
    return p;
}

static KPath kpath_cF() {
    KPath p;
    p.bravais_label = "cF";
    p.points = {{"G", 0, 0, 0},         {"K", 0.375, 0.375, 0.75}, {"L", 0.5, 0.5, 0.5}, {"U", 0.625, 0.25, 0.625},
                {"W", 0.5, 0.25, 0.75}, {"X", 0.5, 0, 0.5}};
    p.segments = {{"G", "X"}, {"X", "W"}, {"W", "K"}, {"K", "G"}, {"G", "L"}, {"L", "U"},
                  {"U", "W"}, {"W", "L"}, {"L", "K"}, {"K", "U"}, {"U", "X"}};
    return p;
}

static KPath kpath_cI() {
    KPath p;
    p.bravais_label = "cI";
    p.points = {{"G", 0, 0, 0}, {"H", 0.5, -0.5, 0.5}, {"N", 0, 0, 0.5}, {"P", 0.25, 0.25, 0.25}};
    p.segments = {{"G", "H"}, {"H", "N"}, {"N", "G"}, {"G", "P"}, {"P", "H"}, {"P", "N"}};
    return p;
}

static KPath kpath_tP() {
    KPath p;
    p.bravais_label = "tP";
    p.points = {{"G", 0, 0, 0},     {"A", 0.5, 0.5, 0.5}, {"M", 0.5, 0.5, 0},
                {"R", 0, 0.5, 0.5}, {"X", 0, 0.5, 0},     {"Z", 0, 0, 0.5}};
    p.segments = {{"G", "X"}, {"X", "M"}, {"M", "G"}, {"G", "Z"}, {"Z", "R"},
                  {"R", "A"}, {"A", "Z"}, {"X", "R"}, {"M", "A"}};
    return p;
}

// tI:  eta = (1 + c²/a²) / 4   (Setyawan Table 6)
// a = |lattice[0]|, c = |lattice[2]| of the conventional tetragonal cell.
static KPath kpath_tI(const POSCAR& conv) {
    const double a = vecnorm(conv.lattice[0]);
    const double c = vecnorm(conv.lattice[2]);
    const double eta = (1.0 + (c * c) / (a * a)) / 4.0;

    KPath p;
    p.bravais_label = "tI";
    p.points = {
        {"G", 0, 0, 0},        {"M", -0.5, 0.5, 0.5},       {"N", 0, 0.5, 0}, {"P", 0.25, 0.25, 0.25}, {"X", 0, 0, 0.5},
        {"Z", eta, eta, -eta}, {"Z1", -eta, 1.0 - eta, eta}};
    p.segments = {{"G", "X"}, {"X", "M"},  {"M", "G"},  {"G", "Z"}, {"Z", "P"},
                  {"P", "N"}, {"N", "Z1"}, {"Z1", "M"}, {"X", "P"}};
    return p;
}

static KPath kpath_oP() {
    KPath p;
    p.bravais_label = "oP";
    p.points = {{"G", 0, 0, 0},     {"R", 0.5, 0.5, 0.5}, {"S", 0.5, 0.5, 0}, {"T", 0, 0.5, 0.5},
                {"U", 0.5, 0, 0.5}, {"X", 0.5, 0, 0},     {"Y", 0, 0.5, 0},   {"Z", 0, 0, 0.5}};
    p.segments = {{"G", "X"}, {"X", "S"}, {"S", "Y"}, {"Y", "G"}, {"G", "Z"}, {"Z", "U"},
                  {"U", "R"}, {"R", "T"}, {"T", "Z"}, {"Y", "T"}, {"U", "X"}, {"S", "R"}};
    return p;
}

static KPath kpath_hP() {
    KPath p;
    p.bravais_label = "hP";
    p.points = {{"G", 0, 0, 0},     {"A", 0, 0, 0.5}, {"H", 1.0 / 3.0, 1.0 / 3.0, 0.5}, {"K", 1.0 / 3.0, 1.0 / 3.0, 0},
                {"L", 0.5, 0, 0.5}, {"M", 0.5, 0, 0}};
    p.segments = {{"G", "M"}, {"M", "K"}, {"K", "G"}, {"G", "A"}, {"A", "L"},
                  {"L", "H"}, {"H", "A"}, {"L", "M"}, {"K", "H"}};
    return p;
}

// hR: two subcases based on the rhombohedral angle alpha_r.
// Using the conventional hexagonal description:
//   cos(alpha_r) = (2*c_h^2 - 3*a_h^2) / (2*c_h^2 + 6*a_h^2)
//
// hR1: alpha_r < pi/2  (cos > 0)   Setyawan Table 8
// hR2: alpha_r > pi/2  (cos < 0)   Setyawan Table 9
static KPath kpath_hR(const POSCAR& conv) {
    const double a_h = vecnorm(conv.lattice[0]);
    const double c_h = vecnorm(conv.lattice[2]);
    const double a2 = a_h * a_h;
    const double c2 = c_h * c_h;
    const double cos_alpha = (2.0 * c2 - 3.0 * a2) / (2.0 * c2 + 6.0 * a2);

    KPath p;
    if (cos_alpha > 0.0) {
        // hR1
        const double eta = (1.0 + 4.0 * cos_alpha) / (2.0 + 4.0 * cos_alpha);
        const double nu = 0.75 - eta / 2.0;
        p.bravais_label = "hR1";
        p.points = {{"G", 0, 0, 0},
                    {"B", eta, 0.5, 1.0 - eta},
                    {"B1", 0.5, 1.0 - eta, eta - 1.0},
                    {"F", 0.5, 0.5, 0},
                    {"L", 0.5, 0, 0},
                    {"L1", 0, 0, -0.5},
                    {"P", eta, nu, nu},
                    {"P1", 1.0 - nu, 1.0 - nu, 1.0 - eta},
                    {"P2", nu, nu, eta - 1.0},
                    {"Q", 1.0 - nu, nu, 0},
                    {"X", nu, 0, -nu},
                    {"Z", 0.5, 0.5, 0.5}};
        p.segments = {{"G", "L"}, {"L", "B1"}, {"B", "Z"},  {"Z", "G"}, {"G", "X"},
                      {"Q", "F"}, {"F", "P1"}, {"P1", "Z"}, {"L", "P"}};
    } else {
        // hR2
        const double eta = (1.0 + cos_alpha) / (2.0 * (1.0 - cos_alpha));
        const double nu = 0.75 - eta / 2.0;
        p.bravais_label = "hR2";
        p.points = {{"G", 0, 0, 0},
                    {"F", 0.5, -0.5, 0},
                    {"L", 0.5, 0, 0},
                    {"P", 1.0 - nu, -nu, 1.0 - nu},
                    {"P1", nu, nu - 1.0, nu - 1.0},
                    {"Q", eta, eta, eta},
                    {"Q1", 1.0 - eta, -eta, -eta},
                    {"Z", 0.5, -0.5, 0.5}};
        p.segments = {{"G", "P"},  {"P", "Z"},   {"Z", "Q"},  {"Q", "G"}, {"G", "F"},
                      {"F", "P1"}, {"P1", "Q1"}, {"Q1", "L"}, {"L", "Z"}};
    }
    return p;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────────────────

std::optional<KPath> getBravaisKPath(const POSCAR& std_conv, const SpglibDataset& dataset) {
    const BravaisType bt = detectBravais(dataset.spacegroup_number, dataset.international_symbol);

    switch (bt) {
        case BravaisType::cP:
            return kpath_cP();
        case BravaisType::cF:
            return kpath_cF();
        case BravaisType::cI:
            return kpath_cI();
        case BravaisType::tP:
            return kpath_tP();
        case BravaisType::tI:
            return kpath_tI(std_conv);
        case BravaisType::oP:
            return kpath_oP();
        case BravaisType::hP:
            return kpath_hP();
        case BravaisType::hR:
            return kpath_hR(std_conv);
        default:
            return std::nullopt;
    }
}
