#ifndef KPATH_H_INCLUDED
#define KPATH_H_INCLUDED

#include <spglib.h>

#include <optional>
#include <string>
#include <vector>

struct POSCAR;

// A labeled high-symmetry point in fractional primitive-reciprocal coordinates.
struct KPoint {
    std::string label;
    double kx, ky, kz;
};

// One path segment between two labeled points.
struct KSegment {
    std::string from;
    std::string to;
};

// Complete k-path for a given Bravais type.
struct KPath {
    std::string bravais_label;       // e.g. "cP", "cF", "hR1"
    std::vector<KPoint> points;      // all distinct labeled special points
    std::vector<KSegment> segments;  // ordered list of line segments
};

// Determine the Bravais lattice type from the standardized conventional cell
// and spglib dataset, then look up the Setyawan-Curtarolo k-path.
//
// std_conv  : standardized conventional cell (primitive=0) — used for lattice
//             parameter extraction (c/a for tI, cos(alpha) for hR).
// dataset   : spglib dataset obtained from std_conv.
//
// K-point coordinates are in fractional units of the primitive reciprocal
// lattice — pair with the primitive POSCAR from spg_standardize_cell(primitive=1).
//
// Returns nullopt for lattice types not yet supported in v1.
std::optional<KPath> getBravaisKPath(const POSCAR& std_conv, const SpglibDataset& dataset);

#endif  // KPATH_H_INCLUDED
