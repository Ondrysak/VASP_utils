#ifndef KPATH_H_INCLUDED
#define KPATH_H_INCLUDED

#include <spglib.h>

#include <optional>
#include <string>
#include <vector>

struct POSCAR;

/**
 * @brief A labeled high-symmetry k-point in fractional primitive-reciprocal coordinates.
 *
 * Coordinates follow the Setyawan & Curtarolo (2010) convention and are expressed
 * in the basis of the **primitive** reciprocal lattice vectors.  The matching
 * real-space cell is produced by spglib with @c primitive=1.
 */
struct KPoint {
    std::string label;  ///< High-symmetry point label (e.g. "G", "X", "K").
    double kx, ky, kz;  ///< Fractional coordinates in the primitive reciprocal basis.
};

/**
 * @brief One directed segment of the k-path between two labeled points.
 */
struct KSegment {
    std::string from;  ///< Label of the starting point.
    std::string to;    ///< Label of the ending point.
};

/**
 * @brief Complete high-symmetry k-path for a given Bravais lattice type.
 *
 * Labels and coordinates follow Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010),
 * covering all 14 Bravais lattice types with their subcases (e.g. tI1/tI2, hR1/hR2,
 * oF1/oF2/oF3, mC1–5, aP1/aP2).
 */
struct KPath {
    std::string bravais_label;       ///< Bravais type string, e.g. "cF", "tI2", "hR1".
    std::vector<KPoint> points;      ///< All distinct labeled high-symmetry points.
    std::vector<KSegment> segments;  ///< Ordered list of line segments forming the path.
};

/**
 * @brief Determine the Bravais lattice type and return the corresponding k-path.
 *
 * The Bravais type is inferred from @p dataset.spacegroup_number and the centering
 * letter of @p dataset.international_symbol.  Lattice parameters required for
 * parametric point coordinates (c/a for tI, cos(α) for hR, angles for mC/aP, etc.)
 * are extracted from the standardised **conventional** cell @p std_conv.
 *
 * @param std_conv  Standardised conventional cell (primitive=0 from spglib).
 *                  Used only for lattice parameter extraction.
 * @param dataset   spglib dataset obtained from @p std_conv.
 * @return          KPath in fractional primitive-reciprocal coordinates, or
 *                  @c std::nullopt if the space group is unrecognised.
 *
 * @note Pair the returned KPOINTS with the **primitive** cell from
 *       @c spg_standardize_cell(primitive=1), not the conventional cell.
 *
 * @see Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010).
 */
/**
 * @brief Determine the Bravais lattice type and return the corresponding k-path.
 *
 * @param std_conv   Standardised conventional cell (primitive=0 from spglib).
 * @param std_prim   Primitive cell used for reciprocal-angle subcase detection
 *                   in triclinic (aP).  Should be the as-read POSCAR rather
 *                   than spglib's re-standardised primitive, because spglib
 *                   may choose a different P-1 basis that changes the angle
 *                   classification.
 * @param dataset    spglib dataset obtained from @p std_conv.
 */
std::optional<KPath> getBravaisKPath(const POSCAR& std_conv, const POSCAR& std_prim, const SpglibDataset& dataset);

/// @brief Convenience overload — uses @p std_conv as both conventional and primitive.
inline std::optional<KPath> getBravaisKPath(const POSCAR& std_conv, const SpglibDataset& dataset) {
    return getBravaisKPath(std_conv, std_conv, dataset);
}

#endif  // KPATH_H_INCLUDED
