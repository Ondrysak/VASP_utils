//! Symmetry analysis and cell standardisation via the [`spglib`] Rust crate.
//!
//! [`spglib::dataset::Dataset`] is re-exported as the primary result type so
//! callers do not need a direct dependency on the `spglib` crate.

use std::collections::{BTreeMap, BTreeSet};
use std::os::raw::c_int;

use spglib::cell::Cell;
pub use spglib::dataset::Dataset;
use spglib_sys as ffi;

use crate::poscar::{Atom, Poscar};

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build the spglib `Cell` from a POSCAR that is already in fractional
/// coordinates.  Also returns the element → integer-type mapping so callers
/// can reconstruct element names from type ids.
///
/// spglib uses **column** vectors (`lat[i][j]` = i-th Cartesian component of
/// the j-th basis vector), whereas POSCAR uses **row** vectors, so we
/// transpose here.
fn poscar_to_cell(poscar: &Poscar) -> (Cell, BTreeMap<String, i32>) {
    let mut lat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            lat[i][j] = poscar.lattice[j][i];
        }
    }

    let positions: Vec<[f64; 3]> = poscar
        .coordinates
        .iter()
        .map(|a| [a.x, a.y, a.z])
        .collect();

    let (types, element_map) = build_types(poscar);
    (Cell::new(&lat, &positions, &types), element_map)
}

/// Assign a unique integer type id to each distinct element symbol.
fn build_types(poscar: &Poscar) -> (Vec<i32>, BTreeMap<String, i32>) {
    let mut element_map: BTreeMap<String, i32> = BTreeMap::new();
    let mut counter: i32 = 1;
    let mut types = Vec::with_capacity(poscar.total_atoms as usize);

    for (el_raw, &count) in poscar.elements.iter().zip(poscar.num_atoms.iter()) {
        // Strip whitespace and slashes (mirrors the C++ implementation)
        let el: String = el_raw
            .chars()
            .filter(|&c| c != '/' && !c.is_whitespace())
            .collect();

        let type_id = *element_map.entry(el).or_insert_with(|| {
            let id = counter;
            counter += 1;
            id
        });
        for _ in 0..count {
            types.push(type_id);
        }
    }

    (types, element_map)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Run spglib symmetry analysis and return a [`Dataset`] with all fields
/// decoded into owned Rust types.
///
/// Returns `None` if spglib returns a null dataset (symmetry search failed).
pub fn analyze_symmetry(poscar: &Poscar, symprec: f64) -> Option<Dataset> {
    let mut base = poscar.clone();
    if !base.is_direct {
        base.to_direct();
    }

    let (cell, _) = poscar_to_cell(&base);

    // Call spg_get_dataset through spglib_sys so we can check for null before
    // handing the pointer to Dataset::try_from (which would panic on null).
    let raw = unsafe {
        ffi::spg_get_dataset(
            cell.lattice.as_ptr() as *mut [f64; 3],
            cell.positions.as_ptr() as *mut [f64; 3],
            cell.types.as_ptr() as *const c_int,
            cell.positions.len() as c_int,
            symprec,
        )
    };

    if raw.is_null() {
        return None;
    }

    // Dataset::try_from transfers ownership of spglib's inner arrays into Vecs.
    Dataset::try_from(raw).ok()
}

/// Standardise a crystal cell using spglib.
///
/// * `primitive = true`  → primitive cell
/// * `primitive = false` → conventional cell
/// * `idealize`          → whether to idealise lattice parameters
///
/// Returns `None` if spglib fails.
///
/// # Note on memory
/// `spglib::cell::Cell::standardize` does not expose the new atom count and
/// does not pre-allocate for larger conventional cells, so we call
/// `spglib_sys::spg_standardize_cell` directly here with a properly-sized
/// buffer.
pub fn standardize_cell(
    poscar: &Poscar,
    symprec: f64,
    primitive: bool,
    idealize: bool,
) -> Option<Poscar> {
    // Warn about empty spheres
    for el in &poscar.elements {
        if matches!(el.as_str(), "X" | "E" | "V" | "Vac") {
            eprintln!(
                "Warning: empty sphere '{el}' detected; \
                 spglib will treat it as a real atom."
            );
        }
    }

    let mut base = poscar.clone();
    if !base.is_direct {
        base.to_direct();
    }

    let (cell_base, element_map) = poscar_to_cell(&base);
    let n_in = base.total_atoms as usize;

    // Pre-allocate 64× the input for worst-case conventional cell expansion.
    let capacity = n_in * 64;
    let mut lat = cell_base.lattice;
    let mut positions: Vec<[f64; 3]> = cell_base.positions.clone();
    positions.resize(capacity, [0.0; 3]);
    let mut types: Vec<i32> = cell_base.types.clone();
    types.resize(capacity, 0);

    let n_out = unsafe {
        ffi::spg_standardize_cell(
            lat.as_mut_ptr(),
            positions.as_mut_ptr(),
            types.as_mut_ptr() as *mut c_int,
            n_in as c_int,
            if primitive { 1 } else { 0 },
            if idealize { 1 } else { 0 },
            symprec,
        )
    };

    if n_out <= 0 {
        return None;
    }
    let n_out = n_out as usize;

    // Build reverse map: type_id → element symbol
    let max_type = types[..n_out].iter().copied().max().unwrap_or(0) as usize;
    let mut type_to_el = vec![String::new(); max_type + 1];
    for (el, &t) in &element_map {
        if (t as usize) <= max_type {
            type_to_el[t as usize] = el.clone();
        }
    }

    // Transpose spglib column-vector lattice back to POSCAR row-vector layout
    let mut out_lat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out_lat[i][j] = lat[j][i];
        }
    }

    let suffix = if primitive { " primitive cell" } else { " conventional cell" };

    // Build output POSCAR, preserving the original element order
    let mut out_elements: Vec<String> = Vec::new();
    let mut out_num_atoms: Vec<i32> = Vec::new();
    let mut out_coords: Vec<Atom> = Vec::new();

    for orig_el in &poscar.elements {
        let mut count = 0i32;
        for i in 0..n_out {
            if type_to_el[types[i] as usize] == *orig_el {
                out_coords.push(Atom::new(positions[i][0], positions[i][1], positions[i][2]));
                count += 1;
            }
        }
        if count > 0 {
            out_elements.push(orig_el.clone());
            out_num_atoms.push(count);
        }
    }

    Some(Poscar {
        comment: format!("{}{suffix}", poscar.comment),
        scale: 1.0,
        lattice: out_lat,
        elements: out_elements,
        num_atoms: out_num_atoms,
        selective_dynamics: false,
        is_direct: true,
        total_atoms: n_out as i32,
        coordinates: out_coords,
    })
}

// ─── Printing ─────────────────────────────────────────────────────────────────

/// Print a human-readable symmetry summary to stdout.
pub fn print_symmetry_info(dataset: &Dataset, wyckoff: bool, symoperations: bool) {
    println!("=== Symmetry Information ===");
    println!("Space group number: {}", dataset.spacegroup_number);
    println!("International symbol: {}", dataset.international_symbol);
    println!("Hall symbol: {}", dataset.hall_symbol);
    println!("Point group: {}", dataset.pointgroup_symbol);
    println!("\nNumber of symmetry operations: {}", dataset.n_operations);

    if symoperations {
        print_symmetry_operations(dataset);
    }

    let irreducible: BTreeSet<i32> = dataset.equivalent_atoms.iter().copied().collect();
    println!(
        "\nNumber of Wyckoff positions (irreducible atoms): {}",
        irreducible.len()
    );

    if wyckoff {
        print!("Wyckoff letters: ");
        for &w in &dataset.wyckoffs {
            print!("{} ", (b'a' + w as u8) as char);
        }
        println!();
    }
}

/// Print all symmetry operations (rotation matrices + translation vectors).
pub fn print_symmetry_operations(dataset: &Dataset) {
    for (i, (rot, tr)) in dataset
        .rotations
        .iter()
        .zip(dataset.translations.iter())
        .enumerate()
    {
        println!("Operation {}:", i + 1);
        println!("  Rotation matrix:");
        for row in rot {
            println!("   {:2} {:2} {:2}", row[0], row[1], row[2]);
        }
        println!("  Translation vector: {} {} {}", tr[0], tr[1], tr[2]);
    }
}
