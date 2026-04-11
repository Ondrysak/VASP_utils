//! Spglib FFI bindings and high-level wrapper functions for symmetry analysis
//! and cell standardisation.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, CStr};
use std::os::raw::c_int;

use crate::poscar::{Atom, Poscar};

// ─── Spglib FFI ───────────────────────────────────────────────────────────────
//
// Bindings for spglib 2.x (tested against 2.3.0).
// The struct layout must exactly match spglib.h; #[repr(C)] ensures that Rust
// follows the same alignment / padding rules as the C compiler.

/// Raw spglib dataset, heap-allocated by `spg_get_dataset`.
///
/// Access through [`SpglibDatasetGuard`] which frees it on drop.
#[repr(C)]
pub struct SpglibDataset {
    pub spacegroup_number: c_int,
    pub hall_number: c_int,
    pub international_symbol: [c_char; 11],
    pub hall_symbol: [c_char; 17],
    pub choice: [c_char; 6],
    pub transformation_matrix: [[f64; 3]; 3],
    pub origin_shift: [f64; 3],
    pub n_operations: c_int,
    pub rotations: *mut [[c_int; 3]; 3],
    pub translations: *mut [f64; 3],
    pub n_atoms: c_int,
    pub wyckoffs: *mut c_int,
    pub site_symmetry_symbols: *mut [c_char; 7],
    pub equivalent_atoms: *mut c_int,
    pub crystallographic_orbits: *mut c_int,
    pub primitive_lattice: [[f64; 3]; 3],
    pub mapping_to_primitive: *mut c_int,
    pub n_std_atoms: c_int,
    pub std_lattice: [[f64; 3]; 3],
    pub std_types: *mut c_int,
    pub std_positions: *mut [f64; 3],
    pub std_rotation_matrix: [[f64; 3]; 3],
    pub std_mapping_to_primitive: *mut c_int,
    pub pointgroup_symbol: [c_char; 7],
}

extern "C" {
    /// Perform full symmetry analysis and return a heap-allocated dataset.
    /// Returns null on failure.
    fn spg_get_dataset(
        lattice: *const [f64; 3],
        position: *const [f64; 3],
        types: *const c_int,
        num_atom: c_int,
        symprec: f64,
    ) -> *mut SpglibDataset;

    /// Free a dataset previously returned by `spg_get_dataset`.
    fn spg_free_dataset(dataset: *mut SpglibDataset);

    /// Standardise a cell (primitive or conventional) in-place.
    /// Returns the number of atoms in the output cell, or ≤ 0 on failure.
    fn spg_standardize_cell(
        lattice: *mut [f64; 3],
        position: *mut [f64; 3],
        types: *mut c_int,
        num_atom: c_int,
        to_primitive: c_int,
        no_idealize: c_int,
        symprec: f64,
    ) -> c_int;
}

// ─── RAII guard ───────────────────────────────────────────────────────────────

/// Owns a `*mut SpglibDataset` and calls `spg_free_dataset` on drop.
pub struct SpglibDatasetGuard(*mut SpglibDataset);

impl SpglibDatasetGuard {
    /// Returns `None` if the pointer is null.
    fn new(ptr: *mut SpglibDataset) -> Option<Self> {
        if ptr.is_null() { None } else { Some(Self(ptr)) }
    }

    pub fn get(&self) -> &SpglibDataset {
        unsafe { &*self.0 }
    }
}

impl Drop for SpglibDatasetGuard {
    fn drop(&mut self) {
        unsafe { spg_free_dataset(self.0) }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Helper: read a null-terminated `[c_char; N]` as a `String`.
fn cchar_to_string(chars: &[c_char]) -> String {
    // Safety: spglib guarantees null-termination within the fixed-size array.
    unsafe { CStr::from_ptr(chars.as_ptr()).to_string_lossy().into_owned() }
}

/// Transpose POSCAR row-vector lattice → spglib column-vector layout,
/// build flat position arrays, and map element symbols to integer type ids.
///
/// Returns `(spg_lattice, positions, types, element_map)`.
fn prepare_spglib_input(
    poscar: &Poscar,
) -> ([[f64; 3]; 3], Vec<[f64; 3]>, Vec<c_int>, BTreeMap<String, c_int>) {
    // spglib wants column vectors: spg[i][j] = i-th Cartesian component of j-th vector.
    // POSCAR row vectors: poscar.lattice[i][j] = j-th component of i-th vector.
    let mut spg_lat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            spg_lat[i][j] = poscar.lattice[j][i];
        }
    }

    let n = poscar.total_atoms as usize;
    let mut positions = vec![[0.0f64; 3]; n];
    for (i, atom) in poscar.coordinates.iter().enumerate() {
        positions[i] = [atom.x, atom.y, atom.z];
    }

    let mut element_map: BTreeMap<String, c_int> = BTreeMap::new();
    let mut types = vec![0i32; n];
    let mut counter: c_int = 1;
    let mut idx = 0usize;

    for (el_raw, &count) in poscar.elements.iter().zip(poscar.num_atoms.iter()) {
        // Strip whitespace and slashes (matches C++ implementation)
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
            types[idx] = type_id;
            idx += 1;
        }
    }

    (spg_lat, positions, types, element_map)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Run spglib symmetry analysis.  The input is converted to fractional
/// coordinates internally if needed.
///
/// Returns `None` if spglib fails (null dataset pointer).
pub fn analyze_symmetry(poscar: &Poscar, symprec: f64) -> Option<SpglibDatasetGuard> {
    let mut base = poscar.clone();
    if !base.is_direct {
        base.to_direct();
    }

    let (spg_lat, positions, types, _) = prepare_spglib_input(&base);

    let ptr = unsafe {
        spg_get_dataset(
            spg_lat.as_ptr(),
            positions.as_ptr(),
            types.as_ptr(),
            base.total_atoms,
            symprec,
        )
    };

    SpglibDatasetGuard::new(ptr)
}

/// Standardise a crystal cell using spglib.
///
/// * `primitive = 1` → primitive cell; `primitive = 0` → conventional cell.
/// * `idealize = 1` → idealise lattice parameters.
///
/// Returns `None` if spglib fails.
pub fn standardize_cell(
    poscar: &Poscar,
    symprec: f64,
    primitive: c_int,
    idealize: c_int,
) -> Option<Poscar> {
    // Warn about empty spheres
    for el in &poscar.elements {
        if matches!(el.as_str(), "X" | "E" | "V" | "Vac") {
            eprintln!(
                "Warning: Empty sphere detected ('{el}'). \
                 spglib will treat it as a real atom and symmetry may change."
            );
        }
    }

    let mut base = poscar.clone();
    if !base.is_direct {
        base.to_direct();
    }

    let (mut spg_lat, mut positions, mut types, element_map) =
        prepare_spglib_input(&base);

    // Allocate space for up to 64× the input atoms (worst-case conventional cell).
    let n_in = base.total_atoms;
    let capacity = (n_in as usize) * 64;
    positions.resize(capacity, [0.0; 3]);
    types.resize(capacity, 0);

    let n_out = unsafe {
        spg_standardize_cell(
            spg_lat.as_mut_ptr(),
            positions.as_mut_ptr(),
            types.as_mut_ptr(),
            n_in,
            primitive,
            idealize,
            symprec,
        )
    };

    if n_out <= 0 {
        return None;
    }

    // Build reverse map: type_id → element symbol
    let max_type = types[..n_out as usize].iter().copied().max().unwrap_or(0) as usize;
    let mut type_to_el = vec![String::new(); max_type + 1];
    for (el, &t) in &element_map {
        if (t as usize) <= max_type {
            type_to_el[t as usize] = el.clone();
        }
    }

    // Transpose lattice back to POSCAR row-vector layout
    let mut out_lat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out_lat[i][j] = spg_lat[j][i];
        }
    }

    let suffix = if primitive == 1 { " primitive cell" } else { " conventional cell" };

    // Build output POSCAR, preserving original element order
    let mut out_elements: Vec<String> = Vec::new();
    let mut out_num_atoms: Vec<i32> = Vec::new();
    let mut out_coords: Vec<Atom> = Vec::new();

    for orig_el in &poscar.elements {
        let mut count = 0i32;
        for i in 0..n_out as usize {
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
        total_atoms: n_out,
        coordinates: out_coords,
    })
}

// ─── Printing helpers ──────────────────────────────────────────────────────────

/// Print a human-readable symmetry summary to stdout.
pub fn print_symmetry_info(dataset: &SpglibDataset, wyckoff: bool, symoperations: bool) {
    println!("=== Symmetry Information ===");
    println!("Space group number: {}", dataset.spacegroup_number);
    println!("International symbol: {}", cchar_to_string(&dataset.international_symbol));
    println!("Hall symbol: {}", cchar_to_string(&dataset.hall_symbol));
    println!("Point group: {}", cchar_to_string(&dataset.pointgroup_symbol));
    println!("\nNumber of symmetry operations: {}", dataset.n_operations);

    if symoperations {
        print_symmetry_operations(dataset);
    }

    let mut irreducible: BTreeSet<c_int> = BTreeSet::new();
    for i in 0..dataset.n_atoms as usize {
        let eq = unsafe { *dataset.equivalent_atoms.add(i) };
        irreducible.insert(eq);
    }
    println!("\nNumber of Wyckoff positions (irreducible atoms): {}", irreducible.len());

    if wyckoff {
        print!("Wyckoff letters: ");
        for i in 0..dataset.n_atoms as usize {
            let w = unsafe { *dataset.wyckoffs.add(i) };
            let letter = (b'a' + w as u8) as char;
            print!("{letter} ");
        }
        println!();
    }
}

/// Print all symmetry operations (rotation matrices + translation vectors).
pub fn print_symmetry_operations(dataset: &SpglibDataset) {
    for i in 0..dataset.n_operations as usize {
        println!("Operation {}:", i + 1);
        println!("  Rotation matrix:");
        let rot = unsafe { &*dataset.rotations.add(i) };
        for row in rot {
            println!("   {:2} {:2} {:2}", row[0], row[1], row[2]);
        }
        let tr = unsafe { &*dataset.translations.add(i) };
        println!("  Translation vector: {} {} {}", tr[0], tr[1], tr[2]);
    }
}

/// Extract the international symbol from a dataset as an owned `String`.
/// Exposed for use by [`crate::kpath`].
pub fn spglib_intl_symbol(dataset: &SpglibDataset) -> String {
    cchar_to_string(&dataset.international_symbol)
}
