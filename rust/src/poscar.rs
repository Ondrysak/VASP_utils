//! POSCAR/CONTCAR file I/O, coordinate conversion, atom displacement, and supercell generation.

use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, BufRead};

use faer::prelude::*;
use faer::Mat;
use rand::Rng;

// ─── Data structures ─────────────────────────────────────────────────────────

/// Single atom in a crystal structure.
///
/// Coordinates are fractional (Direct) or Cartesian depending on the parent
/// [`Poscar::is_direct`] flag.
#[derive(Clone, Debug)]
pub struct Atom {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Per-axis selective dynamics flags (T = free, F = fixed).
    pub selective_flags: [bool; 3],
}

impl Atom {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z, selective_flags: [true, true, true] }
    }
}

/// Options controlling random atom displacement.
#[derive(Clone, Debug)]
pub struct DisplacementOptions {
    /// 0-based atom indices eligible for displacement.  Empty = all atoms.
    pub candidate_indices: Vec<usize>,
    /// Wrap displaced fractional coordinates back into \[0, 1).
    pub wrap_direct: bool,
    /// Remove net Cartesian displacement after perturbation.
    pub zero_net: bool,
    /// Maximum displacement amplitude along x (Å).
    pub ampx: f64,
    /// Maximum displacement amplitude along y (Å).
    pub ampy: f64,
    /// Maximum displacement amplitude along z (Å).
    pub ampz: f64,
}

impl Default for DisplacementOptions {
    fn default() -> Self {
        Self {
            candidate_indices: Vec::new(),
            wrap_direct: false,
            zero_net: false,
            ampx: 0.01,
            ampy: 0.01,
            ampz: 0.01,
        }
    }
}

/// In-memory representation of a VASP POSCAR / CONTCAR file.
///
/// Lattice vectors are stored as **row vectors**: `lattice[i][j]` is the
/// j-th Cartesian component of the i-th basis vector.
///
/// When passing the lattice to spglib, transpose to column-vector layout —
/// see [`crate::symmetry`] for the canonical pattern.
#[derive(Clone, Debug)]
pub struct Poscar {
    pub comment: String,
    pub scale: f64,
    /// Row-vector lattice: `lattice[i]` is the i-th basis vector.
    pub lattice: [[f64; 3]; 3],
    pub elements: Vec<String>,
    pub num_atoms: Vec<i32>,
    pub selective_dynamics: bool,
    /// `true` = Direct (fractional), `false` = Cartesian.
    pub is_direct: bool,
    pub coordinates: Vec<Atom>,
    pub total_atoms: i32,
}

// ─── Parsing helpers ─────────────────────────────────────────────────────────

fn parse_flag(token: &str) -> Option<bool> {
    match token.chars().next()?.to_ascii_uppercase() {
        'T' => Some(true),
        'F' => Some(false),
        _ => None,
    }
}

// ─── Poscar implementation ───────────────────────────────────────────────────

impl Poscar {
    /// Create an empty/default POSCAR.
    pub fn new() -> Self {
        Self {
            comment: String::from("System"),
            scale: 1.0,
            lattice: [[0.0; 3]; 3],
            elements: Vec::new(),
            num_atoms: Vec::new(),
            selective_dynamics: false,
            is_direct: true,
            coordinates: Vec::new(),
            total_atoms: 0,
        }
    }

    /// Read a POSCAR/CONTCAR file from `filename`.  Returns `Err` with a
    /// human-readable message on failure.
    pub fn read(filename: &str) -> Result<Self, String> {
        let file = fs::File::open(filename)
            .map_err(|e| format!("I/O error while reading {filename}: {e}"))?;
        let reader = io::BufReader::new(file);
        let mut lines = reader.lines().enumerate();

        macro_rules! next_line {
            ($ctx:expr) => {{
                let (ln, result) = lines
                    .next()
                    .ok_or_else(|| format!("Unexpected end of file while reading {}", $ctx))?;
                let line = result
                    .map_err(|e| format!("I/O error at line {}: {e}", ln + 1))?;
                (ln + 1, line)
            }};
        }

        // Line 1: comment
        let (_, comment) = next_line!("comment");

        // Line 2: scale
        let (ln, scale_line) = next_line!("scaling factor");
        let scale: f64 = scale_line
            .trim()
            .parse()
            .map_err(|_| format!("Parse error at line {ln}: invalid scaling factor"))?;
        if scale <= 0.0 {
            return Err(format!("Semantic error at line {ln}: scaling factor must be positive"));
        }

        // Lines 3–5: lattice vectors
        let mut lattice = [[0.0f64; 3]; 3];
        for row in &mut lattice {
            let (ln, line) = next_line!("lattice vector");
            let vals: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect::<Result<_, _>>()
                .map_err(|_| format!("Parse error at line {ln}: invalid lattice vector"))?;
            if vals.len() != 3 {
                return Err(format!(
                    "Parse error at line {ln}: lattice vector must have 3 components"
                ));
            }
            row.copy_from_slice(&vals);
        }

        // Line 6: element symbols
        let (ln, elem_line) = next_line!("element symbols");
        let elements: Vec<String> =
            elem_line.split_whitespace().map(str::to_owned).collect();
        if elements.is_empty() {
            return Err(format!(
                "Parse error at line {ln}: element symbols line is empty"
            ));
        }

        // Line 7: atom counts
        let (ln, count_line) = next_line!("atom counts");
        let num_atoms: Vec<i32> = count_line
            .split_whitespace()
            .map(|s| s.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|_| format!("Parse error at line {ln}: invalid atom counts"))?;
        if num_atoms.is_empty() {
            return Err(format!(
                "Parse error at line {ln}: atom counts line is empty"
            ));
        }
        if elements.len() != num_atoms.len() {
            return Err(format!(
                "Semantic error at line {ln}: number of element symbols ({}) \
                 does not match atom counts ({})",
                elements.len(),
                num_atoms.len()
            ));
        }

        let total_atoms: i32 = num_atoms.iter().sum();

        // Line 8: optional "Selective Dynamics" or coordinate mode
        let (ln, mode_line) = next_line!("coordinate mode");
        let (selective_dynamics, coord_mode_line, _ln) =
            if mode_line.trim_start().starts_with(['S', 's']) {
                let (ln2, next) = next_line!("coordinate mode");
                (true, next, ln2)
            } else {
                (false, mode_line, ln)
            };

        let is_direct = match coord_mode_line.trim_start().chars().next() {
            Some('D') | Some('d') => true,
            Some('C') | Some('c') | Some('K') | Some('k') => false,
            _ => {
                return Err(format!(
                    "Semantic error: coordinate mode must be Direct or Cartesian, \
                     got '{}'",
                    coord_mode_line.trim()
                ))
            }
        };

        // Coordinate lines
        let mut coordinates = Vec::with_capacity(total_atoms as usize);
        for i in 0..total_atoms {
            let (ln, line) = next_line!(&format!("atom {i} coordinates"));
            let mut parts = line.split_whitespace();

            let x: f64 = parts
                .next()
                .ok_or_else(|| format!("Parse error at line {ln}: missing x"))?
                .parse()
                .map_err(|_| format!("Parse error at line {ln}: invalid x coordinate"))?;
            let y: f64 = parts
                .next()
                .ok_or_else(|| format!("Parse error at line {ln}: missing y"))?
                .parse()
                .map_err(|_| format!("Parse error at line {ln}: invalid y coordinate"))?;
            let z: f64 = parts
                .next()
                .ok_or_else(|| format!("Parse error at line {ln}: missing z"))?
                .parse()
                .map_err(|_| format!("Parse error at line {ln}: invalid z coordinate"))?;

            let mut atom = Atom::new(x, y, z);
            if selective_dynamics {
                let fx = parts
                    .next()
                    .ok_or_else(|| format!("Parse error at line {ln}: missing selective flag x"))?;
                let fy = parts
                    .next()
                    .ok_or_else(|| format!("Parse error at line {ln}: missing selective flag y"))?;
                let fz = parts
                    .next()
                    .ok_or_else(|| format!("Parse error at line {ln}: missing selective flag z"))?;
                atom.selective_flags[0] = parse_flag(fx).ok_or_else(|| {
                    format!("Parse error at line {ln}: invalid selective flag '{fx}'")
                })?;
                atom.selective_flags[1] = parse_flag(fy).ok_or_else(|| {
                    format!("Parse error at line {ln}: invalid selective flag '{fy}'")
                })?;
                atom.selective_flags[2] = parse_flag(fz).ok_or_else(|| {
                    format!("Parse error at line {ln}: invalid selective flag '{fz}'")
                })?;
            }

            coordinates.push(atom);
        }

        let mut poscar = Self {
            comment,
            scale,
            lattice,
            elements,
            num_atoms,
            selective_dynamics,
            is_direct,
            coordinates,
            total_atoms,
        };
        poscar.set_scale_to_1();
        Ok(poscar)
    }

    /// Write a POSCAR file to `filename`.
    pub fn write(&self, filename: &str) -> Result<(), String> {
        if std::path::Path::new(filename).exists() {
            eprintln!(
                "Warning: file \"{filename}\" already exists and will be overwritten."
            );
        }

        let mut buf = String::new();
        writeln!(buf, "{}", self.comment).unwrap();
        writeln!(buf, "{:.10}", self.scale).unwrap();
        for row in &self.lattice {
            writeln!(buf, "{:.10} {:.10} {:.10}", row[0], row[1], row[2]).unwrap();
        }
        // Element symbols
        for el in &self.elements {
            write!(buf, "{el} ").unwrap();
        }
        writeln!(buf).unwrap();
        // Atom counts
        for n in &self.num_atoms {
            write!(buf, "{n} ").unwrap();
        }
        writeln!(buf).unwrap();
        if self.selective_dynamics {
            writeln!(buf, "Selective Dynamics").unwrap();
        }
        writeln!(buf, "{}", if self.is_direct { "Direct" } else { "Cartesian" }).unwrap();
        for atom in &self.coordinates {
            write!(buf, "{:.10} {:.10} {:.10}", atom.x, atom.y, atom.z).unwrap();
            if self.selective_dynamics {
                write!(
                    buf,
                    " {} {} {}",
                    if atom.selective_flags[0] { "T" } else { "F" },
                    if atom.selective_flags[1] { "T" } else { "F" },
                    if atom.selective_flags[2] { "T" } else { "F" },
                )
                .unwrap();
            }
            writeln!(buf).unwrap();
        }

        fs::write(filename, buf)
            .map_err(|e| format!("Error: failed writing to {filename}: {e}"))
    }

    /// Write an ecalj/Questaal `ctrls` file.  The POSCAR must be in Cartesian
    /// coordinates; returns `Err` otherwise.
    pub fn write_ctrls(&self, filename: &str) -> Result<(), String> {
        if self.is_direct {
            return Err(
                "Error: only POSCAR in Cartesian coordinates can be written to ctrls format"
                    .into(),
            );
        }
        if std::path::Path::new(filename).exists() {
            eprintln!(
                "Warning: file \"{filename}\" already exists and will be overwritten."
            );
        }

        const ANG_TO_BOHR: f64 = 1.889_726_125;

        let mut buf = String::new();
        writeln!(buf, "STRUC    ALAT={}", self.scale * ANG_TO_BOHR).unwrap();
        writeln!(buf, "         PLAT=  {:.10} {:.10} {:.10} ",
            self.lattice[0][0], self.lattice[0][1], self.lattice[0][2]).unwrap();
        for i in 1..3 {
            writeln!(buf, "               {:.10} {:.10} {:.10} ",
                self.lattice[i][0], self.lattice[i][1], self.lattice[i][2]).unwrap();
        }
        writeln!(buf, "SITE ").unwrap();

        let mut atom_idx = 0usize;
        for (i, el) in self.elements.iter().enumerate() {
            for _ in 0..self.num_atoms[i] {
                if atom_idx >= self.coordinates.len() {
                    break;
                }
                let a = &self.coordinates[atom_idx];
                writeln!(
                    buf,
                    "      ATOM={:<3}  POS={:.10} {:.10} {:.10}",
                    el,
                    a.x * ANG_TO_BOHR,
                    a.y * ANG_TO_BOHR,
                    a.z * ANG_TO_BOHR,
                )
                .unwrap();
                atom_idx += 1;
            }
        }

        fs::write(filename, buf)
            .map_err(|e| format!("Error: failed writing to {filename}: {e}"))
    }

    // ── Coordinate conversion ─────────────────────────────────────────────────

    /// Convert from fractional (Direct) to Cartesian.
    ///
    /// Uses faer for the 3×3 matrix–vector product:
    /// **r**_cart = L · **r**_frac   where L has basis vectors as rows.
    pub fn to_cartesian(&mut self) {
        if !self.is_direct {
            return;
        }
        let lat = Self::lattice_mat(&self.lattice);
        for atom in &mut self.coordinates {
            let (x, y, z) = Self::matvec(&lat, atom.x, atom.y, atom.z);
            atom.x = x;
            atom.y = y;
            atom.z = z;
        }
        self.is_direct = false;
    }

    /// Convert from Cartesian to fractional (Direct).
    ///
    /// Uses faer LU to solve L · **r**_frac = **r**_cart.
    pub fn to_direct(&mut self) {
        if self.is_direct {
            return;
        }
        let lat = Self::lattice_mat(&self.lattice);
        let lu = lat.partial_piv_lu();
        for atom in &mut self.coordinates {
            let rhs = Self::col3(atom.x, atom.y, atom.z);
            let sol = lu.solve(&rhs);
            atom.x = sol.read(0);
            atom.y = sol.read(1);
            atom.z = sol.read(2);
        }
        self.is_direct = true;
    }

    // ── Displacement ──────────────────────────────────────────────────────────

    /// Convenience wrapper: displace `n_atoms` randomly chosen atoms with an
    /// isotropic amplitude.
    pub fn displace_atoms_simple(
        &mut self,
        n_atoms: usize,
        amplitude: f64,
        rng: &mut impl Rng,
    ) {
        let opts = DisplacementOptions {
            ampx: amplitude,
            ampy: amplitude,
            ampz: amplitude,
            ..Default::default()
        };
        self.displace_atoms(n_atoms, &opts, rng);
    }

    /// Displace atoms according to the full option set.
    ///
    /// # Steps
    /// 1. Build candidate set (all atoms, or filtered by `options.candidate_indices`).
    /// 2. Shuffle and pick up to `n_atoms`.
    /// 3. Temporarily convert to Cartesian, apply random displacements.
    /// 4. Optionally remove net translation (`zero_net`).
    /// 5. Convert back; optionally wrap fractional coords to \[0, 1).
    pub fn displace_atoms(
        &mut self,
        n_atoms: usize,
        options: &DisplacementOptions,
        rng: &mut impl Rng,
    ) {
        if n_atoms == 0 || self.total_atoms == 0 {
            return;
        }

        // Build and validate candidate set
        let mut candidates: Vec<usize> = if options.candidate_indices.is_empty() {
            (0..self.coordinates.len()).collect()
        } else {
            options
                .candidate_indices
                .iter()
                .filter(|&&i| i < self.coordinates.len())
                .copied()
                .collect()
        };

        if candidates.is_empty() {
            return;
        }

        // Shuffle
        for i in (1..candidates.len()).rev() {
            let j = rng.gen_range(0..=i);
            candidates.swap(i, j);
        }

        let selected_count = n_atoms.min(candidates.len());
        let selected = &candidates[..selected_count];

        // Work in Cartesian
        let was_direct = self.is_direct;
        if was_direct {
            self.to_cartesian();
        }

        let mut net = (0.0f64, 0.0f64, 0.0f64);
        let mut displacements: Vec<(usize, f64, f64, f64)> =
            Vec::with_capacity(selected_count);

        for &idx in selected {
            let dx = rng.gen_range(-options.ampx..=options.ampx);
            let dy = rng.gen_range(-options.ampy..=options.ampy);
            let dz = rng.gen_range(-options.ampz..=options.ampz);
            self.coordinates[idx].x += dx;
            self.coordinates[idx].y += dy;
            self.coordinates[idx].z += dz;
            net.0 += dx;
            net.1 += dy;
            net.2 += dz;
            displacements.push((idx, dx, dy, dz));
        }

        // Remove net translation
        if options.zero_net && !displacements.is_empty() {
            let inv = 1.0 / displacements.len() as f64;
            let (mx, my, mz) = (net.0 * inv, net.1 * inv, net.2 * inv);
            for &(idx, _, _, _) in &displacements {
                self.coordinates[idx].x -= mx;
                self.coordinates[idx].y -= my;
                self.coordinates[idx].z -= mz;
            }
        }

        // Restore original coordinate system
        if was_direct {
            self.to_direct();
        }

        // Wrap fractional coordinates to [0, 1)
        if options.wrap_direct {
            let started_direct = self.is_direct;
            if !self.is_direct {
                self.to_direct();
            }
            for atom in &mut self.coordinates {
                atom.x = wrap01(atom.x);
                atom.y = wrap01(atom.y);
                atom.z = wrap01(atom.z);
            }
            if !started_direct {
                self.to_cartesian();
            }
        }
    }

    // ── Supercell ─────────────────────────────────────────────────────────────

    /// Build an `nx × ny × nz` diagonal supercell.
    pub fn make_supercell(&self, nx: i32, ny: i32, nz: i32) -> Self {
        let mut base = self.clone();
        if !base.is_direct {
            base.to_direct();
        }

        let multiplier = nx * ny * nz;
        let mut sc = Self::new();
        sc.comment = format!("{}_supercell", base.comment);
        sc.scale = base.scale;
        sc.is_direct = true;
        sc.elements = base.elements.clone();

        // Expand lattice
        sc.lattice[0] = scale_row(base.lattice[0], nx as f64);
        sc.lattice[1] = scale_row(base.lattice[1], ny as f64);
        sc.lattice[2] = scale_row(base.lattice[2], nz as f64);

        // Replicate atom counts
        for &n in &base.num_atoms {
            sc.num_atoms.push(n * multiplier);
            sc.total_atoms += n * multiplier;
        }

        // Replicate positions
        sc.coordinates.reserve(sc.total_atoms as usize);
        let mut offset = 0usize;
        for e in 0..base.elements.len() {
            for a in 0..base.num_atoms[e] as usize {
                let orig = &base.coordinates[offset + a];
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            sc.coordinates.push(Atom {
                                x: (orig.x + i as f64) / nx as f64,
                                y: (orig.y + j as f64) / ny as f64,
                                z: (orig.z + k as f64) / nz as f64,
                                selective_flags: [true, true, true],
                            });
                        }
                    }
                }
            }
            offset += base.num_atoms[e] as usize;
        }

        sc
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Return an ordered list of element symbols, one entry per atom.
    pub fn atom_species(&self) -> Vec<String> {
        let mut species = Vec::with_capacity(self.total_atoms as usize);
        for (el, &n) in self.elements.iter().zip(self.num_atoms.iter()) {
            for _ in 0..n {
                species.push(el.clone());
            }
        }
        species.truncate(self.coordinates.len());
        species
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn set_scale_to_1(&mut self) {
        if (self.scale - 1.0).abs() < 1e-12 {
            return;
        }
        for row in &mut self.lattice {
            for v in row.iter_mut() {
                *v *= self.scale;
            }
        }
        if !self.is_direct {
            for atom in &mut self.coordinates {
                atom.x *= self.scale;
                atom.y *= self.scale;
                atom.z *= self.scale;
            }
        }
        self.scale = 1.0;
    }

    /// Build a faer 3×3 matrix from row-vector lattice data.
    fn lattice_mat(lattice: &[[f64; 3]; 3]) -> Mat<f64> {
        Mat::from_fn(3, 3, |i, j| lattice[i][j])
    }

    /// Build a faer column vector from three components.
    fn col3(x: f64, y: f64, z: f64) -> faer::Col<f64> {
        faer::Col::from_fn(3, |i| [x, y, z][i])
    }

    /// Matrix–vector product using faer, returns (x, y, z).
    fn matvec(m: &Mat<f64>, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let v = Self::col3(x, y, z);
        let r = m * &v;
        (r.read(0), r.read(1), r.read(2))
    }
}

impl Default for Poscar {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Free helpers ─────────────────────────────────────────────────────────────

fn wrap01(v: f64) -> f64 {
    let w = v % 1.0;
    if w < 0.0 { w + 1.0 } else { w }
}

fn scale_row(row: [f64; 3], s: f64) -> [f64; 3] {
    [row[0] * s, row[1] * s, row[2] * s]
}
