//! High-symmetry k-path tables.
//!
//! Tables follow Setyawan & Curtarolo, *Comput. Mater. Sci.* **49**, 299 (2010).
//! K-point coordinates are in fractional units of the **primitive** reciprocal lattice.
//! Bravais-lattice classification is performed via the spglib space-group number and
//! the centering letter of the international symbol.

use crate::poscar::Poscar;
use crate::symmetry::SpglibDataset;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A labeled high-symmetry k-point.
#[derive(Clone, Debug)]
pub struct KPoint {
    pub label: String,
    pub kx: f64,
    pub ky: f64,
    pub kz: f64,
}

impl KPoint {
    fn new(label: &str, kx: f64, ky: f64, kz: f64) -> Self {
        Self { label: label.to_owned(), kx, ky, kz }
    }
}

/// A directed segment of the k-path between two labeled points.
#[derive(Clone, Debug)]
pub struct KSegment {
    pub from: String,
    pub to: String,
}

impl KSegment {
    fn new(from: &str, to: &str) -> Self {
        Self { from: from.to_owned(), to: to.to_owned() }
    }
}

/// Complete high-symmetry k-path for one Bravais-lattice type.
#[derive(Clone, Debug)]
pub struct KPath {
    pub bravais_label: String,
    pub points: Vec<KPoint>,
    pub segments: Vec<KSegment>,
}

// ─── Bravais-lattice detection ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bravais {
    CP, CF, CI,
    TP, TI,
    OP, OF, OI, OC,
    HP, HR,
    MP, MC,
    AP,
}

fn detect_bravais(sg: i32, intl_sym: &str) -> Option<Bravais> {
    let centering = intl_sym.chars().find(|c| !c.is_whitespace())?;

    Some(match sg {
        195..=230 => match centering {
            'P' => Bravais::CP,
            'F' => Bravais::CF,
            'I' => Bravais::CI,
            _ => return None,
        },
        75..=142 => match centering {
            'P' => Bravais::TP,
            'I' => Bravais::TI,
            _ => return None,
        },
        16..=74 => match centering {
            'P' => Bravais::OP,
            'F' => Bravais::OF,
            'I' => Bravais::OI,
            'C' | 'A' | 'B' => Bravais::OC,
            _ => return None,
        },
        168..=194 => Bravais::HP,
        143..=167 => match centering {
            'R' => Bravais::HR,
            _ => Bravais::HP,
        },
        3..=15 => match centering {
            'P' => Bravais::MP,
            'C' | 'A' | 'B' | 'I' => Bravais::MC,
            _ => return None,
        },
        1..=2 => Bravais::AP,
        _ => return None,
    })
}

// ─── Vector math helpers ──────────────────────────────────────────────────────

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cos = (dot(a, b) / (norm(a) * norm(b))).clamp(-1.0, 1.0);
    cos.acos()
}

/// Compute reciprocal lattice vectors (without 2π factor) from POSCAR row vectors.
/// b1 = (a2 × a3) / V,  b2 = (a3 × a1) / V,  b3 = (a1 × a2) / V
fn reciprocal(conv: &Poscar) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let a1 = conv.lattice[0];
    let a2 = conv.lattice[1];
    let a3 = conv.lattice[2];

    let a2xa3 = cross(a2, a3);
    let a3xa1 = cross(a3, a1);
    let a1xa2 = cross(a1, a2);
    let v = dot(a1, a2xa3);

    let b1 = [a2xa3[0] / v, a2xa3[1] / v, a2xa3[2] / v];
    let b2 = [a3xa1[0] / v, a3xa1[1] / v, a3xa1[2] / v];
    let b3 = [a1xa2[0] / v, a1xa2[1] / v, a1xa2[2] / v];
    (b1, b2, b3)
}

// ─── K-path builders (Setyawan-Curtarolo Tables 2-14) ────────────────────────

fn pts(data: &[(&str, f64, f64, f64)]) -> Vec<KPoint> {
    data.iter().map(|(l, x, y, z)| KPoint::new(l, *x, *y, *z)).collect()
}

fn segs(data: &[(&str, &str)]) -> Vec<KSegment> {
    data.iter().map(|(f, t)| KSegment::new(f, t)).collect()
}

fn kpath_cp() -> KPath {
    KPath {
        bravais_label: "cP".into(),
        points: pts(&[("G",0.,0.,0.),("M",0.5,0.5,0.),("R",0.5,0.5,0.5),("X",0.,0.5,0.)]),
        segments: segs(&[("G","X"),("X","M"),("M","G"),("G","R"),("R","X"),("X","M"),("M","R")]),
    }
}

fn kpath_cf() -> KPath {
    KPath {
        bravais_label: "cF".into(),
        points: pts(&[
            ("G",0.,0.,0.),("K",0.375,0.375,0.75),("L",0.5,0.5,0.5),
            ("U",0.625,0.25,0.625),("W",0.5,0.25,0.75),("X",0.5,0.,0.5),
        ]),
        segments: segs(&[
            ("G","X"),("X","W"),("W","K"),("K","G"),("G","L"),("L","U"),
            ("U","W"),("W","L"),("L","K"),("K","U"),("U","X"),
        ]),
    }
}

fn kpath_ci() -> KPath {
    KPath {
        bravais_label: "cI".into(),
        points: pts(&[("G",0.,0.,0.),("H",0.5,-0.5,0.5),("N",0.,0.,0.5),("P",0.25,0.25,0.25)]),
        segments: segs(&[("G","H"),("H","N"),("N","G"),("G","P"),("P","H"),("P","N")]),
    }
}

fn kpath_tp() -> KPath {
    KPath {
        bravais_label: "tP".into(),
        points: pts(&[
            ("G",0.,0.,0.),("A",0.5,0.5,0.5),("M",0.5,0.5,0.),
            ("R",0.,0.5,0.5),("X",0.,0.5,0.),("Z",0.,0.,0.5),
        ]),
        segments: segs(&[
            ("G","X"),("X","M"),("M","G"),("G","Z"),("Z","R"),("R","A"),("A","Z"),("X","R"),("M","A"),
        ]),
    }
}

fn kpath_ti(conv: &Poscar) -> KPath {
    let a = norm(conv.lattice[0]);
    let c = norm(conv.lattice[2]);
    let a2 = a * a;
    let c2 = c * c;

    if c <= a {
        let h = (1.0 + c2 / a2) / 4.0;
        KPath {
            bravais_label: "tI1".into(),
            points: pts(&[
                ("G",0.,0.,0.),("M",-0.5,0.5,0.5),("N",0.,0.5,0.),("P",0.25,0.25,0.25),
                ("X",0.,0.,0.5),("Z",h,h,-h),("Z0",-h,1.0-h,h),
            ]),
            segments: segs(&[
                ("G","X"),("X","M"),("M","G"),("G","Z"),("Z0","M"),("X","P"),("P","N"),("N","G"),
            ]),
        }
    } else {
        let h = (1.0 + a2 / c2) / 4.0;
        let z = a2 / (2.0 * c2);
        KPath {
            bravais_label: "tI2".into(),
            points: pts(&[
                ("G",0.,0.,0.),("M",0.5,0.5,-0.5),("N",0.,0.5,0.),("P",0.25,0.25,0.25),
                ("X",0.,0.,0.5),("S0",-h,h,h),("S",h,1.0-h,-h),("R",-z,z,0.5),("Gp",0.5,0.5,-z),
            ]),
            segments: segs(&[
                ("G","X"),("X","P"),("P","N"),("N","G"),("G","M"),("M","S"),("S0","G"),("X","R"),("Gp","M"),
            ]),
        }
    }
}

fn kpath_op() -> KPath {
    KPath {
        bravais_label: "oP".into(),
        points: pts(&[
            ("G",0.,0.,0.),("R",0.5,0.5,0.5),("S",0.5,0.5,0.),("T",0.,0.5,0.5),
            ("U",0.5,0.,0.5),("X",0.5,0.,0.),("Y",0.,0.5,0.),("Z",0.,0.,0.5),
        ]),
        segments: segs(&[
            ("G","X"),("X","S"),("S","Y"),("Y","G"),("G","Z"),("Z","U"),
            ("U","R"),("R","T"),("T","Z"),("Y","T"),("U","X"),("S","R"),
        ]),
    }
}

fn kpath_of(conv: &Poscar) -> KPath {
    let a = norm(conv.lattice[0]);
    let b = norm(conv.lattice[1]);
    let c = norm(conv.lattice[2]);
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;

    let inv_a2 = 1.0 / a2;
    let inv_b2 = 1.0 / b2;
    let inv_c2 = 1.0 / c2;

    if inv_a2 > inv_b2 + inv_c2 {
        // oF1
        let zeta = (1.0 + a2 / b2 - a2 / c2) / 4.0;
        let eta  = (1.0 + a2 / b2 + a2 / c2) / 4.0;
        KPath {
            bravais_label: "oF1".into(),
            points: pts(&[
                ("G",0.,0.,0.),("A",0.5,0.5+zeta,zeta),("A1",0.5,0.5-zeta,1.0-zeta),
                ("L",0.5,0.5,0.5),("T",1.,0.5,0.5),("X",0.,eta,eta),
                ("X1",1.,1.-eta,1.-eta),("Y",0.5,0.,0.5),("Z",0.5,0.5,0.),
            ]),
            segments: segs(&[
                ("G","Y"),("Y","T"),("T","Z"),("Z","G"),("G","X"),("X","A1"),
                ("A1","Y"),("T","X1"),("X","A"),("A","Z"),("L","G"),
            ]),
        }
    } else if inv_a2 < inv_b2 + inv_c2 {
        // oF2
        let phi   = (1.0 + c2 / b2 - c2 / a2) / 4.0;
        let eta   = (1.0 + a2 / b2 - a2 / c2) / 4.0;
        let delta = (1.0 + b2 / a2 - b2 / c2) / 4.0;
        KPath {
            bravais_label: "oF2".into(),
            points: pts(&[
                ("G",0.,0.,0.),("C",0.5,0.5-eta,1.0-eta),("C1",0.5,0.5+eta,eta),
                ("D",0.5-delta,0.5,1.0-delta),("D1",0.5+delta,0.5,delta),
                ("L",0.5,0.5,0.5),("H",1.0-phi,0.5-phi,0.5),("H1",phi,0.5+phi,0.5),
                ("X",0.,0.5,0.5),("Y",0.5,0.,0.5),("Z",0.5,0.5,0.),
            ]),
            segments: segs(&[
                ("G","Y"),("Y","C"),("C","D"),("D","X"),("X","G"),("G","Z"),("Z","D1"),
                ("D1","H"),("H","C"),("C1","Z"),("X","H1"),("H","Y"),("L","G"),
            ]),
        }
    } else {
        // oF3 (degenerate)
        let zeta = (1.0 + a2 / b2 - a2 / c2) / 4.0;
        let eta  = (1.0 + a2 / b2 + a2 / c2) / 4.0;
        KPath {
            bravais_label: "oF3".into(),
            points: pts(&[
                ("G",0.,0.,0.),("A",0.5,0.5+zeta,zeta),("A1",0.5,0.5-zeta,1.0-zeta),
                ("L",0.5,0.5,0.5),("T",1.,0.5,0.5),("X",0.,eta,eta),
                ("X1",1.,1.-eta,1.-eta),("Y",0.5,0.,0.5),("Z",0.5,0.5,0.),
            ]),
            segments: segs(&[
                ("G","Y"),("Y","T"),("T","Z"),("Z","G"),("G","X"),
                ("X","A1"),("A1","Y"),("X","A"),("A","Z"),("L","G"),
            ]),
        }
    }
}

fn kpath_oi(conv: &Poscar) -> KPath {
    let a = norm(conv.lattice[0]);
    let b = norm(conv.lattice[1]);
    let c = norm(conv.lattice[2]);
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let zeta  = (1.0 + a2 / c2) / 4.0;
    let eta   = (1.0 + b2 / c2) / 4.0;
    let delta = (b2 - a2) / (4.0 * c2);
    let mu    = (a2 + b2) / (4.0 * c2);

    KPath {
        bravais_label: "oI".into(),
        points: pts(&[
            ("G",0.,0.,0.),("L",-mu,mu,0.5-delta),("L1",mu,-mu,0.5+delta),
            ("L2",0.5-delta,0.5+delta,-mu),("R",0.,0.5,0.),("S",0.5,0.,0.),
            ("T",0.,0.,0.5),("W",0.25,0.25,0.25),("X",-zeta,zeta,zeta),
            ("X1",zeta,1.0-zeta,-zeta),("Y",eta,-eta,eta),("Y1",1.0-eta,eta,-eta),
            ("Z",0.5,0.5,-0.5),
        ]),
        segments: segs(&[
            ("G","X"),("X","L"),("L","T"),("T","W"),("W","R"),("R","X1"),("X1","Z"),
            ("Z","G"),("G","Y"),("Y","S"),("S","W"),("L1","Y"),("Y1","Z"),
        ]),
    }
}

fn kpath_oc(conv: &Poscar) -> KPath {
    let a = norm(conv.lattice[0]);
    let b = norm(conv.lattice[1]);
    let zeta = (1.0 + a * a / (b * b)) / 4.0;

    KPath {
        bravais_label: "oC".into(),
        points: pts(&[
            ("G",0.,0.,0.),("A",zeta,zeta,0.5),("A1",-zeta,1.0-zeta,0.5),
            ("R",0.,0.5,0.5),("S",0.,0.5,0.),("T",-0.5,0.5,0.5),
            ("X",zeta,zeta,0.),("X1",-zeta,1.0-zeta,0.),("Y",-0.5,0.5,0.),("Z",0.,0.,0.5),
        ]),
        segments: segs(&[
            ("G","X"),("X","S"),("S","R"),("R","A"),("A","Z"),("Z","G"),
            ("G","Y"),("Y","X1"),("X1","A1"),("A1","T"),("T","Y"),("Z","T"),
        ]),
    }
}

fn kpath_hp() -> KPath {
    KPath {
        bravais_label: "hP".into(),
        points: pts(&[
            ("G",0.,0.,0.),("A",0.,0.,0.5),("H",1./3.,1./3.,0.5),
            ("K",1./3.,1./3.,0.),("L",0.5,0.,0.5),("M",0.5,0.,0.),
        ]),
        segments: segs(&[
            ("G","M"),("M","K"),("K","G"),("G","A"),("A","L"),("L","H"),("H","A"),("L","M"),("K","H"),
        ]),
    }
}

fn kpath_hr(conv: &Poscar) -> KPath {
    let ah = norm(conv.lattice[0]);
    let ch = norm(conv.lattice[2]);
    let a2 = ah * ah;
    let c2 = ch * ch;
    let cos_alpha = (2.0 * c2 - 3.0 * a2) / (2.0 * c2 + 6.0 * a2);

    if cos_alpha > 0.0 {
        // hR1
        let eta = (1.0 + 4.0 * cos_alpha) / (2.0 + 4.0 * cos_alpha);
        let nu  = 0.75 - eta / 2.0;
        KPath {
            bravais_label: "hR1".into(),
            points: pts(&[
                ("G",0.,0.,0.),("B",eta,0.5,1.0-eta),("B1",0.5,1.0-eta,eta-1.0),
                ("F",0.5,0.5,0.),("L",0.5,0.,0.),("L1",0.,0.,-0.5),
                ("P",eta,nu,nu),("P1",1.0-nu,1.0-nu,1.0-eta),("P2",nu,nu,eta-1.0),
                ("Q",1.0-nu,nu,0.),("X",nu,0.,-nu),("Z",0.5,0.5,0.5),
            ]),
            segments: segs(&[
                ("G","L"),("L","B1"),("B","Z"),("Z","G"),("G","X"),
                ("Q","F"),("F","P1"),("P1","Z"),("L","P"),
            ]),
        }
    } else {
        // hR2
        let eta = (1.0 + cos_alpha) / (2.0 * (1.0 - cos_alpha));
        let nu  = 0.75 - eta / 2.0;
        KPath {
            bravais_label: "hR2".into(),
            points: pts(&[
                ("G",0.,0.,0.),("F",0.5,-0.5,0.),("L",0.5,0.,0.),
                ("P",1.0-nu,-nu,1.0-nu),("P1",nu,nu-1.0,nu-1.0),
                ("Q",eta,eta,eta),("Q1",1.0-eta,-eta,-eta),("Z",0.5,-0.5,0.5),
            ]),
            segments: segs(&[
                ("G","P"),("P","Z"),("Z","Q"),("Q","G"),("G","F"),
                ("F","P1"),("P1","Q1"),("Q1","L"),("L","Z"),
            ]),
        }
    }
}

fn kpath_mp(conv: &Poscar) -> KPath {
    use std::f64::consts::PI;
    let b = norm(conv.lattice[0]);
    let c = norm(conv.lattice[2]);
    let beta_raw = angle(conv.lattice[0], conv.lattice[2]);
    let beta = if beta_raw > PI / 2.0 { PI - beta_raw } else { beta_raw };
    let sin2 = beta.sin() * beta.sin();
    let eta = (1.0 - b * beta.cos() / c) / (2.0 * sin2);
    let nu  = 0.5 - eta * c * beta.cos() / b;

    KPath {
        bravais_label: "mP".into(),
        points: pts(&[
            ("G",0.,0.,0.),("A",0.5,0.5,0.),("C",0.,0.5,0.5),("D",0.5,0.,0.5),
            ("D1",0.5,0.,-0.5),("E",0.5,0.5,0.5),("H",0.,eta,1.0-nu),
            ("H1",0.,1.0-eta,nu),("H2",0.,eta,-nu),("M",0.5,eta,1.0-nu),
            ("M1",0.5,1.0-eta,nu),("M2",0.5,eta,-nu),("X",0.,0.5,0.),
            ("Y",0.,0.,0.5),("Y1",0.,0.,-0.5),("Z",0.5,0.,0.),
        ]),
        segments: segs(&[
            ("G","Y"),("Y","H"),("H","C"),("C","E"),("E","M1"),("M1","A"),
            ("A","X"),("X","H1"),("M","D"),("D","Z"),("Y","D"),
        ]),
    }
}

fn kpath_mc(conv: &Poscar) -> KPath {
    use std::f64::consts::PI;
    let a = norm(conv.lattice[1]); // unique axis
    let b = norm(conv.lattice[0]);
    let c = norm(conv.lattice[2]);
    let beta_raw = angle(conv.lattice[0], conv.lattice[2]);
    let beta = if beta_raw > PI / 2.0 { PI - beta_raw } else { beta_raw };
    let sin2 = beta.sin() * beta.sin();

    // Compute kgamma from primitive reciprocal lattice.
    // C-centering: ap = (a_conv + b_conv)/2, bp = (a_conv - b_conv)/2, cp = c_conv
    let ap: [f64; 3] = [
        (conv.lattice[0][0] + conv.lattice[1][0]) * 0.5,
        (conv.lattice[0][1] + conv.lattice[1][1]) * 0.5,
        (conv.lattice[0][2] + conv.lattice[1][2]) * 0.5,
    ];
    let bp: [f64; 3] = [
        (conv.lattice[0][0] - conv.lattice[1][0]) * 0.5,
        (conv.lattice[0][1] - conv.lattice[1][1]) * 0.5,
        (conv.lattice[0][2] - conv.lattice[1][2]) * 0.5,
    ];
    let cp = conv.lattice[2];

    let bpxcp = cross(bp, cp);
    let cpxap = cross(cp, ap);
    let vp = dot(ap, bpxcp);
    let b1p = [bpxcp[0] / vp, bpxcp[1] / vp, bpxcp[2] / vp];
    let b2p = [cpxap[0] / vp, cpxap[1] / vp, cpxap[2] / vp];
    let kgamma_deg = angle(b1p, b2p) * 180.0 / PI;

    if kgamma_deg > 90.0 {
        // mC1
        let zeta = (2.0 - b * beta.cos() / c) / (4.0 * sin2);
        let eta  = 0.5 + 2.0 * zeta * c * beta.cos() / b;
        let psi  = 0.75 - a * a / (4.0 * b * b * sin2);
        let phi  = psi + (0.75 - psi) * b * beta.cos() / c;
        KPath {
            bravais_label: "mC1".into(),
            points: pts(&[
                ("G",0.,0.,0.),("N",0.5,0.,0.),("N1",0.,-0.5,0.),
                ("F",1.-zeta,1.-zeta,1.-eta),("F1",zeta,zeta,eta),("F2",-zeta,-zeta,1.-eta),
                ("I",phi,1.-phi,0.5),("I1",1.-phi,phi-1.,0.5),("L",0.5,0.5,0.5),
                ("M",0.5,0.,0.5),("X",1.-psi,psi-1.,0.),("X1",psi,1.-psi,0.),
                ("X2",psi-1.,-psi,0.),("Y",0.5,0.5,0.),("Y1",-0.5,-0.5,0.),("Z",0.,0.,0.5),
            ]),
            segments: segs(&[
                ("G","Y"),("Y","F"),("F","L"),("L","I"),("I1","Z"),
                ("Z","F1"),("Y","X1"),("X","G"),("G","N"),("M","G"),
            ]),
        }
    } else if kgamma_deg < 90.0 {
        let test = b * beta.cos() / c + b * b * sin2 / (a * a);
        if (test - 1.0).abs() < 1e-10 {
            // mC4 (degenerate)
            let mu    = (1.0 + b * b / (a * a)) / 4.0;
            let delta = b * c * beta.cos() / (2.0 * a * a);
            let zeta  = mu - 0.25 + (1.0 - b * beta.cos() / c) / (4.0 * sin2);
            let eta   = 0.5 + 2.0 * zeta * c * beta.cos() / b;
            let phi   = 1.0 + zeta - 2.0 * mu;
            let psi   = eta - 2.0 * delta;
            KPath {
                bravais_label: "mC4".into(),
                points: mC3_mC4_points(mu, delta, zeta, eta, phi, psi),
                segments: segs(&[
                    ("G","Y"),("Y","F"),("F","H"),("H","Z"),("Z","I"),
                    ("H1","Y1"),("Y1","X"),("X","G"),("G","N"),("M","G"),
                ]),
            }
        } else if test < 1.0 {
            // mC3
            let mu    = (1.0 + b * b / (a * a)) / 4.0;
            let delta = b * c * beta.cos() / (2.0 * a * a);
            let zeta  = mu - 0.25 + (1.0 - b * beta.cos() / c) / (4.0 * sin2);
            let eta   = 0.5 + 2.0 * zeta * c * beta.cos() / b;
            let phi   = 1.0 + zeta - 2.0 * mu;
            let psi   = eta - 2.0 * delta;
            KPath {
                bravais_label: "mC3".into(),
                points: mC3_mC4_points(mu, delta, zeta, eta, phi, psi),
                segments: segs(&[
                    ("G","Y"),("Y","F"),("F","H"),("H","Z"),("Z","I"),("I","F1"),
                    ("H1","Y1"),("Y1","X"),("X","G"),("G","N"),("M","G"),
                ]),
            }
        } else {
            // mC5
            let zeta  = (b * b / (a * a) + (1.0 - b * beta.cos() / c) / sin2) / 4.0;
            let eta   = 0.5 + 2.0 * zeta * c * beta.cos() / b;
            let mu    = eta / 2.0 + b * b / (4.0 * a * a) - b * c * beta.cos() / (2.0 * a * a);
            let nu    = 2.0 * mu - zeta;
            let rho   = 1.0 - zeta * a * a / (b * b);
            let omega = (4.0 * nu - 1.0 - b * b * sin2 / (a * a)) * c / (2.0 * b * beta.cos());
            let delta = zeta * c * beta.cos() / b + omega / 2.0 - 0.25;
            KPath {
                bravais_label: "mC5".into(),
                points: pts(&[
                    ("G",0.,0.,0.),("F",nu,nu,omega),("F1",1.-nu,1.-nu,1.-omega),
                    ("F2",nu,nu-1.,omega),("H",zeta,zeta,eta),("H1",1.-zeta,-zeta,1.-eta),
                    ("H2",-zeta,-zeta,1.-eta),("I",rho,1.-rho,0.5),("I1",1.-rho,rho-1.,0.5),
                    ("L",0.5,0.5,0.5),("M",0.5,0.,0.5),("N",0.5,0.,0.),("N1",0.,-0.5,0.),
                    ("X",0.5,-0.5,0.),("Y",mu,mu,delta),("Y1",1.-mu,-mu,-delta),
                    ("Y2",-mu,-mu,-delta),("Y3",mu,mu-1.,delta),("Z",0.,0.,0.5),
                ]),
                segments: segs(&[
                    ("G","Y"),("Y","F"),("F","L"),("L","I"),("I1","Z"),("Z","H"),
                    ("H","F1"),("H1","Y1"),("Y1","X"),("X","G"),("G","N"),("M","G"),
                ]),
            }
        }
    } else {
        // mC2 (kgamma == 90°)
        let zeta = (2.0 - b * beta.cos() / c) / (4.0 * sin2);
        let eta  = 0.5 + 2.0 * zeta * c * beta.cos() / b;
        let psi  = 0.75 - a * a / (4.0 * b * b * sin2);
        let phi  = psi + (0.75 - psi) * b * beta.cos() / c;
        KPath {
            bravais_label: "mC2".into(),
            points: pts(&[
                ("G",0.,0.,0.),("N",0.5,0.,0.),("N1",0.,-0.5,0.),
                ("F",1.-zeta,1.-zeta,1.-eta),("F1",zeta,zeta,eta),
                ("F2",-zeta,-zeta,1.-eta),("F3",1.-zeta,-zeta,1.-eta),
                ("I",phi,1.-phi,0.5),("I1",1.-phi,phi-1.,0.5),("L",0.5,0.5,0.5),
                ("M",0.5,0.,0.5),("X",1.-psi,psi-1.,0.),("X1",psi,1.-psi,0.),
                ("X2",psi-1.,-psi,0.),("Y",0.5,0.5,0.),("Y1",-0.5,-0.5,0.),("Z",0.,0.,0.5),
            ]),
            segments: segs(&[
                ("G","Y"),("Y","F"),("F","L"),("L","I"),("I1","Z"),("Z","F1"),("N","G"),("G","M"),
            ]),
        }
    }
}

/// Shared point set for mC3 and mC4.
#[allow(non_snake_case)]
fn mC3_mC4_points(mu: f64, delta: f64, zeta: f64, eta: f64, phi: f64, psi: f64) -> Vec<KPoint> {
    pts(&[
        ("G",0.,0.,0.),("F",1.-phi,1.-phi,1.-psi),("F1",phi,phi-1.,psi),
        ("F2",1.-phi,-phi,1.-psi),("H",zeta,zeta,eta),("H1",1.-zeta,-zeta,1.-eta),
        ("H2",-zeta,-zeta,1.-eta),("I",0.5,-0.5,0.5),("M",0.5,0.,0.5),
        ("N",0.5,0.,0.),("N1",0.,-0.5,0.),("X",0.5,-0.5,0.),
        ("Y",mu,mu,delta),("Y1",1.-mu,-mu,-delta),("Y2",-mu,-mu,-delta),
        ("Y3",mu,mu-1.,delta),("Z",0.,0.,0.5),
    ])
}

fn kpath_ap(prim: &Poscar) -> KPath {
    use std::f64::consts::PI;
    let (b1, b2, b3) = reciprocal(prim);
    // Angles between reciprocal vectors
    let ka = angle(b2, b3) * 180.0 / PI;
    let kb = angle(b1, b3) * 180.0 / PI;
    let kg = angle(b1, b2) * 180.0 / PI;
    let is_tri1a = ka > 90.0 && kb > 90.0 && kg >= 90.0;

    let segs_ap = segs(&[("X","G"),("G","Y"),("L","G"),("G","Z"),("N","G"),("G","M"),("R","G")]);
    if is_tri1a {
        KPath {
            bravais_label: "aP1".into(),
            points: pts(&[
                ("G",0.,0.,0.),("L",0.5,0.5,0.),("M",0.,0.5,0.5),("N",0.5,0.,0.5),
                ("R",0.5,0.5,0.5),("X",0.5,0.,0.),("Y",0.,0.5,0.),("Z",0.,0.,0.5),
            ]),
            segments: segs_ap,
        }
    } else {
        KPath {
            bravais_label: "aP2".into(),
            points: pts(&[
                ("G",0.,0.,0.),("L",0.5,-0.5,0.),("M",0.,0.,0.5),("N",-0.5,-0.5,0.5),
                ("R",0.,-0.5,0.5),("X",0.,-0.5,0.),("Y",0.5,0.,0.),("Z",-0.5,0.,0.5),
            ]),
            segments: segs_ap,
        }
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Determine the Bravais lattice type and return the corresponding k-path.
///
/// * `std_conv`  — Standardised **conventional** cell (use `primitive=0` from spglib).
///                 Provides lattice parameters for parametric k-points.
/// * `std_prim`  — Primitive cell used as orientation reference for the triclinic (aP)
///                 subcase.  Pass the as-read POSCAR (not spglib's re-standardised
///                 primitive) so that the reciprocal-angle classification matches the
///                 user's/pymatgen's choice.
/// * `dataset`   — spglib dataset obtained from `std_conv`.
///
/// Returns `None` if the space group is unrecognised or unsupported.
pub fn get_bravais_kpath(
    std_conv: &Poscar,
    std_prim: &Poscar,
    dataset: &SpglibDataset,
) -> Option<KPath> {
    let intl = crate::symmetry::spglib_intl_symbol(dataset);
    let bravais = detect_bravais(dataset.spacegroup_number, &intl)?;

    Some(match bravais {
        Bravais::CP => kpath_cp(),
        Bravais::CF => kpath_cf(),
        Bravais::CI => kpath_ci(),
        Bravais::TP => kpath_tp(),
        Bravais::TI => kpath_ti(std_conv),
        Bravais::OP => kpath_op(),
        Bravais::OF => kpath_of(std_conv),
        Bravais::OI => kpath_oi(std_conv),
        Bravais::OC => kpath_oc(std_conv),
        Bravais::HP => kpath_hp(),
        Bravais::HR => kpath_hr(std_conv),
        Bravais::MP => kpath_mp(std_conv),
        Bravais::MC => kpath_mc(std_conv),
        Bravais::AP => kpath_ap(std_prim),
    })
}
