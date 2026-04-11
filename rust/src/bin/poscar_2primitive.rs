use clap::Parser;
use vasp_utils::symmetry::standardize_cell;
use vasp_utils::poscar::Poscar;

#[derive(Parser)]
#[command(about = "Transform POSCAR to primitive cell using spglib")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "POSCAR_primitive")]
    output: String,
    /// Symmetry tolerance (Å)
    #[arg(long, default_value_t = 1e-5)]
    symprec: f64,
    /// Idealize the primitive cell
    #[arg(long, default_value_t = false)]
    idealize: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.symprec > 1e-3 {
        eprintln!("Warning: symprec is too high! Consider using the default value 1e-5.");
    }

    let poscar = match Poscar::read(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    match standardize_cell(&poscar, cli.symprec, true, cli.idealize) {
        Some(prim) => {
            if let Err(e) = prim.write(&cli.output) {
                eprintln!("{e}");
                std::process::exit(1);
            }
            println!("Primitive cell written to: {}", cli.output);
        }
        None => {
            eprintln!("Error: spglib failed to create primitive cell.");
            std::process::exit(1);
        }
    }
}
