use clap::Parser;
use vasp_utils::poscar::Poscar;

#[derive(Parser)]
#[command(about = "Convert POSCAR to ecalj/Questaal ctrls format (Å → Bohr)")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "ctrls.system")]
    output: String,
}

fn main() {
    let cli = Cli::parse();

    let mut poscar = match Poscar::read(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if poscar.is_direct {
        poscar.to_cartesian();
        println!("Converted Direct -> Cartesian");
    }

    if let Err(e) = poscar.write_ctrls(&cli.output) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    println!("Output written to: {}", cli.output);
}
