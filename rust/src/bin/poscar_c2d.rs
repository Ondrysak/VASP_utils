use clap::Parser;
use vasp_utils::poscar::Poscar;

#[derive(Parser)]
#[command(about = "Convert POSCAR coordinates from Cartesian to Direct (fractional)")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "POSCAR_direct")]
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

    if !poscar.is_direct {
        poscar.to_direct();
        println!("Converted Cartesian -> Direct");
    } else {
        println!("POSCAR is already in Direct (fractional) coordinates");
    }

    if let Err(e) = poscar.write(&cli.output) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    println!("Output written to: {}", cli.output);
}
