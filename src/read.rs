use ndarray::prelude::*;
use csv::Reader;
use std::error::Error;

pub fn read(path: &str) -> Result<(Array2<f32>, Array1<String>), Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut target = Vec::new();
    let mut rows = 0;
    let columns = reader.headers()?.len() - 1;

    for r in reader.records() {
        let entry = r?;
        target.push(entry[0].to_string());
        for e in 1..entry.len() {
            let parse = entry[e].parse::<f32>();
            match parse {
                Ok(o) => {
                    output.push(o);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        rows += 1;
    }
    
    let output = 
        Array2::<f32>::from_shape_vec((rows, columns), output)?;
    let target = 
        Array1::<String>::from_shape_vec(rows, target)?;
    Ok( (output, target) )
}
