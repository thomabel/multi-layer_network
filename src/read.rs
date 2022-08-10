use ndarray::prelude::*;
use csv::Reader;
use std::error::Error;

pub fn read(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut rows = 0;
    let columns = reader.records().next().unwrap().unwrap().len();

    for r in reader.records() {
        let entry = r?;
        for e in entry.into_iter() {
            let parse = e.parse::<f32>();
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
    Ok(Array2::<f32>::from_shape_vec((columns, rows), output)?)
}