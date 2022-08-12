use ndarray::prelude::*;
use csv::Writer;
use crate::utility::epoch;
use std::{error::Error, fmt};

#[derive(Debug)]
struct WriteError(String);
impl Error for WriteError {}
impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

pub fn write_csv(path: &str, input: &epoch::Results) -> Result<(), Box<dyn Error>> {
    
    let confusion = match &input.confusion {
        Some(s) => {s}
        None => { return Err(Box::new(WriteError("Cannot find confusion matrix".to_string()))); }
    };
    
    let mut writer = Writer::from_path(path)?;

    for row in confusion.rows() {
        //writer.write_record(row.into_iter());
    }

    Ok(())
}
