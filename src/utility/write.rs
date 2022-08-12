use csv::WriterBuilder;
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

pub fn write_csv(name: &str, input: &epoch::Results) -> Result<(), Box<dyn Error>> {
    // Test to make sure the confusion matrix exists.
    let confusion = match &input.confusion {
        Some(s) => {s}
        None => { return Err(Box::new(WriteError("Cannot find confusion matrix".to_string()))); }
    };
    
    // Configure writer
    let path = format!("./confusion/{}.csv", name);
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_path(path)?;

    // Write to the file
    for row in confusion.rows() {
        writer.write_record(row.iter().map(|x| x.to_string()))?;
    }
    writer.flush()?;

    Ok(())
}
