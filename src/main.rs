/*
Thomas Abel
2022-08-09
Machine Learning
*/
mod utility;
mod network;

use std::error::Error;
use ndarray::prelude::*;
use crate::utility::constants::*;
use crate::network::train_test::*;

type Input = (Array2<f32>, Array1<String>);

// Reads the data file. Change the index to read from a different file.
fn read() -> Result<Input, Box<dyn Error>> {
    let index: usize = 0;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    print!("Reading data, please be patient... ");
    let result
        = utility::read::read(path[index]);
    println!("Done.");
    result
}

/// Main functions.
fn main() {
    // Reading the files.
    let temp = read();
    let input;
    match temp {
        Ok(mut o) => {
            // Normalize the data.
            o.0 /= DIVISOR;
            input = o;
        }
        Err(e) => {
            println!("{}", e);
            return;
        }
    }

    // The network itself.
    let mut network = create_network();
    let confusion = train_epoch(&mut network, &input.0, &input.1);
    print_confusion(&confusion);

    // End message.
    println!("Ending Session.");
}
