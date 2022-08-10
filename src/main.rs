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

type Matrix = Array2<f32>;
type Target = Array1<String>;
type Input = (Matrix, Target);
type ReadResult = Result<Input, Box<dyn Error>>;

// Reads the data file. Change the index to read from a different file.
fn read(path: &str) -> ReadResult {
    print!("Reading data, please be patient... ");
    let result
        = utility::read::read(path);
    println!("Done.");
    result
}
fn evaluate(result: ReadResult) -> Input {
    let input;
    match result {
        Ok(mut o) => {
            // Normalize the data.
            o.0 /= DIVISOR;
            input = o;
        }
        Err(e) => {
            panic!("{}", e);
        }
    }
    input
}
fn read_eval(path: &str) -> Input {
    evaluate(read(path))
}

/// Main functions.
fn main() {
    // Reading the files.
    let index_train: usize = 0;
    let index_test: usize = 0;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    
    let input_train = read_eval(path[index_train]);
    let input_test = read_eval(path[index_test]);

    // The network itself.
    let mut network = create_network();
    let print = true;
    let confusion_train 
        = epoch(&mut network, &input_train.0, &input_train.1, EvaluateState::Train, !print);
    print_confusion(&confusion_train);
    
    let confusion_test 
        = epoch(&mut network, &input_test.0, &input_test.1, EvaluateState::Test, print);
    print_confusion(&confusion_test);

    // End message.
    println!("Ending Session.");
}
