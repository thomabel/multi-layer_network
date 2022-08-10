/*
Thomas Abel
2022-08-09
Machine Learning
*/
mod utility;
mod network;

use std::error::Error;
use ndarray::prelude::*;
use crate::network::train_test::*;
use crate::utility::constants::*;
use crate::utility::print_data::*;

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

// Print the confusion matrix and accuracy.
fn print_confusion(confusion: &Array2<u32>) -> f32 {
    println!();
    _print_matrix(&confusion.view(), "CONFUSION");
    let correct = confusion.diag().sum();
    let total = confusion.sum();
    _print_total_error(correct, total);

    correct as f32 / total as f32
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
    
    // Gathering relevent settings.
    let print = true;
    let hidden = HIDDEN[2];
    let momentum = MOMENTUM[0];
    let training = TRAIN[0];

    // Create the network and data storage.
    let mut network = create_network(hidden);
    let mut confusion_train;
    let mut confusion_test;
    let mut accuracy_train = Vec::with_capacity(EPOCH);
    let mut accuracy_test = Vec::with_capacity(EPOCH);

    // Train and test the network over a number of epochs.
    for e in 0..EPOCH {
        // TRAIN
        confusion_train
            = epoch(&mut network, 
                &input_train.0, 
                &input_train.1, 
                EvaluateState::Train, 
                momentum,
                !print,
            );
        accuracy_train[e] = print_confusion(&confusion_train);

        // TEST
        confusion_test
            = epoch(&mut network, 
                &input_test.0, 
                &input_test.1, 
                EvaluateState::Test,
                momentum, 
                print
            );
        accuracy_test[e] = print_confusion(&confusion_test);
    }

    // End message.
    println!("Ending Session.");
}
