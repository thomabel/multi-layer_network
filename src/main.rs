/*
Thomas Abel
2022-08-09
Machine Learning
*/

mod read;
mod constants;
mod print_data;

use std::error::Error;
use ndarray::prelude::*;
use print_data::*;
use crate::constants::*;
use rand::{self, seq::SliceRandom};

fn main() {
    let temp = read();
    let input;
    match temp {
        Ok(mut o) => {
            input = o/DIVISOR;
        }
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    _print_matrix(&input.view(), "INPUT")

}

fn random_index(size: usize) -> Vec<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    vec
}

fn read() -> Result<Array2<f32>, Box<dyn Error>> {
    let path_index = 0;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    println!("Reading data, please be patient...");
    read::read(path[path_index])
}
