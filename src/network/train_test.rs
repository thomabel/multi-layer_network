use ndarray::prelude::*;
use rand::thread_rng;
use rand::{self, seq::SliceRandom};
use crate::Input;
use crate::network::layer_size::LayerSize;
use crate::network::multi_layer::MultiLayer;
use crate::utility::epoch::{Info, Results};
use crate::utility::print_data;

/// Public
pub enum EvaluateState { Train, Test }
// Trains and tests a network using one set of parameters.
pub fn epoch_set(
    network: &mut MultiLayer, 
    input_train: &Input, 
    input_test: &Input, 
    classes: &[&str],
    info: &mut Info) -> (Results, Results)
{
    // Results needed for final evaluation of model.
    let mut train = Results::new(info.epoch);
    let mut test = Results::new(info.epoch);

    // Train and test the network over a number of epochs.
    for e in 0..info.epoch {
        println!("EPOCH: {}", e);

        // TRAIN
        info.state = EvaluateState::Train;
        train.confusion = Some(epoch(network, 
                &input_train.0, &input_train.1, 
                classes, info));
        train.accuracy[e] = print_confusion(train.confusion.as_ref().unwrap(), info.print);

        // TEST
        info.state = EvaluateState::Test;
        test.confusion = Some(epoch(network, 
                &input_test.0, &input_test.1, 
                classes, info));
        test.accuracy[e] = print_confusion(test.confusion.as_ref().unwrap(), info.print);
    }

    (train, test)
}

// Trains the network over a single epoch and returns the resulting confusion matrix.
pub fn epoch(network: &mut MultiLayer, input: &Array2<f32>, target: &Array1<String>, classes: &[&str], info: &Info) -> Array2<u32> {
    // Create confusion matrix and random index array.
    let output = classes.len();
    let mut confusion = Array2::<u32>::zeros((output, output));
    //let input_random = create_random_index(target.len());
    let input_random = create_random_index_fraction(target, classes, info.fraction);

    // Feed each input into the network and update the weights.
    // Completing this is considered 1 epoch.
    for i in 0..target.len() {
        // Set-up
        if info.print { print!("{:>7}: ", i); }
        let index = input_random[i];
        let row = input.row(index);
        let output = network.output(&row);
        
        // Printing things
        let target_str = &target[index];
        let predict_str = &classify(&output, classes);
        if info.print { println!("[ {} ] => {}", target_str, predict_str); }
        // Update the confusion matrix
        confusion[[class_to_index(target_str, classes).unwrap(), class_to_index(predict_str, classes).unwrap()]] += 1;
        
        match info.state {
            EvaluateState::Train => {
                // Find the error values and update the weights.
                let target = target_array(target_str, classes);
                network.error(&target.view());
                network.weight(&row, info.learn_rate, info.momentum);
            }
            EvaluateState::Test => {
                // Continue the testing.
            }
        }
    }

    confusion
}

// Create the network using some predefined layer sizes.
pub fn create_network(input: usize, hidden: usize, output: usize, low: f32, high: f32) -> MultiLayer {
    let storage = 1;
    let size = [
        LayerSize::new(hidden, input, storage),
        LayerSize::new(output, hidden, storage),
    ];
    MultiLayer::new(&size, low, high)
}


/// Private

// Print the confusion matrix and accuracy.
fn print_confusion(confusion: &Array2<u32>, print: bool) -> f32 {
    let correct = confusion.diag().sum();
    let total = confusion.sum();
    if print {
        println!();
        print_data::_print_matrix(&confusion.view(), "CONFUSION");
        print_data::_print_total_error(correct, total);
    }
    correct as f32 / total as f32
}

// Creates a vector of indicies using only a fraction of the available data.
fn create_random_index_fraction(
    input: &Array1<String>, 
    classes: &[&str], 
    fraction: f32
) -> Array1<usize> 
{
    let total = input.len();
    
    // Create vector of vectors to hold indicies of each class.
    let class_len = classes.len();
    let mut class_vec = Vec::with_capacity(class_len);
    for _c in 0..class_len {
        class_vec.push(Vec::with_capacity(total/class_len));
    }

    // For each input vector, find class and push index into that class's vector.
    for i in 0..total {
        class_vec[class_to_index(&input[i], classes).unwrap()].push(i);
    }
    
    // Shuffle each class and then
    // Grab the correct fraction of entries and add to final vector.
    let mut rng = thread_rng();
    let mut collect = Vec::new();
    for class in class_vec.iter_mut() {
        class.shuffle(&mut rng);
        let fraction_index = (class.len() as f32 * fraction) as usize;
        let slice = class[0..fraction_index].to_vec();
        collect.push(slice);
    }
    
    // Flatten the vector and shuffle its elements.
    let mut vec = collect.into_iter().flatten().collect::<Vec<usize>>();
    vec.shuffle(&mut rng);
    Array1::<usize>::from_vec(vec)
}

// Creates a vector of indicies for randomizing the data.
fn _create_random_index(size: usize) -> Array1<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    Array1::<usize>::from_vec(vec)
}

// Finds the class that the model predicted.
fn classify(output: &ArrayView1<f32>, classes: &[&str]) -> String {
    // Find the largest output value.
    let mut index = 0;
    let mut value = 0.;

    for i in 0..output.len() {
        //print!("{}, ", output[i]);
        if output[i] >= value {
            index = i;
            value = output[i];
        }
    }
    //println!();
    classes[index].to_string()
}

// Converts a class to an index value.
fn class_to_index(target: &str, classes: &[&str]) -> Result<usize, String> {
    for (i, class) in classes.iter().enumerate() {
        if target == *class {
            return Ok(i);
        }
    }
    Err("No Match.".to_string())
}

// Returns the target value needed for error calculation.
fn target_array(target: &str, classes: &[&str]) -> Array1<f32> {
    let mut target_arr = Array1::<f32>::from_elem(classes.len(), 0.1);
    let i = class_to_index(target, classes);
    match i {
        Ok(o) => {
            target_arr[o] = 0.9;
        }
        Err(e) => {
            println!("{}", e);
        }
    }
    target_arr
}
