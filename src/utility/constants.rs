/*
Constants
*/

// The number of nodes in the inner layer.
pub const HIDDEN:   [usize; 3]   = [100, 50, 20];
// Value of the momentum term when adjusting weights.
pub const MOMENTUM: [f32; 4]     = [0.9, 0.5, 0.25, 0.0];
// The fraction of the training data used to train the network.
pub const TRAIN:    [f32; 3]     = [1.0, 0.5, 0.25];

// The number of epochs to train for
pub const EPOCH:    usize   = 5;
// Number of output nodes in the final layer.
pub const OUTPUT:   usize   = 10;
// Number of input nodes in the initial layer.
pub const INPUT:    usize   = 784;
// The amount of storage available for when using batching.
//pub const STORAGE:  usize   = 1;

// Used for normalizing the data.
pub const DIVISOR:  f32     = 255.;
// For randomizing initial weight values.
pub const LOW:      f32     = -0.05;
pub const HIGH:     f32     =  0.05;
// The learning rate.
pub const LEARN:    f32     = 0.2;

// A list of the classes we are testing for.
pub const CLASS: [&str; 10] = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ];
