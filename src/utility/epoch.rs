use crate::network::train_test::EvaluateState;
pub struct Info {
    pub epoch: usize,
    pub state: EvaluateState, 
    pub learn_rate: f32,
    pub momentum: f32,
    pub fraction: f32,
    pub print: bool,
}

use ndarray::prelude::{Array1, Array2};
pub struct Results {
    pub confusion: Option<Array2<u32>>,
    pub accuracy: Array1<f32>,
}
impl Results {
    pub fn new(epochs: usize) -> Results {
        let confusion = None;
        let accuracy = Array1::<f32>::zeros(epochs);
        Results {confusion, accuracy}
    }
}