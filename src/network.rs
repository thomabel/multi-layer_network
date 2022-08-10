use crate::layer::Layer;
use crate::layer_size::LayerSize;
use ndarray::prelude::*;

type Vector<'a> = ArrayView1<'a, f32>;

pub struct Network {
    layer: Vec<Layer>,
}
impl Network {
    pub fn new(size: &[LayerSize]) -> Network {
        let mut layer = Vec::with_capacity(size.len());
        for l in size {
            layer.push(Layer::new(*l));
        }
        Network { layer }
    }

    pub fn output(&mut self, input: &Vector) {
        let mut iter = self.layer.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        layer_prev.output(input);
        
        for layer in iter {
            layer.output(&layer_prev.output_vector().view());
            layer_prev = layer;
        }
    }
}