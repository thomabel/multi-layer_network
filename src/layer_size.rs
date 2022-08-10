
#[derive(Clone, Copy)]
pub struct LayerSize {
    pub input: usize,
    pub output: usize,
    pub storage: usize,
}
impl LayerSize {
    pub fn new(output: usize, input: usize, storage: usize) -> LayerSize {
        LayerSize { output, input, storage }
    }
}
