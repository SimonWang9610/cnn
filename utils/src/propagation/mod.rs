use ndarray::Array2;

pub trait Propagation {
    fn forward(&self, inputs: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>>;
    
    fn backward(
        &self, 
        inputs: Vec<Vec<Array2<f32>>>,
        deltas: Vec<Vec<Array2<f32>>>
    ) -> Vec<Vec<Array2<f32>>>;
}