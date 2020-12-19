use std::vec::Vec;
extern crate ndarray;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;


fn main() {
    
    let a: Array2<f32> = Array::random((3,3), StandardNormal);
    println!("a {}", a);
    let b = Array::from_shape_vec((1, 1), vec![a]).unwrap();
}
