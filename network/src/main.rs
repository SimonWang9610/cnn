extern crate ndarray;
extern crate utils;
extern crate image;

use utils::network::{nn, forward};
use utils::pooling::Pool;
use utils::convolution::{Conv3D, Conv2D};
use utils::full_connected::FullLayer;
use utils::utils::utils::flip_matrix;
use ndarray::{Array2, Array};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use image::io::Reader;
fn main() {
    /* let pool = Pool::new(3, 2, 0);
    let input:Vec<Array2<f32>> = (0..2).map(|_| Array::random((7, 7), StandardNormal)).collect();
    println!("input {:#?}", input);

    let (outputs, max_positions) = pool.max_pool(input);
    
    for (output, pos) in outputs.iter().zip(max_positions.iter()) {
        println!("pos {:?}", pos);
    }

    let a: Vec<f32> = (0..9).map(|_| 1.).collect();
    let b: Vec<f32> = (0..9).map(|_| 2.).collect();
    let deltas = vec![Array2::from_shape_vec((3,3), a).unwrap(), Array2::from_shape_vec((3,3), b).unwrap()];

    let out_delta = pool.upsample(deltas, max_positions);
    println!(" out delta {:?}", out_delta); */

    let image_bytes = Reader::open("../second.png").unwrap().decode().unwrap();
    let image_pixels = image_bytes.into_rgb8().into_vec().into_iter();
    
    let mut a: Vec<f32> = vec![];
    let mut b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];

    for (i, v) in image_pixels.enumerate() {
        
        if i / 784 == 0 {
            a.push(v as f32 / 255.)
        } else if i / 784 == 1 {
            b.push(v as f32 / 255.);
        } else {
            c.push(v as f32 / 255.);
        }
    }
    let image = vec![Array2::from_shape_vec((28, 28), a).unwrap(), 
        Array2::from_shape_vec((28, 28), b).unwrap(),
        Array2::from_shape_vec((28, 28), c).unwrap()];
    let img = vec![image];


    let conv1 = nn::new("Conv".to_string(), vec![3, 6, 1, 1, 28, 3, 0], 0.001);
    let relu1 = nn::new("Relu".to_string(), vec![0], 0.001);
    let max1 = nn::new("Pool".to_string(), vec![4, 2, 0, 6, 28, 0], 0.001);

    let conv2 = nn::new("Conv".to_string(), vec![6, 10, 1, 1, 13, 3, 0], 0.001);
    let relu2 = nn::new("Relu".to_string(), vec![0], 0.001);
    let max2 = nn::new("Pool".to_string(), vec![3, 2, 0, 10, 13, 1], 0.001);

    let fc1 = nn::new("Full".to_string(), vec![100, 360, 10], 0.001);
    let relu3 = nn::new("Relu".to_string(), vec![0], 0.001);
    let fc2 = nn::new("Full".to_string(), vec![10, 100, 0], 0.001);
    let soft = nn::new("Softmax".to_string(), vec![1], 0.001);

    let model = vec![conv1, relu1, max1, conv2, relu2, max2, fc1, relu3, fc2, soft];

    let outputs = forward(model, img);

    println!("output {:?}", outputs[outputs.len() - 1]);
}
