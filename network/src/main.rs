extern crate image;
extern crate ndarray;
extern crate utils;

use utils::network::{nn, forward, backward};
use utils::pooling::Pool;
use utils::convolution::{Conv3D, Conv2D};
use utils::full_connected::FullLayer;
use ndarray::{Array2, Array};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use image::io::Reader;
fn main() {
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

    let mut model = vec![conv1, relu1, max1, conv2, relu2, max2, fc1, relu3, fc2, soft];

    println!("conv1 {}", model[0]);

    let mut outputs = forward(&model, img);
    let target = Array2::from_shape_vec((1, 10), vec![1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
    let deltas = vec![vec![outputs.pop().unwrap()[0][0].clone().reversed_axes() - target]];

    backward(&mut model, outputs, deltas);
    println!("updated conv1 {}", model[0]);

}
