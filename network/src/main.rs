extern crate ndarray;
extern crate utils;
extern crate image;

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

    let conv3d = Conv3D::new(3, 6, 1, 1, 28, 3);
    let outputs = conv3d.forward(&image);
    println!("outputs length {:?} output[0] shape {:?}", outputs.len(), outputs[0].shape());

    /* let output = conv2d.forward(&image[0]);
    println!("output shape {:?}", output.shape());
    
    let delta: Array2<f32> = Array::random((28, 28), StandardNormal);
    let result = conv2d.cal_delta(&delta);
    println!("result shape {:?}", result.shape()); */
    // println!(" a {}", a);
    // let flipped = flip_matrix(&a, 3, 1, 1);
    // println!("flipped {:?} * {:?}", flipped.len(), flipped[0].len());
    // println!("flipped {:#?}", flipped);
}
