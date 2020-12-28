extern crate image;
extern crate ndarray;
extern crate utils;

use utils::dataset::load_mnist;
use utils::network::{nn, train, train_one_by_one};

use std::path::Path;

fn main() {

    let paths: Vec<&Path> = vec![
        "./mnist/train-images.idx3-ubyte".as_ref(),
        "./mnist/train-labels.idx1-ubyte".as_ref(),
        "./mnist/t10k-images.idx3-ubyte".as_ref(),
        "./mnist/t10k-labels.idx1-ubyte".as_ref(),

    ];

    let ((x_train, y_train), (x_test, y_test)) = load_mnist(paths);
    println!("Data loaded!");

    let alpha = 0.001;
    let mut network = create_network(alpha);
    println!("Network created!");

    println!("Starting training...");
    train_one_by_one(&mut network, 5, x_test, y_test);

}

pub fn create_network(alpha: f32) -> Vec<nn> {
    let conv1 = nn::new("Conv".to_string(), vec![1, 3, 1, 1, 28, 3, 0], alpha);
    let relu1 = nn::new("Relu".to_string(), vec![0], alpha);
    let max1 = nn::new("Pool".to_string(), vec![4, 2, 0, 3, 28, 0], alpha);

    let conv2 = nn::new("Conv".to_string(), vec![3, 6, 1, 1, 13, 3, 0], alpha);
    let relu2 = nn::new("Relu".to_string(), vec![0], alpha);
    let max2 = nn::new("Pool".to_string(), vec![3, 2, 0, 10, 13, 1], alpha);

    let fc1 = nn::new("Full".to_string(), vec![100, 216, 6], alpha);
    let relu3 = nn::new("Relu".to_string(), vec![0], alpha);
    let fc2 = nn::new("Full".to_string(), vec![10, 100, 0], alpha);
    let soft = nn::new("Softmax".to_string(), vec![1], alpha);

    vec![conv1, relu1, max1, conv2, relu2, max2, fc1, relu3, fc2, soft]
}
