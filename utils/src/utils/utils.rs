use ndarray::{Array2, Array, Axis};
use std::f32::consts::E;

pub fn im2col_filter(filter: &Array2<f32>, width: usize) -> Array2<f32> {
    let filter_as_vector = filter.iter().map(|&x| x).collect::<Vec<f32>>();
    Array2::from_shape_vec((width * width, 1), filter_as_vector).unwrap()
}

pub fn im2col_input(input: &Array2<f32>, width: usize, filter_width: usize) -> Array2<f32> {
    //currently, only support stride=1
    let mut input_as_vector: Vec<f32> = vec![];
    let windows = input.windows((filter_width, filter_width));
    
    for slide in windows {
        for &num in slide.into_iter() {
            input_as_vector.push(num);
        }
    }
    Array2::from_shape_vec((filter_width * filter_width, width * width), input_as_vector).unwrap()
}


pub fn padding(matrix: &Array2<f32>) -> Array2<f32> {
    // default pad=1
    // currently only support pad=1
    let width = matrix.shape()[0];
    let mut matrix_iter = matrix.into_iter();
    let mut padded_matrix_as_vec: Vec<f32> = (0..width+2).map(|_| 0.).collect();

    for _ in 0..width {
        padded_matrix_as_vec.push(0.);
        for _ in 0..width {
            padded_matrix_as_vec.push(*matrix_iter.next().unwrap());
        }
        padded_matrix_as_vec.push(0.);
    }

    padded_matrix_as_vec.append(&mut (0..width+2).map(|_| 0.).collect::<Vec<f32>>());
    Array2::from_shape_vec((width+2, width+2), padded_matrix_as_vec).unwrap()
}

pub fn flip_matrix(matrix: &Array2<f32>, width: usize, stride: usize, _padding: usize) -> Vec<Vec<f32>> {
    // input must be padded before scan the input array
    let padded_matrix = padding(matrix);
    
    let input_width = padded_matrix.shape()[0];
    let mut flipped_vector = vec![];
    let mut x = 0 as usize;
    let mut y = 0  as usize;


    while x < input_width - width + 1 {

        while y < input_width - width + 1 {
            let mut filter_vector = vec![];
            for i in 0..width {
                for j in 0..width {
                    let temp = padded_matrix.get((x+i, y+j)).unwrap();
                    filter_vector.push(*temp);
                }
            }
            flipped_vector.push(filter_vector);
            y += stride;
        }
        y = 0 as usize;
        x += stride;
    }
    flipped_vector
}

pub fn im2col(matrix: &Array2<f32>, width: usize, stride: usize, padding: usize) -> Array2<f32> {
    let flipped_as_nested_vector: Vec<Vec<f32>> = flip_matrix(matrix, width, stride, padding);
    let shape = cal_shape(matrix.shape()[0], width, stride, padding);

    Array::from_shape_vec((shape * shape, width * width), 
    flipped_as_nested_vector.into_iter().flatten().collect::<Vec<f32>>()).unwrap()
}

pub fn cal_shape(input_width: usize, filter_width: usize, stride: usize, padding: usize) -> usize {
    (input_width - filter_width + 2 * padding) / stride + 1
}

pub fn _rotate(matrix: &Array2<f32>, degree: usize) ->  Array2<f32> {
    // 1 == 90 degree( clockwise rotation)
    // 2 == 180 degree
    // 3 == 270 degree
    let length = matrix.shape()[0];
    let offset = length - 1;

    let mut b: Vec<f32> = (0..length*length).map(|_| 0.).collect();

    for row in 0..length {
        for col in 0..length {
            let index = if degree == 1 {
                col * length + offset - row
            } else if degree == 2 {
                (offset - row) * length + offset - col
            } else if degree == 3 {
                (offset - col) * length + row
            } else {
                row * length + col
            };
            b[index] = *matrix.get((row, col)).unwrap();
        }
    }

    Array2::from_shape_vec((length, length), b).unwrap()
}

pub fn cal_backward_shape(input_width: usize, width: usize, stride: usize, padding: usize) -> usize {
    (input_width - 1) * stride - 2 * padding + width
}

pub fn _restore_max_index(offset: usize, pair: (usize, usize), stride: usize, length: usize, width: usize) -> usize {
    // offset = (w - f + 2p) / s + 1
    // pair: (index, pos)
    // length = input_width
    // width = pooling filter width
    // original coordinate:
    // row -> (index / output_width) * s + pos / s
    //col -> (index % output_width) * s + pos % s
    let row = pair.0 / offset * stride + pair.1 / width;
    let col = pair.0 % offset * stride + pair.1 % width;
    row * length + col
}

////////////////////////////////////////////////////////////
// below functions for full connected layer
////////////////////////////////////////////////////////////
pub fn compute_loss(output: &Array2<f32>, labels: &Array2<f32>) -> f32 {
    // output [sample, 10]
    // labels [sample, 10]
    let average = -1. / labels.shape()[0] as f32;
    output
        .into_iter()
        .zip(labels.iter())
        .fold(0., |acc, (o, l)| acc + l * o.log(E)) * average
}

pub fn _relu(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|ele| if ele >= 0. { ele } else { 0. })
}

pub fn _softmax(input: &mut Array2<f32>) -> Vec<Array2<f32>> {
    input.swap_axes(0, 1);
    let exp_sum = input
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f32| acc + ele.exp())
        })
        .into_shape((input.shape()[0], 1))
        .unwrap();
    let exp_input = input.mapv(|ele| ele.exp());
    vec![exp_input / exp_sum]
}

pub fn relu_derivate(matrix: Array2<f32>) -> Array2<f32> {
    matrix.mapv_into(|ele| if ele >= 0. {1.} else {0.})
}