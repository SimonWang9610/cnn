pub mod utils;
use ndarray::{Array2};

use utils::{flip_matrix, cal_shape, im2col, im2col_filter, _rotate, _restore_max_index};

pub fn _max_pool(
    input: &Array2<f32>, 
    width: usize, 
    stride: usize, 
    padding: usize, 
    input_width: usize
) -> (Array2<f32>, Vec<usize>) {
    let output_width = cal_shape(input_width, width, stride, padding);
    let mut indices: Vec<usize> = vec![];
    // return (max_pooled_input, index_max_values)
    let flipped_matrix: Vec<(usize, usize, f32)> = flip_matrix(input, width, stride)
    .into_iter()
    .enumerate()
    .map(
        |(index, v)| {
            let pair = v.into_iter()
            .enumerate().fold((0, f32::MIN), |(max_pos, acc), (pos, x)| {
            let max = acc.max(x);
            if max == acc {
                (max_pos, acc)
            } else {
                (pos, max)
            }
        });
        (index, pair.0, pair.1)
        }
    ).collect();
    let max_values: Vec<f32> = flipped_matrix.into_iter().map(|(index, pos, val)| {
        indices.push(_restore_max_index(output_width, (index, pos), stride, input.shape()[0], width));
        val
    }).collect();
    (Array2::from_shape_vec((output_width, output_width), max_values).unwrap(), indices)
}

pub fn _convolution(filter: &Array2<f32>, input: &Array2<f32>, stride: usize, padding: usize) -> Array2<f32> {
    // return [1, f*f]x[f*f, out*out] = [out, out]
    let filter_width = filter.shape()[0];
    let input_width = input.shape()[0];
    let output_width = cal_shape(input_width, filter_width, stride, padding);


    let im2col_input = im2col(&input, filter_width, stride, padding);
    let im2col_filter = im2col_filter(&filter, filter_width);

    im2col_input.dot(&im2col_filter).into_shape((output_width, output_width)).unwrap()
}


pub fn rotation(filters: &Vec<Array2<f32>>) -> Vec<Array2<f32>> {

    filters.iter().map(|filter| _rotate(filter, 1)).collect::<Vec<Array2<f32>>>()

}


pub fn _upsample(delta: Array2<f32>, positions: &Vec<usize>, input_width: usize) -> Array2<f32> {
    let mut output: Vec<f32> = (0..input_width * input_width).map(|_| 0.).collect();

    for (i, &val) in delta.iter().enumerate() {
        output[positions[i]] = val;
    }
    Array2::from_shape_vec((input_width, input_width), output).unwrap()
    
    // the elements of positions[i] are not in an order, as a result, the below logic was wrong

    /* println!("positions {:?}", positions);
    let mut delta_iter = delta.iter().enumerate();
    let mut pair = delta_iter.next().unwrap();
    let output = (0..output_width*output_width).map(|index| {
        let ele = if index == positions[pair.0] {
            let v = *pair.1;
            pair = delta_iter.next().unwrap();
            v
        } else {
            0.
        };
        println!("index {:?} ele {}", index, ele);
        ele
    }).collect::<Vec<f32>>(); */

}

pub fn _restore_with_channel(matrix: Vec<Array2<f32>>,
    out_channel: usize,
    prev_width: usize,
    filter_width: usize,
    stride: usize,
    padding: usize,
    boundary: usize) -> Vec<Array2<f32>> {
    
    if boundary == 1 {
        let data_width = cal_shape(prev_width, filter_width, stride, padding);
        let mut data_iter = matrix[0].into_iter();
        
        (0..out_channel).map(|_| {
            let one_channel_vector: Vec<f32> = (0..data_width * data_width).map(|_| *data_iter.next().unwrap())
            .collect();
            Array2::from_shape_vec((data_width, data_width), one_channel_vector).unwrap()
        }).collect::<Vec<Array2<f32>>>()
    } else {
        matrix
    }
}

pub fn _flatten_withno_channel(
    inputs: &Vec<Vec<Array2<f32>>>,
    prev_neurons: usize
) -> Vec<Array2<f32>> {
    // input [sample, prev_neurons]
    // at the boundary, flattened pixels == prev_neurons
    let samples = inputs.len();
    let vectors = inputs.iter().map(|input| {
        let input_as_vector = input.iter().map(
            |v| v.iter().map(|ele| *ele).collect::<Vec<f32>>()
        ).collect::<Vec<Vec<f32>>>();
        input_as_vector.into_iter().flatten().collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>();

    vec![Array2::from_shape_vec((samples, prev_neurons), vectors.into_iter().flatten().collect::<Vec<f32>>()).unwrap()]
}
    
pub fn sum_nested_vector(a: Vec<Array2<f32>>, b: Vec<Array2<f32>>) -> Vec<Array2<f32>> {
    a.into_iter().zip(b.iter()).map(|(i, j)| {
        i + j
    }).collect::<Vec<Array2<f32>>>()
}


