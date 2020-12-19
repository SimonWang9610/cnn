use crate::utils;
use utils::{_convolution, _restore_with_channel, sum_nested_vector};
use utils::utils::{cal_shape, _rotate};

use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use std::cell::RefCell;

// transposed_filter will change after updating filter
// therefore, the method to update the filter must re-calculate the transposed filter
pub struct Conv2D {
    pub prev: usize,
    pub filter_width: usize,
    pub filter: RefCell<Array2<f32>>,
    pub bias: RefCell<Array2<f32>>,
    pub stride: usize,
    pub padding: usize,
    pub alpha: f32,
    pub transposed_filter: RefCell<Array2<f32>>
}

impl Conv2D {
    pub fn new(prev_width: usize, width: usize, padding: usize, stride: usize, alpha: f32) -> Conv2D {
        let (filter, bias) = Conv2D::initialization(prev_width, width, padding, stride);
        let transposed_filter = _rotate(&filter, 2);
        Conv2D {
            prev: prev_width,
            filter_width: width,
            filter: RefCell::new(filter),
            bias: RefCell::new(bias),
            stride,
            padding,
            alpha,
            transposed_filter: RefCell::new(transposed_filter)
        }
    }

    fn initialization(prev_width: usize, width: usize, stride: usize,padding: usize) 
    -> (Array2<f32>, Array2<f32>) {
        (
            Array::random((width, width), StandardNormal) * 0.05,
            Array::zeros((cal_shape(prev_width, width, stride, padding), 1))
        )
    }
}

impl Conv2D {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        _convolution(&*self.filter.borrow(), input, self.stride, self.padding) + &*self.bias.borrow()
    }

    pub fn cal_delta(&self, next_delta: &Array2<f32>) -> Array2<f32> {
        _convolution(&*self.transposed_filter.borrow(), next_delta, self.stride, self.padding)
    }

    pub fn update(&self, derivate_bias: &Array2<f32>, derivate_filter: &Array2<f32>) {
        let cloned_bias = self.bias.borrow().clone();
        let cloned_filter = self.filter.borrow().clone();

        *self.filter.borrow_mut() = cloned_filter - self.alpha * derivate_filter;
        *self.bias.borrow_mut() = cloned_bias - self.alpha * derivate_bias.sum_axis(Axis(1)) / self.prev as f32;
        *self.transposed_filter.borrow_mut() = _rotate(&*self.filter.borrow(), 2);
    }
}


pub struct Conv3D {
    pub in_channel: usize,
    pub out_channel: usize,
    pub stride: usize,
    pub padding: usize,
    pub prev_width: usize,
    pub output_width: usize,
    pub filter_width: usize,
    pub alpha: f32,
    pub boundary: usize,
    pub conv2d: RefCell<Vec<Vec<Conv2D>>>
}

impl Conv3D {
    pub fn new(in_channel: usize,
        out_channel: usize,
        stride: usize,
        padding: usize,
        prev_width: usize,
        filter_width: usize,
        alpha: f32,
        boundary: usize
    ) -> Conv3D {
        // conv2d: (out_channel, in_channel)
        let conv2d: Vec<Vec<Conv2D>> = (0..out_channel).map(
            |_| (0..in_channel).map(
                |_| Conv2D::new(prev_width, filter_width, padding, stride, alpha)
            ).collect::<Vec<Conv2D>>()
        ).collect();
        let output_width = cal_shape(prev_width, filter_width, stride, padding);

        Conv3D {
            in_channel,
            out_channel,
            stride,
            padding,
            prev_width,
            output_width,
            filter_width,
            alpha,
            boundary,
            conv2d: RefCell::new(conv2d)
        }
    }

    pub fn forward(&self, inputs: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        println!("input shape [{:?}, {:?}, {:?}]", inputs.len(), inputs[0].len(), inputs[0][0].shape());

        println!("Conv forwarding....");
        inputs.iter().map(|input| self._forward(input)).collect::<Vec<Vec<Array2<f32>>>>()
    }

    
    pub fn backward(&self, next_deltas: Vec<Vec<Array2<f32>>>, inputs: Vec<Vec<Array2<f32>>>) 
    -> Vec<Vec<Array2<f32>>> {
        // next_deltas : [sample, out_channel, output_width, output_width]
        // inputs: [sample, in_channel, input_width, input_width]
        // output: [sample, in_channel, input_width, input_width]
        let samples = next_deltas.len();

        let derivate_filters: Vec<Vec<Array2<f32>>> = next_deltas.iter().zip(inputs.into_iter()).fold(
            (0..self.out_channel).map(|_| {
                (0..self.in_channel).map(|_| {
                    Array::zeros((self.filter_width, self.filter_width))
                }).collect::<Vec<Array2<f32>>>()
            }).collect::<Vec<Vec<Array2<f32>>>>(),
            |acc, (delta, input)| {
                self.cal_derivate_filters(delta, input).into_iter().zip(acc.into_iter()).map(|(a, b)| {
                    sum_nested_vector(a, b)
                }).collect::<Vec<Vec<Array2<f32>>>>()
            }
        ); // [out_channel, in_channel, filter_width, filter_width]

        let average_deltas =  next_deltas.iter().fold(
            (0..self.out_channel).map(
                |_| 
                Array::zeros((self.output_width, self.output_width)))
                .collect::<Vec<Array2<f32>>>(),
                |acc, delta| {
                    let average_delta: Vec<Array2<f32>> = delta.iter().map(
                        |ele| ele / samples as f32
                    ).collect();
                    sum_nested_vector(acc, average_delta)
            }
        ); // [out_channel, output_width, output_width]

        for out in 0..self.out_channel {
            for i in 0..self.in_channel {
                self.conv2d.borrow_mut()[out][i].update(&average_deltas[out], &derivate_filters[out][i]);
            }
        }

        next_deltas.into_iter().map(|delta| {
            self.cal_delta(delta)
        }).collect::<Vec<Vec<Array2<f32>>>>()
    }
}

impl Conv3D {

    fn _forward(&self, input: &Vec<Array2<f32>>) -> Vec<Array2<f32>> {
        // input.len() == in_channel
        // output.len() == out_channel
        (0..self.out_channel).map(
            |out_index| {
                let outputs: Vec<Array2<f32>> = input.iter().zip(self.conv2d.borrow()[out_index].iter()).map(
                    |(data, conv)| conv.forward(data)
                ).collect();
                let summary = Array::zeros((self.output_width, self.output_width));
                outputs.into_iter().fold(summary, |acc, output| acc + output)
            }
        ).collect::<Vec<Array2<f32>>>()
    }

    fn cal_delta(&self, next_delta: Vec<Array2<f32>>) -> Vec<Array2<f32>> {
        // next_delta.len() == out_channel
        // output.len() == in_channel
        // when backward 
        // if conv layer is connected with full layer, must convert delta [sample, channel * width * width] to [sample, channel, width, width] 
        let delta_with_channel = _restore_with_channel(next_delta, self.out_channel,
            self.prev_width, self.filter_width, self.stride, self.padding, self.boundary);
        
        (0..self.in_channel).map(|in_index| {
            let outputs: Vec<Array2<f32>> = delta_with_channel.iter().enumerate().map(|(out_index, data)| {
                self.conv2d.borrow()[out_index][in_index].cal_delta(data)
            }).collect();
            let summary = Array::zeros((self.prev_width, self.prev_width));
            outputs.into_iter().fold(summary, |acc, output| acc + output)
        }).collect::<Vec<Array2<f32>>>()
    }

    fn cal_derivate_filters(&self, delta: &Vec<Array2<f32>>, input: Vec<Array2<f32>>) -> Vec<Vec<Array2<f32>>> {
        // delta [out_channel, width, width]
        // input [in_channel, width, width]
        (0..self.out_channel).map(|out_index| {
            (0..self.in_channel).map(|in_index| {
                _convolution(&delta[out_index], &input[in_index], self.stride, self.padding)
            }).collect::<Vec<Array2<f32>>>()
        }).collect::<Vec<Vec<Array2<f32>>>>()
    }
}