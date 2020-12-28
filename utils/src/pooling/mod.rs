use ndarray::Array2;
use crate::propagation::Propagation;
use crate::utils;
use utils::{_max_pool, _upsample};
use std::cell::RefCell;

pub struct Pool {
    pub width: usize,
    pub stride: usize,
    pub padding: usize,
    pub boundary: usize,
    pub out_channel: usize,
    pub input_width: usize,
    pub positions: RefCell<Vec<Vec<Vec<usize>>>>
}

impl Propagation for Pool {
    fn forward(&self, inputs:&Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {

        // at pooling layer, out_channel==in_channel
        // inputs [sample, out_channel, input_width, input_width]
        // positions [sample, out_channel, index]    (index < input_width * input_width)
        let mut positions: Vec<Vec<Vec<usize>>> = vec![]; 

        let outputs = inputs.iter().map(|input| {
            let (output, pos) = self.single_max_pool(input);
            positions.push(pos);
            output
        }).collect::<Vec<Vec<Array2<f32>>>>();

        *self.positions.borrow_mut() = positions;
        outputs
    }

    fn backward(&self, _: Vec<Vec<Array2<f32>>>, next_deltas: Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        // next_deltas [samples, out_channel, output_width, output_width]

        next_deltas.into_iter().zip(self.positions.borrow().iter()).map(|(delta, pos)| {
            self.single_upsample(delta, pos)
        }).collect::<Vec<Vec<Array2<f32>>>>()
    }
}

impl Pool {

    pub fn new(width: usize,
        stride: usize,
        padding: usize, 
        out_channel: usize, 
        input_width: usize, 
        boundary: usize
    ) -> Pool {
        Pool {
            width,
            stride,
            padding,
            out_channel,
            input_width,
            boundary,
            positions: RefCell::new(vec![])
        }
    }

}

impl Pool {

    fn single_max_pool(&self, input: &Vec<Array2<f32>>) -> (Vec<Array2<f32>>, Vec<Vec<usize>>) {
        // input [out_channel, input_width, input_width]
        // outputs [out_channel, output_width, output_width]
        // max_positions [out_channel, index]
        let mut outputs: Vec<Array2<f32>> = vec![];
        let mut max_positions: Vec<Vec<usize>> = vec![];

        for val in input.iter() {
            let (output, position) = _max_pool(val, self.width, self.stride, self.padding, self.input_width);
            outputs.push(output);
            max_positions.push(position);
        }
        (outputs, max_positions)
    }

    fn single_upsample(&self, deltas: Vec<Array2<f32>>, pos: &Vec<Vec<usize>>) -> Vec<Array2<f32>> {
        // deltas [out_channel, output_width, output_width]
        // pos [out_channel, index]
        
        /* let deltas_with_channel = _restore_with_channel(deltas, self.out_channel, 
            self.input_width, self.width, self.stride, self.padding, self.boundary); */
        
        deltas.into_iter().enumerate().map(|(index, delta)| {
            _upsample(delta, &pos[index], self.input_width)
        }).collect::<Vec<Array2<f32>>>()
    }
}