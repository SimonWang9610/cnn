use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::Debug;

use ndarray::{Array,Array2};
use std::cell::RefCell;

use crate::convolution::{Conv2D, Conv3D};
use crate::utils::utils::cal_shape;
use crate::trained::Convert;
use std::string::ToString;

#[derive(Deserialize, Serialize, Debug)]
pub struct Conv3DJson {
    pub in_channel: usize,
    pub out_channel: usize,
    pub stride: usize,
    pub padding: usize,
    pub prev_width: usize,
    pub output_width: usize,
    pub filter_width: usize,
    pub alpha: f32,
    pub boundary: usize,
    pub conv2d: Vec<Vec<Conv2DJson>>
}

impl Convert<Conv3D, Conv3DJson> for Conv3DJson {
    fn new(conv: Conv3D) -> Conv3DJson {
        let conv2d_json: Vec<Vec<Conv2DJson>> = conv.conv2d.into_inner().into_iter()
            .map(|convs| {
                convs.into_iter().map(|conv| Conv2DJson::new(conv)).collect::<Vec<Conv2DJson>>()
            }).collect();

        Conv3DJson {
            in_channel: conv.in_channel,
            out_channel: conv.out_channel,
            stride: conv.stride,
            padding: conv.padding,
            prev_width: conv.prev_width,
            output_width: conv.output_width,
            filter_width: conv.filter_width,
            alpha: conv.alpha,
            boundary: conv.boundary,
            conv2d: conv2d_json
        }
    }

    fn to_layer(self) -> Conv3D {

        let conv2d: Vec<Vec<Conv2D>> = self.conv2d.into_iter().map(
            |convs_json| {
                convs_json.into_iter().map(|conv_json| conv_json.to_layer()).collect::<Vec<Conv2D>>()
            }
        ).collect();

        Conv3D {
            in_channel: self.in_channel,
            out_channel: self.out_channel,
            stride: self.stride,
            padding: self.padding,
            prev_width: self.prev_width,
            output_width: self.output_width,
            filter_width: self.filter_width,
            alpha: self.alpha,
            boundary: self.boundary,
            conv2d: RefCell::new(conv2d)
        }
    }
}

impl ToString for Conv3DJson {
    
    fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Conv2DJson {
    pub prev: usize,
    pub filter_width: usize,
    pub filter: Vec<f32>,
    pub bias: Vec<f32>,
    pub stride: usize,
    pub padding: usize,
    pub alpha: f32
}

impl Convert<Conv2D, Conv2DJson> for Conv2DJson {
    fn new(conv: Conv2D) -> Conv2DJson {
        Conv2DJson {
            prev: conv.prev,
            filter_width: conv.filter_width,
            filter: conv.filter.borrow()
                .iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            bias: conv.bias.borrow()
                .iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            stride: conv.stride,
            padding: conv.padding,
            alpha: conv.alpha
        }
    }

    fn to_layer(self) -> Conv2D {

        let filter: Array2<f32> = Array2::from_shape_vec((self.filter_width, self.filter_width), self.filter).unwrap();
        let bias: Array2<f32> = Array2::from_shape_vec((
            cal_shape(self.prev, self.filter_width, self.stride, self.padding), 1), 
            self.bias
        ).unwrap();

        Conv2D {
            prev: self.prev,
            filter_width: self.filter_width,
            filter: RefCell::new(filter),
            bias: RefCell::new(bias),
            stride: self.stride,
            padding: self.padding,
            alpha: self.alpha,
            transposed_filter: RefCell::new(Array::zeros((self.filter_width, self.filter_width)))
        }
    }
}

impl ToString for Conv2DJson {
    
    fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}