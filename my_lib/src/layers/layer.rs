use serde::{Serialize, Deserialize};

use crate::layers::DenseLayer;
use crate::layers::ConvLayer;
use crate::layers::MaxPoolingLayer;
use crate::optimizer::{OptimizerAlg};
use crate::activation::{Activation};
use crate::utils::*;

#[derive(Serialize, Deserialize, Clone)]
pub enum LayerType {
    Conv(ConvLayer),
    MaxPooling(MaxPoolingLayer),
    Dense(DenseLayer),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    pub layer: LayerType,
}

impl Layer {
    pub fn new_conv(
        input_shape: (usize, usize, usize), 
        kernel_size: usize, 
        stride: usize, 
        n_kernels: usize, 
        optimizer_alg: OptimizerAlg, 
        activation: Activation, 
        seed: Option<u64>
    ) -> Self {
        Layer {
            layer: LayerType::Conv(ConvLayer::new(input_shape, kernel_size, stride, n_kernels, optimizer_alg, activation, seed)),
        }
    }
    
    pub fn new_dense(
        input_size: usize, 
        output_size: usize, 
        activation: Activation, 
        optimizer_alg: OptimizerAlg, 
        dropout: f64, 
        original_shape: (usize, usize, usize),
        seed: Option<u64>
    ) -> Self {
        Layer {
            layer: LayerType::Dense(DenseLayer::new(input_size, output_size, activation, optimizer_alg, dropout, original_shape, seed)),
        }
    }

    pub fn new_maxpooling(input_shape: (usize, usize, usize), pool_size: usize) -> Self {
        Layer {
            layer: LayerType::MaxPooling(MaxPoolingLayer::new(input_shape, pool_size)),
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        match &mut self.layer {
            LayerType::Conv(layer) => layer.forward(input),
            LayerType::MaxPooling(layer) => layer.forward(input),
            LayerType::Dense(layer) => {
                let input_flat = flatten(input);
                let output = layer.forward(input_flat, true);
                vec![vec![output]]
            },
        }
    }

    pub fn backward(&mut self, output_gradient: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        match &mut self.layer {
            LayerType::Conv(layer) => layer.backward(output_gradient),
            LayerType::MaxPooling(layer) => layer.backward(output_gradient),
            LayerType::Dense(layer) => {
                let output_gradient_flat = flatten(output_gradient);
                let input_gradient_flat = layer.backward(output_gradient_flat);
                reshape(input_gradient_flat, layer.original_shape)
            },
        }
    }

    pub fn update(&mut self, minibatch_size: usize, optimizer: OptimizerAlg) {
        match &mut self.layer {
            LayerType::Conv(layer) => layer.update(minibatch_size),
            LayerType::Dense(layer) => layer.update(minibatch_size),
            _ => (),
        }
    }
}

