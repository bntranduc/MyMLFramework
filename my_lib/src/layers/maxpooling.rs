use crate::utils::print_non_zero_elements;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MaxPoolingLayer {
    pub pool_size: usize,
    pub input_shape: (usize, usize, usize),
    pub output_shape: (usize, usize, usize),
    pub gradient_mapping: Vec<Vec<Vec<(usize, usize, usize)>>>,
}

impl MaxPoolingLayer {
    pub fn new(input_shape:(usize,usize,usize), pool_size: usize) -> MaxPoolingLayer {
        MaxPoolingLayer {
            pool_size: pool_size,
            input_shape: input_shape,
            output_shape: (input_shape.0 - pool_size + 1, input_shape.1 - pool_size + 1, input_shape.2),
            gradient_mapping: vec![],
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let (input_height, input_width, input_depth) = (input.len(), input[0].len(), input[0][0].len());
        let (output_height, output_width, output_depth) = (
            self.output_shape.0,
            self.output_shape.1,
            self.output_shape.2,
        );

        let mut output = vec![vec![vec![0.; output_depth]; output_width]; output_height];
        let mut gradient_mapping = vec![vec![vec![(0, 0, 0); output_depth]; output_width]; output_height];
        for h in 0..output_height {
            for w in 0..output_width {
                for d in 0..output_depth {
                    let mut max = f64::NEG_INFINITY;
                    let mut pos_max: (usize, usize, usize) = (h, w, d);
                    for i in 0..self.pool_size {
                        for j in 0..self.pool_size {
                            if input[h + i][w + j][d] > max {
                                max = input[h + i][w + j][d].clone();
                                pos_max = (h + i, w + j, d);
                            }
                        }
                    }
                    output[h][w][d] = max;
                    gradient_mapping[h][w][d] = pos_max;
                }
            }
        }
        self.gradient_mapping = gradient_mapping.clone();
        output
    }

    pub fn backward(&self, output_gradient: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let mut input_gradient: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.; self.input_shape.2]; self.input_shape.1]; self.input_shape.0];
        let output_shape = (output_gradient.len(), output_gradient[0].len(), output_gradient[0][0].len());
        for h in 0..output_shape.0 {
            for w in 0..output_shape.1 {
                for d in 0..output_shape.2 {
                    let pos = self.gradient_mapping[h][w][d].clone();
                    input_gradient[pos.0][pos.1][pos.2] = output_gradient[h][w][d];
                }
            }
        }
        input_gradient
    }
}