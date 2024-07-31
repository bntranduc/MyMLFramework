use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand::rngs::StdRng;

use crate::optimizer::{self, Optimizer4D, OptimizerAlg};
use crate::activation::Activation;
use crate::utils::*;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConvLayer {
    pub input_shape: (usize, usize, usize),
    pub kernel_size: usize,
    pub output_shape: (usize, usize, usize),
    pub input: Vec<Vec<Vec<f64>>>,
    pub output: Vec<Vec<Vec<f64>>>,
    pub stride: usize,
    pub n_kernels: usize,
    pub kernels: Vec<Vec<Vec<Vec<f64>>>>,
    pub kernels_update: Vec<Vec<Vec<Vec<f64>>>>,
    pub optimizer: Optimizer4D,
    pub activation: Activation,
}

impl ConvLayer {
    pub fn new(input_shape: (usize, usize, usize), kernel_size: usize, stride: usize, n_kernels: usize, optimizer_alg: OptimizerAlg, activation: Activation, seed: Option<u64>) -> ConvLayer {
        let output_width: usize = ((input_shape.0 - kernel_size) / stride) + 1;
        let output_shape = (output_width, output_width, n_kernels);
        let mut kernels = vec![vec![vec![vec![0.; input_shape.2]; kernel_size]; kernel_size]; n_kernels];

        let mut rng: StdRng = match seed {
            Some(s) => SeedableRng::seed_from_u64(s),
            None => SeedableRng::from_entropy(),
        };

        let normal = Normal::new(0.0, 1.0).unwrap();

        for f in 0..n_kernels {
            for i in 0..kernel_size {
                for j in 0..kernel_size {
                    for k in 0..input_shape.2 {
                        kernels[f][i][j][k] = normal.sample(&mut rng) * (2.0/(input_shape.0.pow(2)) as f64).sqrt();
                    }
                }
            }
        }
        let optimizer = Optimizer4D::new(optimizer_alg);
        ConvLayer {
            input_shape,
            kernel_size,
            output_shape,
            input: vec![vec![vec![0.0; input_shape.2]; input_shape.1]; input_shape.0],
            output: vec![vec![vec![0.0; n_kernels]; output_width]; output_width],
            stride,
            n_kernels,
            kernels,
            kernels_update: vec![vec![vec![vec![0.0; input_shape.2]; kernel_size]; kernel_size]; n_kernels],
            optimizer,
            activation,
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        self.input = input;
        let mut output = vec![vec![vec![0.0; self.output_shape.2]; self.output_shape.1]; self.output_shape.0];

        for d in 0..self.n_kernels {
            for y in 0..self.output_shape.1 {
                for x in 0..self.output_shape.0 {
                    let mut sum = 0.0;
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            for c in 0..self.input_shape.2 {
                                sum += self.input[y * self.stride + ky][x * self.stride + kx][c] * self.kernels[d][ky][kx][c];
                            }
                        }
                    }
                    output[y][x][d] = self.activation.apply(sum);
                }
            }
        }

        self.output = output.clone();
        output
    }

    pub fn backward(&mut self, output_gradient: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let mut input_gradient = vec![vec![vec![0.0; self.input_shape.2]; self.input_shape.1]; self.input_shape.0];

        for k in 0..self.n_kernels {
            for y in 0..self.output_shape.1 {
                for x in 0..self.output_shape.0 {
                    let delta = self.activation.derivative(self.output[y][x][k]) * output_gradient[y][x][k];
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            for c in 0..self.input_shape.2 {
                                input_gradient[y * self.stride + ky][x * self.stride + kx][c] += delta * self.kernels[k][ky][kx][c];
                                self.kernels_update[k][ky][kx][c] += delta * self.input[y * self.stride + ky][x * self.stride + kx][c];
                            }
                        }
                    }
                }
            }
        }

        input_gradient
    }

    pub fn update(&mut self, batch_size: usize) {
        let kernel_updates = self.optimizer.weight_changes(&self.kernels_update);

        for f in 0..self.n_kernels {
            for i in 0..self.kernel_size {
                for j in 0..self.kernel_size {
                    for k in 0..self.input_shape.2 {
                        self.kernels[f][i][j][k] -= (kernel_updates[f][i][j][k] / batch_size as f64);
                        self.kernels_update[f][i][j][k] = 0.0;
                    }
                }
            }
        }
    }
}
