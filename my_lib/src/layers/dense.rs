use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand::rngs::StdRng;

use crate::activation::Activation;
use crate::optimizer::{OptimizerAlg, Optimizer2D};
use crate::utils::*;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub input: Vec<f64>,
    pub output: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub weights_update: Vec<Vec<f64>>,
    pub biases_update: Vec<f64>,
    pub activation: Activation,
    pub original_shape: (usize, usize, usize),
    pub optimizer: Optimizer2D,
    pub dropout: f64,
    pub dropout_mask: Vec<f64>,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        optimizer_alg: OptimizerAlg,
        dropout: f64,
        original_shape: (usize, usize, usize),
        seed: Option<u64>,
    ) -> DenseLayer {
        // Utilisation de la seed si elle est fournie
        let mut rng: StdRng = match seed {
            Some(s) => SeedableRng::seed_from_u64(s),
            None => SeedableRng::from_entropy(),
        };
        
        let normal = Normal::new(0.0, (2.0 / input_size as f64).sqrt()).unwrap();

        let mut weights = vec![vec![0.0; input_size]; output_size];
        let mut biases = vec![0.01; output_size];

        for i in 0..output_size {
            for j in 0..input_size {
                weights[i][j] = normal.sample(&mut rng);
            }
        }
        let optimizer = Optimizer2D::new(optimizer_alg);
        DenseLayer {
            input_size,
            output_size,
            input: vec![0.0; input_size],
            output: vec![0.0; output_size],
            weights,
            biases,
            weights_update: vec![vec![0.0; input_size]; output_size],
            biases_update: vec![0.0; output_size],
            activation,
            original_shape,
            optimizer,
            dropout,
            dropout_mask: vec![1.0; input_size],
        }
    }

    pub fn forward(&mut self, input: Vec<f64>, training: bool) -> Vec<f64> {
        self.input = input.clone();
        let mut output = vec![0.0; self.output_size];
    
        if training {
            let mut rng = rand::thread_rng();
            for i in 0..self.input_size {
                let random = rng.gen::<f64>();
                if random < self.dropout {
                    self.dropout_mask[i] = 0.0;
                } else {
                    self.dropout_mask[i] = 1.0;
                }
            }
        }
    
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                output[i] += self.weights[i][j] * input[j] * self.dropout_mask[j];
            }
            output[i] += self.biases[i];
        }
    
        self.output = if let Activation::Softmax = self.activation {
            self.activation.apply_softmax(&output)
        } else {
            output.iter().map(|&o| self.activation.apply(o)).collect()
        };
    
        self.output.clone()
    }
    

    pub fn backward(&mut self, mut output_gradient: Vec<f64>) -> Vec<f64> {
        if let Activation::Softmax = self.activation {
            // Softmax derivative logic
            let softmax_derivative: Vec<f64> = self.output.iter().map(|&o| o * (1.0 - o)).collect();
            for i in 0..output_gradient.len() {
                output_gradient[i] *= softmax_derivative[i];
            }
        } else {
            for i in 0..output_gradient.len() {
                output_gradient[i] *= self.activation.derivative(self.output[i]);
            }
        }
    
        for i in 0..output_gradient.len() {
            output_gradient[i] *= self.dropout_mask[i];
        }
    
        let mut input_gradient = vec![0.0; self.input_size];
        for i in 0..self.input_size {
            for j in 0..self.output_size {
                input_gradient[i] += self.weights[j][i] * output_gradient[j];
            }
        }
    
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights_update[i][j] += output_gradient[i] * self.input[j];
            }
            self.biases_update[i] += output_gradient[i];
        }
        input_gradient
    }
    
    

    pub fn update(&mut self, minibatch_size: usize) {
        let weight_updates = self.optimizer.weight_changes(&self.weights_update);
        let bias_updates = self.optimizer.bias_changes(&self.biases_update);

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights[i][j] -= weight_updates[i][j] / minibatch_size as f64;
            }
            self.biases[i] -= bias_updates[i] / minibatch_size as f64;
        }

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights_update[i][j] = 0.0;
            }
            self.biases_update[i] = 0.0;
        }

        self.weights_update = weight_updates;
        self.biases_update = bias_updates;
    }
}
