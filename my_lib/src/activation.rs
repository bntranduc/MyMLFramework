use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    None,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::None => x,
            _ => panic!("apply method not used for Softmax"),
        }
    }

    pub fn apply_softmax(&self, x: &Vec<f64>) -> Vec<f64> {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = x.iter().map(|&xi| (xi - max).exp()).collect();
        let sum_exp_values: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&xi| xi / sum_exp_values).collect()
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => x * (1.0 - x),
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::None => 1.0,
            _ => panic!("derivative method not used for Softmax"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "relu" => Activation::ReLU,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            "none" => Activation::None,
            _ => panic!("Unknown activation function"),
        }
    }

    pub fn get_name(&self) -> &str {
        match self {
            Activation::ReLU => "ReLU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::None => "None",
            Activation::Softmax => "Softmax",
        }
    }
}

