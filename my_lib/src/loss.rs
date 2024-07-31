use serde::{Serialize, Deserialize};

// Définition de la structure Loss
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Loss {
    MSE,
    CrossCategoricalEntropy,
}

impl Loss {
    pub fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        match self {
            Loss::MSE => {
                let mut loss = 0.0;
                for i in 0..predictions.len() {
                    loss += (predictions[i] - targets[i]).powi(2);
                }
                loss / predictions.len() as f64
            }
            Loss::CrossCategoricalEntropy => {
                let mut loss = 0.0;
                for i in 0..predictions.len() {
                    loss -= targets[i] * predictions[i].ln();
                }
                loss
            }
        }
    }

    pub fn compute_gradient(&self, predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        match self {
            Loss::MSE => predictions.iter().zip(targets).map(|(p, t)| p - t).collect(),
            Loss::CrossCategoricalEntropy => predictions.iter().zip(targets).map(|(p, t)| p - t).collect(),
        }
    }

    pub fn get_name(&self) -> &str {
        match self {
            Loss::MSE => "MSE",
            Loss::CrossCategoricalEntropy => "Categorical Cross Entropy",
        }
    }
}