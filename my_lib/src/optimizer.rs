use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OptimizerAlg {
    SGD(f64),

}

impl OptimizerAlg {
    pub fn get_name(&self) -> &str {
        match self {
            OptimizerAlg::SGD(_) => "SGD",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Optimizer2D {
    pub alg: OptimizerAlg,
}

impl Optimizer2D {
    pub fn new(alg: OptimizerAlg) -> Self {
        Optimizer2D {alg}
    }

    pub fn weight_changes(&self, gradients: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut weights_update = vec![vec![0.;gradients[0].len()];gradients.len()];

        match self.alg {
            OptimizerAlg::SGD(lr) => {
                for i in 0..gradients.len(){
                    for j in 0..gradients[0].len(){
                        weights_update[i][j] = gradients[i][j] * lr
                    }
                }
                weights_update
            },
        }
    }

    pub fn bias_changes(&self, gradients: &Vec<f64>) -> Vec<f64> {
        let mut bias_update = vec![0.; gradients.len()]; 
        match self.alg {
            OptimizerAlg::SGD(lr) => {
                for i in 0..gradients.len(){
                    bias_update[i] = gradients[i] * lr;
                }
                bias_update
            },
        }
    }

    pub fn get_name(&self) -> &str {
        self.alg.get_name()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Optimizer4D {
    pub alg: OptimizerAlg,
}

impl Optimizer4D {
    pub fn new(alg: OptimizerAlg) -> Self {
        Optimizer4D {alg}
    }

    pub fn weight_changes(&self, gradients: &Vec<Vec<Vec<Vec<f64>>>>) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut weights_update = vec![vec![vec![vec![0.; gradients[0][0][0].len()]; gradients[0][0].len()]; gradients[0].len()]; gradients.len()];

        match self.alg {
            OptimizerAlg::SGD(lr) => {
                for i in 0..gradients.len() {
                    for j in 0..gradients[0].len() {
                        for k in 0..gradients[0][0].len() {
                            for l in 0..gradients[0][0][0].len() {
                                weights_update[i][j][k][l] = gradients[i][j][k][l] * lr;
                            }
                        }
                    }
                }
                weights_update
            },
        }
    }

    pub fn bias_changes(&self, gradients: &Vec<f64>) -> Vec<f64> {
        let mut bias_update = vec![0.; gradients.len()];
        match self.alg {
            OptimizerAlg::SGD(lr) => {
                for i in 0..gradients.len() {
                    bias_update[i] = gradients[i] * lr;
                }
                bias_update
            },
        }
    }

    pub fn get_name(&self) -> &str {
        self.alg.get_name()
    }
}