use rand::Rng;
extern crate nalgebra as na;
use crate::utils::{reshape2D, print_matrix, gauss_kernel};

use na::{DMatrix, DVector};
use std::f64::consts::E;

pub struct MyRBF_Regression {
    pub weights: Vec<Vec<f64>>,
    pub gamma: f64,
    pub centers: Vec<Vec<f64>>,
}

impl MyRBF_Regression {
    pub fn new(centers: Vec<Vec<f64>>, gamma: f64 ) -> Self {
        let num_centers = centers.len();
        Self {
            weights: vec![vec![0.0; num_centers]; 0],
            gamma,
            centers,
        }
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>) {
        let num_samples = X_train.len();
        let num_centers = self.centers.len();
        let y_dim = y_train[0].len();

        let mut matrix = DMatrix::zeros(num_samples, num_centers);
        for i in 0..num_samples {
            for j in 0..num_centers {
                matrix[(i, j)] = gauss_kernel(&X_train[i], &self.centers[j], self.gamma);
            }
        }

        let target_matrix = DMatrix::from_fn(num_samples, y_dim, |i, j| y_train[i][j]);

        let matrix_t = matrix.transpose();
        let matrix_t_matrix = &matrix_t * &matrix;
        let inv_matrix = matrix_t_matrix.try_inverse().expect("Inverse matrix not found!");
        let matrix_t_y = &matrix_t * target_matrix;
        let weight_matrix = inv_matrix * matrix_t_y;
        
        self.weights = weight_matrix.row_iter().map(|row| row.iter().cloned().collect()).collect();
    }

    pub fn predict(&self, input: Vec<Vec<f64>>, is_classification:bool) -> Vec<Vec<f64>> {
        let num_samples = input.len();
        let num_centers = self.centers.len();
        let y_dim = self.weights[0].len();

        let mut predictions = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let mut pred: Vec<f64> = vec![0.0; y_dim];
            for j in 0..num_centers {
                let rbf_value = gauss_kernel(&input[i], &self.centers[j], self.gamma);
                let mut sum = 0.;
                for k in 0..y_dim {
                    pred[k] += self.weights[j][k] * rbf_value;
                }
            }
            if is_classification{
                for k in 0..y_dim{
                    if pred[k] < 0.{pred[k] = -1.;} else {pred[k] = 1.;}
                }
            }
            predictions.push(pred);
        }
        predictions
    }
}

#[no_mangle]
pub extern "C" fn create_MyRBF_Regression(p_centers:*const f64, 
                                centers_shape_0:i32, centers_shape_1:i32,

                                gamma:f64) -> *mut MyRBF_Regression{

    let flatten_centers = unsafe {
        {std::slice::from_raw_parts(p_centers, (centers_shape_0 * centers_shape_1) as usize)}
    }.to_vec();

    let centers = reshape2D(flatten_centers, (centers_shape_0 as usize, centers_shape_1 as usize));

    let model = MyRBF_Regression::new(centers, gamma);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);  
    leaked_model 
}

#[no_mangle]
pub extern "C" fn train_MyRBF_Regression(p_model:*mut MyRBF_Regression,

                                p_X_train:*const f64, 
                                X_train_shape_0:i32, X_train_shape_1:i32,

                                p_y_train:*const f64, 
                                y_train_shape_0:i32, y_train_shape_1:i32){

    let mut model = unsafe {&mut *p_model};

    let flatten_X_train = unsafe {
        {std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)}
    }.to_vec();
    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let flatten_y_train = unsafe {
        {std::slice::from_raw_parts(p_y_train, (y_train_shape_0 * y_train_shape_1) as usize)}
    }.to_vec();
    let y_train = reshape2D(flatten_y_train, (y_train_shape_0 as usize, y_train_shape_1 as usize));
    
    model.train(X_train, y_train);
}

#[no_mangle]
pub extern "C" fn predict_MyRBF_Regression(p_model:*mut MyRBF_Regression,
                                    
                                p_input:*const f64, 
                                input_shape_0:i32, input_shape_1:i32,
                            
                                is_classification:bool) -> *const f64{

    let mut model = unsafe {&mut *p_model};

    let flatten_input = unsafe {
        {std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)}
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = model.predict(input, is_classification);

    let mut flatten_pred = vec![];
    for i in 0..predictions.len(){
        for j in 0..predictions[0].len(){
            flatten_pred.push(predictions[i][j].clone());
        }
    }

    let leaked_predictions = Vec::leak(flatten_pred);
    leaked_predictions.as_ptr()
}
