use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Serialize, Deserialize};
use chrono::prelude::*;
use std::fs::{self, create_dir_all};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use std::slice;
use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_void};

extern crate nalgebra as na;
use crate::utils::*;
use na::{DMatrix};

#[derive(Serialize, Deserialize, Clone)]
pub struct MyLinearModel {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl MyLinearModel {
    pub fn new(n_features: usize, n_classes: usize, seed: Option<u64>) -> Self {
        let mut weights = vec![vec![0.0; n_features]; n_classes];
        let mut biases = vec![0.0; n_classes];

        if let Some(seed) = seed {
            let mut rng = StdRng::seed_from_u64(seed);
            for i in 0..weights.len() {
                for j in 0..weights[i].len() {
                    weights[i][j] = rng.gen_range(-1.0..1.0);
                }
                biases[i] = rng.gen_range(-1.0..1.0);
            }
        } else {
            let mut rng = rand::thread_rng();
            for i in 0..weights.len() {
                for j in 0..weights[i].len() {
                    weights[i][j] = rng.gen_range(-1.0..1.0);
                }
                biases[i] = rng.gen_range(-1.0..1.0);
            }
        }

        MyLinearModel {
            weights,
            biases,
        }
    }

    pub fn train(&mut self, 
        X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, 
        X_test: Vec<Vec<f64>>, y_test: Vec<Vec<f64>>, 
        learning_rate: f64, epochs: usize, 
        is_classification: bool) {

        let datetime_str = current_datetime_str();
        let base_dir = "../saved_models/LinearModel";
        let (model_dir, epoch_dir, best_model_dir) = create_model_dirs(base_dir, &datetime_str);

        let metrics_file_path = format!("{}/metrics.json", model_dir);
        let mut metrics = if std::path::Path::new(&metrics_file_path).exists() {
            let contents = fs::read_to_string(&metrics_file_path).expect("Failed to read metrics file");
            serde_json::from_str(&contents).expect("Failed to parse metrics file")
        } else {
            serde_json::json!({
                "train_accuracies": [],
                "train_losses": [],
                "test_accuracies": [],
                "test_losses": [],
                "epoch_times": []
            })
        };
        
        let X_train_size = X_train.len();
        let X_test_size = X_test.len();

        let mut best_test_acc = 0.0;

        println!("Training starting...");

        for epoch in 0..epochs {
            let start_time = SystemTime::now();
            
            let mut avg_train_acc = 0.0;
            let mut avg_train_loss = 0.0;

            for i in 0..X_train.len() {
                let label = y_train[i].clone();
                
                let pred = self.predict_single(&X_train[i], is_classification);
                let mut error = vec![0.0; y_train[0].len()];
                for j in 0..y_train[0].len() {
                    error[j] = pred[j] - label[j];
                }

                for j in 0..label.len() {
                    for k in 0..X_train[0].len() {
                        self.weights[j][k] -= learning_rate * error[j] * X_train[i][k];
                    }
                    self.biases[j] -= learning_rate * error[j];
                }

                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (pred[j] - label[j]).powf(2.);
                }
                avg_train_loss += loss;
                avg_train_acc += (argmax(&pred) == argmax(&label)) as i32 as f64;
            }

            avg_train_loss /= X_train.len() as f64;
            avg_train_acc /= X_train.len() as f64;

            let mut avg_test_acc = 0.0;
            let mut avg_test_loss = 0.0;
            for i in 0..X_test.len() {
                let label = y_train[i].clone();

                let pred = self.predict_single(&X_test[i], is_classification);
                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (pred[j] - label[j]).powf(2.);
                }
                avg_test_loss += loss;
                avg_test_acc += (argmax(&pred) == argmax(&label)) as i32 as f64;
            }
            avg_test_acc /= X_test_size as f64;
            avg_test_loss /= X_test_size as f64;

            println!("Epoch {}: Train Loss: {:.4}, Test Loss: {:.4}, Train Acc: {:.1}%, Test Acc: {:.1}%", 
                    epoch, 
                    avg_train_loss, 
                    avg_test_loss, 
                    avg_train_acc * 100., 
                    avg_test_acc * 100.
                    );

            // self.save(&format!("{}/model_epoch_{}.json", epoch_dir, epoch)).expect("Failed to save model");

            let epoch_duration = SystemTime::now().duration_since(start_time).unwrap().as_secs() as usize;

            metrics["train_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_train_acc));
            metrics["train_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_train_loss));
            metrics["test_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_test_acc));
            metrics["test_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_test_loss));
            metrics["epoch_times"].as_array_mut().unwrap().push(serde_json::json!(epoch_duration));
            fs::write(&metrics_file_path, serde_json::to_string_pretty(&metrics).expect("Failed to serialize metrics")).expect("Failed to save metrics");

            if avg_test_acc as f64 > best_test_acc {
                best_test_acc = avg_test_acc;
                self.save(&format!("{}/best_model.json", best_model_dir)).expect("Failed to save best model");
            }
        }
    }

    fn predict_single(&self, x: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        let mut linear_outputs: Vec<f64> = self.biases.clone();
        for (class, class_weights) in self.weights.iter().enumerate() {
            for (weight, &feature) in class_weights.iter().zip(x.iter()) {
                linear_outputs[class] += weight * feature;
            }
        }
        if is_classification {
            linear_outputs.iter().map(|&x| x.tanh()).collect()
        } else {
            linear_outputs
        }
    }

    pub fn predict(&self, X_test: Vec<Vec<f64>>, is_classification: bool) -> Vec<Vec<f64>> {
        X_test.iter().map(|x| self.predict_single(x, is_classification)).collect()
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let serialized = serde_json::to_string(&self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("Serialization Error: {}", e))
        })?;
        std::fs::write(path, serialized)
    }

    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let contents = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&contents).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("Deserialization Error: {}", e))
        })?;
        Ok(model)
    }
}

#[no_mangle]
pub extern "C" fn create_MyLinearModel(n_features: usize, n_classes: usize, seed: u64) -> *mut MyLinearModel {
    let seed_option = if seed == 0 { None } else { Some(seed) };
    let model = MyLinearModel::new(n_features, n_classes, seed_option);
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn destroy_MyLinearModel(model: *mut MyLinearModel) {
    if !model.is_null() {
        unsafe {
            Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn train_MyLinearModel(
    model: *mut MyLinearModel,
    X_train_ptr: *const c_double,
    y_train_ptr: *const c_double,
    X_test_ptr: *const c_double,
    y_test_ptr: *const c_double,
    num_train_samples: usize,
    num_test_samples: usize,
    n_features: usize,
    n_classes: usize,
    learning_rate: f64,
    epochs: usize,
    is_classification: c_int,
) {
    let model = unsafe { &mut *model };

    let X_train = unsafe { 
        slice::from_raw_parts(X_train_ptr, num_train_samples * n_features)
            .chunks(n_features)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let y_train = unsafe { 
        slice::from_raw_parts(y_train_ptr, num_train_samples * n_classes)
            .chunks(n_classes)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let X_test = unsafe { 
        slice::from_raw_parts(X_test_ptr, num_test_samples * n_features)
            .chunks(n_features)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let y_test = unsafe { 
        slice::from_raw_parts(y_test_ptr, num_test_samples * n_classes)
            .chunks(n_classes)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    model.train(X_train, y_train, X_test, y_test, learning_rate, epochs, is_classification != 0);
}

#[no_mangle]
pub extern "C" fn predict_MyLinearModel(
    model: *mut MyLinearModel,
    X_test_ptr: *const c_double,
    num_samples: usize,
    n_features: usize,
    is_classification: c_int
) -> *mut c_double {
    let model = unsafe { &mut *model };

    let X_test = unsafe { 
        slice::from_raw_parts(X_test_ptr, num_samples * n_features)
            .chunks(n_features)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let predictions = model.predict(X_test, is_classification != 0);

    let mut flat_predictions = predictions.into_iter().flat_map(|x| x).collect::<Vec<f64>>();
    let predictions_ptr = flat_predictions.as_mut_ptr();
    std::mem::forget(flat_predictions);

    predictions_ptr
}

#[no_mangle]
pub extern "C" fn save_MyLinearModel(model: *mut MyLinearModel, path: *const c_char) {
    let model = unsafe { &mut *model };
    let c_str = unsafe {CStr::from_ptr(path)};
    let path_str = c_str.to_str().unwrap();
    model.save(path_str).expect("Failed to save model");
}

#[no_mangle]
pub extern "C" fn load_MyLinearModel(path: *const c_char) -> *mut MyLinearModel {
    let c_str = unsafe {CStr::from_ptr(path)};
    let path_str = c_str.to_str().unwrap();
    let model = MyLinearModel::load(path_str).expect("Failed to load model");
    Box::into_raw(Box::new(model))
}


#[derive(Serialize, Deserialize, Clone)]
pub struct MyLinearModel_Regression {
    weights: Vec<Vec<f64>>,
}

impl MyLinearModel_Regression {
    pub fn new() -> Self {
        MyLinearModel_Regression { 
            weights: vec![] 
        }
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, y: Vec<Vec<f64>>) {
        let X_train_matrix = DMatrix::from_vec(X_train.len(), X_train[0].len(), X_train.iter().flatten().cloned().collect());
        let y_matrix = DMatrix::from_vec(y.len(), y[0].len(), y.iter().flatten().cloned().collect());


        let x_pseudo_inverse = (X_train_matrix.clone().transpose() * X_train_matrix.clone()).try_inverse().unwrap() * X_train_matrix.clone().transpose();
        let weights_matrix = x_pseudo_inverse * y_matrix;

        self.weights = weights_matrix.row_iter().map(|row| row.iter().cloned().collect()).collect();
    }

    pub fn predict(&self, input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let input_matrix = DMatrix::from_vec(input.len(), input[0].len(), input.iter().flatten().cloned().collect());
        let weights_matrix = DMatrix::from_vec(self.weights.len(), self.weights[0].len(), self.weights.iter().flatten().cloned().collect());

        let predictions = input_matrix * weights_matrix;

        let mut predictions:Vec<Vec<f64>> = predictions.row_iter().map(|row| row.iter().cloned().collect()).collect();
        predictions
    }
}

#[no_mangle]
pub extern "C" fn create_MyLinearModel_Regression() -> *mut MyLinearModel_Regression {
    Box::into_raw(Box::new(MyLinearModel_Regression::new()))
}

#[no_mangle]
pub extern "C" fn destroy_MyLinearModel_Regression(model: *mut MyLinearModel_Regression) {
    if !model.is_null() {
        unsafe {
            Box::from_raw(model);
        }
    }
}

#[no_mangle]
pub extern "C" fn train_MyLinearModel_Regression(
    model: *mut MyLinearModel_Regression,
    X_train_ptr: *const c_double,
    y_train_ptr: *const c_double,
    num_samples: usize,
    n_features: usize,
    n_outputs: usize
) {
    let model = unsafe { &mut *model };

    let X_train = unsafe { 
        slice::from_raw_parts(X_train_ptr, num_samples * n_features)
            .chunks(n_features)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let y_train = unsafe { 
        slice::from_raw_parts(y_train_ptr, num_samples * n_outputs)
            .chunks(n_outputs)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    model.train(X_train, y_train);
}

#[no_mangle]
pub extern "C" fn predict_MyLinearModel_Regression(
    model: *mut MyLinearModel_Regression,
    X_test_ptr: *const c_double,
    num_samples: usize,
    n_features: usize
) -> *mut c_double {
    let model = unsafe { &mut *model };

    let X_test = unsafe { 
        slice::from_raw_parts(X_test_ptr, num_samples * n_features)
            .chunks(n_features)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let predictions = model.predict(X_test);

    let mut flat_predictions = predictions.into_iter().flat_map(|x| x).collect::<Vec<f64>>();
    let predictions_ptr = flat_predictions.as_mut_ptr();
    std::mem::forget(flat_predictions);

    predictions_ptr
}

#[no_mangle]
pub extern "C" fn save_MyLinearModel_Regression(model: *mut MyLinearModel_Regression, path: *const c_char) {
    let model = unsafe { &mut *model };
    let c_str = unsafe {CStr::from_ptr(path)};
    let path_str = c_str.to_str().unwrap();
    let serialized = serde_json::to_string(&model).expect("Failed to serialize model");
    std::fs::write(path_str, serialized).expect("Failed to save model");
}

#[no_mangle]
pub extern "C" fn load_MyLinearModel_Regression(path: *const c_char) -> *mut MyLinearModel_Regression {
    let c_str = unsafe {CStr::from_ptr(path)};
    let path_str = c_str.to_str().unwrap();
    let contents = std::fs::read_to_string(path_str).expect("Failed to read file");
    let model: MyLinearModel_Regression = serde_json::from_str(&contents).expect("Failed to deserialize model");
    Box::into_raw(Box::new(model))
}