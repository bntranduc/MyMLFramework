use rand::{Rng, SeedableRng, rngs::StdRng};
use rand::seq::SliceRandom;
use std::fs::{self, create_dir_all};
use std::time::{SystemTime, UNIX_EPOCH};
use std::slice;
use std::ffi::CStr;
use std::ptr;
use std::os::raw::{c_char, c_double, c_int, c_void};
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};

use crate::utils::*;

#[derive(Serialize, Deserialize)]
pub struct MyRBF {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    centers: Vec<Vec<f64>>,
    y_dim: usize,
    gamma: f64,
}

impl MyRBF {
    pub fn new(centers: Vec<Vec<f64>>, y_dim: usize, gamma: f64, seed: Option<u64>) -> Self {
        let (n_centers, n_features) = (centers.len(), centers[0].len());
        let mut rng: StdRng = match seed {
            Some(s) => SeedableRng::seed_from_u64(s),
            None => SeedableRng::from_entropy(),
        };
        
        let mut weights = vec![vec![0.0; n_centers]; y_dim];
        let mut biases = vec![0.0; y_dim];
        
        for i in 0..y_dim {
            for j in 0..n_centers {
                weights[i][j] = rng.gen_range(-1.0..=1.0);
            }
            biases[i] = rng.gen_range(-1.0..=1.0);
        }

        MyRBF {
            weights,
            biases,
            centers,
            y_dim,
            gamma,
        }
    }

    fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, 
                            y_train: Vec<Vec<f64>>, 
                            X_test: Vec<Vec<f64>>, 
                            y_test: Vec<Vec<f64>>, 
                            learning_rate: f64, 
                            epochs: usize, 
                            is_classification: bool) {

        println!("LOOOL");
        let datetime_str = current_datetime_str();
        let base_dir = "../saved_models/RBF_models";
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

        let n_centers = self.centers.len();

        let mut best_test_acc = 0.0;

        println!("Training starting...");

        let X_train_size = X_train.len();
        let X_test_size = X_test.len();

        for e in 0..epochs {
            let start_time = SystemTime::now();

            let mut avg_train_acc = 0.0;
            let mut avg_train_loss = 0.0;

            for k in 0..X_train.len() {
                let label = &y_train[k];

                let mut hidden_output = vec![0.0; n_centers];
                for c in 0..n_centers {
                    hidden_output[c] = gauss_kernel(&X_train[k], &self.centers[c], self.gamma);
                }

                let mut pred = vec![0.0; self.y_dim];
                for i in 0..self.y_dim {
                    for j in 0..n_centers {
                        pred[i] += self.weights[i][j] * hidden_output[j];
                    }
                    pred[i] += self.biases[i];
                }

                if is_classification {
                    for i in 0..self.y_dim {
                        pred[i] = if pred[i] < 0.0 { -1.0 } else { 1.0 };
                    }
                }

                let mut error = vec![0.0; self.y_dim];
                for i in 0..self.y_dim {
                    error[i] = label[i] - pred[i];
                }

                for i in 0..self.y_dim {
                    for j in 0..n_centers {
                        self.weights[i][j] += learning_rate * error[i] * hidden_output[j];
                    }
                    self.biases[i] += learning_rate * error[i];
                }

                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (pred[j] - label[j]).powf(2.);
                }
                avg_train_loss += loss;
                avg_train_acc += (argmax(&pred) == argmax(&label)) as i32 as f64;
            }

            avg_train_loss /= X_train_size as f64;
            avg_train_acc /= X_train_size as f64;

            let mut avg_test_acc = 0.0;
            let mut avg_test_loss = 0.0;
            for k in 0..X_test.len() {
                let label = &y_test[k];

                let mut hidden_output = vec![0.0; n_centers];
                for c in 0..n_centers {
                    hidden_output[c] = gauss_kernel(&X_train[k], &self.centers[c], self.gamma);
                }

                let mut pred = vec![0.0; self.y_dim];
                for i in 0..self.y_dim {
                    for j in 0..n_centers {
                        pred[i] += self.weights[i][j] * hidden_output[j];
                    }
                    pred[i] += self.biases[i];
                }

                if is_classification {
                    for i in 0..self.y_dim {
                        pred[i] = if pred[i] < 0.0 { -1.0 } else { 1.0 };
                    }
                }

                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (pred[j] - label[j]).powf(2.);
                }
                avg_test_loss += loss;

                let acc = (argmax(&pred) == argmax(label)) as i32;
                avg_test_acc += (argmax(&pred) == argmax(&label)) as i32 as f64;
            }
            avg_test_acc /= X_test_size as f64;
            avg_test_loss /= X_test_size as f64;

            println!("Epoch {}: Train Loss: {:.4}, Test Loss: {:.4}, Train Acc: {:.1}%, Test Acc: {:.1}%", 
                    e, 
                    avg_train_loss, 
                    avg_test_loss, 
                    avg_train_acc * 100., 
                    avg_test_acc * 100.
                    );

            let epoch_duration = SystemTime::now().duration_since(start_time).unwrap().as_secs() as usize;

            // self.save(&format!("{}/model_epoch_{}.json", epoch_dir, epoch)).expect("Failed to save model");

            if avg_test_acc > best_test_acc {
                best_test_acc = avg_test_acc;
                self.save(&format!("{}/best_model.json", best_model_dir)).expect("Failed to save best model");
            }

            metrics["train_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_train_acc));
            metrics["train_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_train_loss));
            metrics["test_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_test_acc));
            metrics["test_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_test_loss));
            metrics["epoch_times"].as_array_mut().unwrap().push(serde_json::json!(epoch_duration));
            fs::write(&metrics_file_path, serde_json::to_string_pretty(&metrics).expect("Failed to serialize metrics")).expect("Failed to save metrics");
        }
    }

    pub fn predict(&self, input: Vec<Vec<f64>>, is_classification: bool) -> Vec<Vec<f64>> {
        let n_centers = self.centers.len();
        let mut predictions = vec![];

        for k in 0..input.len() {
            let mut hidden_output = vec![0.0; n_centers];
            for c in 0..n_centers {
                hidden_output[c] = gauss_kernel(&input[k], &self.centers[c], self.gamma);
            }

            let mut pred = vec![0.0; self.y_dim];
            for i in 0..self.y_dim {
                for j in 0..n_centers {
                    pred[i] += self.weights[i][j] * hidden_output[j];
                }
                pred[i] += self.biases[i];
            }

            if is_classification {
                for i in 0..self.y_dim {
                    pred[i] = if pred[i] < 0.0 { -1.0 } else { 1.0 };
                }
            }
            predictions.push(pred);
        }
        predictions
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string_pretty(self)?;
        fs::write(path, serialized)?;
        Ok(())
    }
    
    fn load(path: &str) -> Result<MyRBF, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let model: MyRBF = serde_json::from_str(&contents)?;
        Ok(model)
    }
}

#[no_mangle]
pub extern "C" fn create_MyRBF(
    centers_ptr: *const c_double,
    num_centers: usize,
    input_dim: usize,
    y_dim: usize,
    gamma: c_double,
    seed: u64,
) -> *mut MyRBF {
    let centers = unsafe { 
        slice::from_raw_parts(centers_ptr, num_centers * input_dim)
            .chunks(input_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };
    let seed_option = if seed == 0 { None } else { Some(seed) };
    let rbf = MyRBF::new(centers, y_dim, gamma, seed_option);
    let boxed_rbf = Box::new(rbf);
    Box::into_raw(boxed_rbf)
}

#[no_mangle]
pub extern "C" fn destroy_MyRBF(rbf: *mut MyRBF) {
    if !rbf.is_null() {
        unsafe {
            Box::from_raw(rbf);
        }
    }
}

#[no_mangle]
pub extern "C" fn train_MyRBF(
    rbf: *mut MyRBF,
    X_train_ptr: *const c_double,
    y_train_ptr: *const c_double,
    X_test_ptr: *const c_double,
    y_test_ptr: *const c_double,
    num_train_samples: usize,
    num_test_samples: usize,
    input_dim: usize,
    output_dim: usize,
    learning_rate: c_double,
    epochs: usize,
    is_classification: bool,
) {
    let rbf = unsafe { &mut *rbf };

    let X_train = unsafe { 
        slice::from_raw_parts(X_train_ptr, num_train_samples * input_dim)
            .chunks(input_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let y_train = unsafe { 
        slice::from_raw_parts(y_train_ptr, num_train_samples * output_dim)
            .chunks(output_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let X_test = unsafe { 
        slice::from_raw_parts(X_test_ptr, num_test_samples * input_dim)
            .chunks(input_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let y_test = unsafe { 
        slice::from_raw_parts(y_test_ptr, num_test_samples * output_dim)
            .chunks(output_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    rbf.train(X_train, y_train, X_test, y_test, learning_rate, epochs, is_classification);
}

#[no_mangle]
pub extern "C" fn predict_MyRBF(
    rbf: *mut MyRBF,
    input_ptr: *const c_double,
    num_samples: usize,
    input_dim: usize,
    is_classification: bool
) -> *mut c_double {
    let rbf = unsafe { &mut *rbf };

    let inputs = unsafe { 
        slice::from_raw_parts(input_ptr, num_samples * input_dim)
            .chunks(input_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let predictions = rbf.predict(inputs, is_classification);

    let mut flat_predictions = Vec::new();
    for prediction in predictions {
        flat_predictions.extend(prediction);
    }

    let predictions_ptr = flat_predictions.as_mut_ptr();
    std::mem::forget(flat_predictions);

    predictions_ptr
}

#[no_mangle]
pub extern "C" fn save_MyRBF(rbf: *mut MyRBF, path: *const c_char) -> c_int {
    let rbf = unsafe { &mut *rbf };

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match rbf.save(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn load_MyRBF(path: *const c_char) -> *mut MyRBF {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match MyRBF::load(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}