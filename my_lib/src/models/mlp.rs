extern crate rand;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};
use std::io::{self, Write};
use std::fs::{self, create_dir_all};
use std::slice;
use std::ffi::CStr;
use std::ptr;
use std::os::raw::{c_char, c_double, c_int, c_void};
use crate::utils::*;

#[derive(Serialize, Deserialize)]
pub struct MyMLP {
    d: Vec<usize>,
    W: Vec<Vec<Vec<f64>>>,
    L: usize,
    X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MyMLP {
    pub fn new(npl: Vec<usize>, seed: Option<u64>) -> MyMLP {
        let L = npl.len() - 1;
        let mut W = vec![vec![vec![]; 0]; L + 1];
        let mut rng: StdRng = match seed {
            Some(s) => SeedableRng::seed_from_u64(s),
            None => SeedableRng::from_entropy(),
        };

        for l in 0..=L {
            W[l] = vec![vec![0.0; npl[l] + 1]; if l == 0 { 0 } else { npl[l - 1] + 1 }];
            if l == 0 {
                continue;
            }
            for i in 0..=npl[l - 1] {
                for j in 1..=npl[l] {
                    W[l][i][j] = rng.gen_range(-1.0..1.0);
                }
            }
        }

        let mut X = vec![vec![0.0; 0]; L + 1];
        let mut deltas = vec![vec![0.0; 0]; L + 1];
        for l in 0..=L {
            X[l] = vec![0.0; npl[l] + 1];
            deltas[l] = vec![0.0; npl[l] + 1];
            X[l][0] = 1.0;
        }

        MyMLP { d: npl, W, L, X, deltas }
    }

    pub fn propagate(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) {
        for j in 0..sample_inputs.len() {
            self.X[0][j + 1] = sample_inputs[j];
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l] {
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][i][j] * self.X[l - 1][i];
                }
                if is_classification || l < self.L {
                    total = total.tanh();
                }
                self.X[l][j] = total;
            }
        }
    }

    pub fn predict(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification);
        self.X[self.L][1..].to_vec()
    }

    pub fn summary(&self) {
        println!("Model Summary:");
        println!("Number of layers: {}", self.L + 1);
        for (i, size) in self.d.iter().enumerate() {
            println!("Layer {}: {} neurons", i, size);
        }
    }

    pub fn train(&mut self, 
                 X_train: &Vec<Vec<f64>>, 
                 y_train: &Vec<Vec<f64>>, 
                 X_test: &Vec<Vec<f64>>, 
                 y_test: &Vec<Vec<f64>>, 
                 alpha: f64, 
                 epochs: usize, 
                 is_classification: bool) {

        let datetime_str = current_datetime_str();
        let base_dir = "../saved_models/MLP_models";
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

        let mut best_test_acc = 0.0;

        println!("Training starting...");

        let X_train_size = X_train.len();
        let X_test_size = X_test.len();

        for epoch in 0..epochs {
            let start_time = SystemTime::now();

            let mut avg_train_acc = 0.0;
            let mut avg_train_loss = 0.0;

            for k in 0..X_train_size {
                let sample_inputs = &X_train[k];
                let label = &y_train[k];

                self.propagate(sample_inputs, is_classification);
                let output = self.X[self.L][1..].to_vec();

                for j in 1..=self.d[self.L] {
                    self.deltas[self.L][j] = self.X[self.L][j] - label[j - 1];
                    if is_classification {
                        self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                    }
                }

                for l in (2..=self.L).rev() {
                    for i in 1..=self.d[l - 1] {
                        let mut total = 0.0;
                        for j in 1..=self.d[l] {
                            total += self.W[l][i][j] * self.deltas[l][j];
                        }
                        total *= 1.0 - self.X[l - 1][i].powi(2);
                        self.deltas[l - 1][i] = total;
                    }
                }

                for l in 1..=self.L {
                    for i in 0..=self.d[l - 1] {
                        for j in 1..=self.d[l] {
                            self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                        }
                    }
                }

                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (output[j] - label[j]).powf(2.);
                }
                avg_train_loss += loss;
                avg_train_acc += (argmax(&output) == argmax(&label)) as i32 as f64;
            }

            avg_train_loss /= X_train_size as f64;
            avg_train_acc /= X_train_size as f64;

            let mut avg_test_acc = 0.0;
            let mut avg_test_loss = 0.0;
            for k in 0..X_test.len() {
                let sample_inputs = &X_test[k];
                let label = &y_test[k]; 

                self.propagate(sample_inputs, is_classification);

                for j in 1..(self.d[self.L] as usize) + 1 {
                    let error = self.X[self.L][j] - label[j - 1];
                }

                let output = self.X[self.L][1..].to_vec();

                let mut loss = 0.;
                for j in 0..label.len(){
                    loss += (output[j] - label[j]).powf(2.);
                }
                avg_test_loss += loss;

                let acc = (argmax(&output) == argmax(label)) as i32;
                avg_test_acc += (argmax(&output) == argmax(&label)) as i32 as f64;
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

            let epoch_duration = SystemTime::now().duration_since(start_time).unwrap().as_secs() as usize;

            // self.save(&format!("{}/model_epoch_{}.json", epoch_dir, epoch)).expect("Failed to save model");

            // if avg_test_acc > best_test_acc {
            //     best_test_acc = avg_test_acc;
            //     self.save(&format!("{}/best_model.json", best_model_dir)).expect("Failed to save best model");
            // }

            metrics["train_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_train_acc));
            metrics["train_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_train_loss));
            metrics["test_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_test_acc));
            metrics["test_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_test_loss));
            metrics["epoch_times"].as_array_mut().unwrap().push(serde_json::json!(epoch_duration));
            fs::write(&metrics_file_path, serde_json::to_string_pretty(&metrics).expect("Failed to serialize metrics")).expect("Failed to save metrics");
        }
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let serialized = serde_json::to_string(&self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("Serialization Error: {}", e))
        })?;
        std::fs::write(path, serialized)
    }

    
    fn load(path: &str) -> Result<MyMLP, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let model: MyMLP = serde_json::from_str(&contents)?;
        Ok(model)
    }
}

#[no_mangle]
pub extern "C" fn create_MyMLP(layer_sizes_ptr: *const c_int, num_layers: c_int, seed: u64) -> *mut MyMLP {
    let layer_sizes = unsafe { slice::from_raw_parts(layer_sizes_ptr, num_layers as usize) };
    let npl: Vec<usize> = layer_sizes.iter().map(|&size| size as usize).collect();
    let seed_option = if seed == 0 { None } else { Some(seed) };
    let mlp = MyMLP::new(npl, seed_option);
    let boxed_mlp = Box::new(mlp);
    Box::into_raw(boxed_mlp)
}

#[no_mangle]
pub extern "C" fn destroy_MyMLP(mlp: *mut MyMLP) {
    if !mlp.is_null() {
        unsafe {
            Box::from_raw(mlp);
        }
    }
}

#[no_mangle]
pub extern "C" fn train_MyMLP(
    mlp: *mut MyMLP,
    X_train_ptr: *const c_double,
    y_train_ptr: *const c_double,
    X_test_ptr: *const c_double,
    y_test_ptr: *const c_double,
    num_train_samples: usize,
    num_test_samples: usize,
    input_dim: usize,
    output_dim: usize,
    alpha: c_double,
    epochs: usize,
    is_classification: bool,
) {
    let mlp = unsafe { &mut *mlp };

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

    mlp.train(&X_train, &y_train, &X_test, &y_test, alpha, epochs, is_classification);
}

#[no_mangle]
pub extern "C" fn predict_MyMLP(
    mlp: *mut MyMLP,
    input_ptr: *const c_double,
    num_samples: usize,
    input_dim: usize,
    is_classification: bool
) -> *mut c_double {
    let mlp = unsafe { &mut *mlp };

    let inputs = unsafe { 
        slice::from_raw_parts(input_ptr, num_samples * input_dim)
            .chunks(input_dim)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f64>>>()
    };

    let mut predictions = Vec::new();
    for input in inputs {
        let pred = mlp.predict(&input, is_classification);
        predictions.extend(pred);
    }

    let predictions_ptr = predictions.as_mut_ptr();
    std::mem::forget(predictions); // Prevent Rust from freeing the memory

    predictions_ptr
}

#[no_mangle]
pub extern "C" fn save_MyMLP(mlp: *mut MyMLP, path: *const c_char) -> c_int {
    let mlp = unsafe { &mut *mlp };

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match mlp.save(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn load_MyMLP(path: *const c_char) -> *mut MyMLP {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match MyMLP::load(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn summary_MyMLP(mlp: *mut MyMLP) {
    let mlp = unsafe { &mut *mlp };
    mlp.summary();
}
