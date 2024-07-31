use std::env;
use std::fs;
use std::fs::create_dir_all;
use std::f32::consts::E;
use std::time::{SystemTime, UNIX_EPOCH};
use std::io::{self, Write}; // Ajout de cet import
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::error;
use chrono::prelude::*;

use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::thread_rng;

use crate::activation::*;
use crate::optimizer::*;
use crate::loss::*;
use crate::layers::*;
use crate::utils::*;

use std::slice;
use std::ffi::CStr;
use std::ptr;
use std::os::raw::{c_char, c_double, c_int, c_void};

#[derive(Serialize, Deserialize, Clone)]
pub struct MyCNN {
    pub layers: Vec<Layer>,
    pub layer_order: Vec<String>,
    pub optimizer: OptimizerAlg,
    pub input_shape: (usize, usize, usize),
}

impl MyCNN {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        MyCNN {
            layers: vec![],
            layer_order: vec![],
            optimizer : OptimizerAlg::SGD(0.01),
            input_shape,
        }
    }

    // Ajout de la méthode pour obtenir l'heure et la date actuelles
    fn current_datetime_str() -> String {
        let now: DateTime<Local> = Local::now();
        now.format("%Y-%m-%d_%H-%M-%S").to_string()
    }

    pub fn add_conv_layer(&mut self, num_filters: usize, kernel_size: usize, stride: usize, activation: Activation, seed: Option<u64>) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use set_input_shape() before adding layers.");
        }
        let input_shape: (usize, usize, usize) = match self.layers.last() {
            Some(layer) => match &layer.layer {
                LayerType::Conv(conv_layer) => conv_layer.output_shape,
                LayerType::MaxPooling(maxpooling_layer) => maxpooling_layer.output_shape,
                LayerType::Dense(_) => panic!("Convolutional Layer cannot follow a Dense Layer"),
            },
            None => self.input_shape,
        };
        let conv_layer = Layer::new_conv(input_shape, kernel_size, stride, num_filters, self.optimizer.clone(), activation, seed);
        self.layers.push(conv_layer);
        self.layer_order.push(String::from("conv"));
    }

    pub fn add_maxpooling_layer(&mut self, pool_size: usize) {
        let input_shape: (usize, usize, usize) = match self.layers.last() {
            Some(layer) => match &layer.layer {
                LayerType::Conv(conv_layer) => conv_layer.output_shape,
                LayerType::MaxPooling(maxpooling_layer) => maxpooling_layer.output_shape,
                LayerType::Dense(_) => panic!("Max Pooling Layer cannot follow a Dense Layer"),
            },
            None => self.input_shape,
        };
        let maxpooling_layer = Layer::new_maxpooling(input_shape, pool_size);
        self.layers.push(maxpooling_layer);
        self.layer_order.push(String::from("max pooling"));
    }

    pub fn add_dense_layer(&mut self, output_size: usize, activation: Activation, dropout: f64, seed: Option<u64>) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use set_input_shape() before adding layers.");
        }
        let original_shape: (usize, usize, usize) = match self.layers.last() {
            Some(layer) => match &layer.layer {
                LayerType::Conv(conv_layer) => conv_layer.output_shape,
                LayerType::MaxPooling(maxpooling_layer) => maxpooling_layer.output_shape,
                LayerType::Dense(dense_layer) => (dense_layer.output_size, 1, 1),
            },
            None => self.input_shape,
        };
        let input_size = original_shape.0 * original_shape.1 * original_shape.2;
        let dense_layer = Layer::new_dense(input_size, output_size, activation, self.optimizer.clone(), dropout, original_shape, seed);
        self.layers.push(dense_layer);
        self.layer_order.push(String::from("dense"));
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }
        output[0][0].clone()
    }

    pub fn backward(&mut self, errors: Vec<f64>) {
        let mut output_gradient = vec![vec![errors]];
        for layer in self.layers.iter_mut().rev() {
            output_gradient = layer.backward(output_gradient);
        }
    }

    pub fn update(&mut self, batch_size:usize) {
        for layer in self.layers.iter_mut() {
            layer.update(batch_size, self.optimizer.clone());
        }
    }

    pub fn predict(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        self.forward(input)
    }

    pub fn set_optimizer(&mut self, optimizer:OptimizerAlg){
        self.optimizer = optimizer
    }

    pub fn train(&mut self, 
        X_train: Vec<Vec<Vec<Vec<f64>>>>, 
        y_train: Vec<Vec<f64>>, 
        X_test: Vec<Vec<Vec<Vec<f64>>>>, 
        y_test: Vec<Vec<f64>>, 
        epochs: usize,
        batch_size:usize) {

        let datetime_str = current_datetime_str(); // Récupérer la date et l'heure actuelles
        let base_dir = "../saved_models/CNN_models";
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

        for epoch in 0..epochs {
            let start_time = SystemTime::now();
            let total_batches = (X_train.len() / batch_size) as u64;
            let bar_width = 50;

            let mut avg_train_acc = 0.0;
            let mut avg_train_loss = 0.0;

            let pb = ProgressBar::new(total_batches);
            pb.set_style(ProgressStyle::default_bar()
                .template(&format!("Epoch {}: [{{bar:.cyan/blue}}] {{pos}}/{{len}} - ETA: {{eta}} - acc: {{msg}}", epoch))
                .unwrap()
                .progress_chars("#>-"));
            
            for i in 0..X_train.len() {
                let image = &X_train[i];
                let label = y_train[i].clone();

                let output = self.forward(image.clone());

                let mut loss = 0.;
                for i in 0..label.len(){
                    loss += (output[i] - label[i]).powf(2.);
                }
                avg_train_loss += loss;

                let mut errors = vec![0.; output.len()];
                for i in 0..output.len() {
                    errors[i] = output[i].clone() - label[i]
                }
                
                self.backward(errors);

                avg_train_acc += (argmax(&output) == argmax(&label)) as i32 as f64;

                if i % batch_size == batch_size - 1 {
                    self.update(batch_size);
                    pb.inc(1);
                    pb.set_message(format!("Train acc {:.1}% - Train loss {:.4}", avg_train_acc / (i + 1) as f64 * 100.0, avg_train_loss / (i + 1) as f64));
                    self.update(batch_size);
                    // let progress = ((i / batch_size) as f64 / total_batches as f64 * bar_width as f64) as usize;
                    // print!("\rEpoch {}: [{}>{}] {}% - Train acc {:.1}% - Train loss {:.4}",
                    //         epoch,
                    //         "=".repeat(progress),
                    //         " ".repeat(bar_width - progress),
                    //         (i / batch_size + 1) * 100 / total_batches as usize,
                    //         avg_train_acc / (i + 1) as f64 * 100.0,
                    //         avg_train_loss / (i + 1) as f64);
                    // io::stdout().flush().unwrap();
                }
            }
        
            avg_train_loss /= X_train.len() as f64;
            avg_train_acc /= X_train.len() as f64;
            pb.set_message(format!("{:.1}% - Testing...", avg_train_acc));
            // println!("\rEpoch {}: [{}] 100% - Train acc {:.1}% - Train loss {:.4} - Testing...",
            //         epoch,
            //         "=".repeat(bar_width),
            //         avg_train_acc * 100.0,
            //         avg_train_loss);

            let mut avg_test_acc = 0.0;
            let mut avg_test_loss = 0.0;
            for i in 0..X_test.len() {
                let image = &X_test[i];
                let label = y_test[i].clone();
                let output = self.forward(image.clone());

                let mut loss = 0.;
                for i in 0..label.len(){
                    loss += (output[i] - label[i]).powf(2.);
                }
                avg_test_loss += loss;
                avg_test_acc += (argmax(&output) == argmax(&label)) as i32 as f64;
            }

            avg_test_loss /= X_test.len() as f64;
            avg_test_acc /= X_test.len() as f64;
            pb.finish_with_message(format!("{:.1}% - Test: {:.1}%", avg_train_acc * 100.0, avg_test_acc * 100.0));
            // println!("Epoch {}: Train acc {:.1}% - Test acc {:.1}% - Test loss {:.4}",
            //         epoch,
            //         avg_train_acc * 100.0,
            //         avg_test_acc * 100.0,
            //         avg_test_loss);

            let epoch_duration = SystemTime::now().duration_since(start_time).unwrap().as_secs() as usize;

            // MyCNN::save(self, &format!("{}/model_epoch_{}.json", epoch_dir, epoch)).expect("Failed to save model"); // Sauvegarder le modèle à chaque époque

            metrics["train_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_train_acc));
            metrics["train_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_train_loss));
            metrics["test_accuracies"].as_array_mut().unwrap().push(serde_json::json!(avg_test_acc));
            metrics["test_losses"].as_array_mut().unwrap().push(serde_json::json!(avg_test_loss));
            metrics["epoch_times"].as_array_mut().unwrap().push(serde_json::json!(epoch_duration));
            fs::write(&metrics_file_path, serde_json::to_string_pretty(&metrics).expect("Failed to serialize metrics")).expect("Failed to save metrics");

            if avg_test_acc > best_test_acc {
                best_test_acc = avg_test_acc;
                MyCNN::save(self,&format!("{}/best_model.json", best_model_dir)).expect("Failed to save best model"); // Sauvegarder le meilleur modèle
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string_pretty(self)?;
        fs::write(path, serialized)?;
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<MyCNN, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let model: MyCNN = serde_json::from_str(&contents)?;
        Ok(model)
    }
}

#[no_mangle]
pub extern "C" fn create_MyCNN(input_shape_0: usize, input_shape_1: usize, input_shape_2: usize) -> *mut MyCNN {
    let input_shape = (input_shape_0, input_shape_1, input_shape_2);
    let cnn = MyCNN::new(input_shape);
    let boxed_cnn = Box::new(cnn);
    Box::into_raw(boxed_cnn)
}

#[no_mangle]
pub extern "C" fn destroy_MyCNN(cnn: *mut MyCNN) {
    if !cnn.is_null() {
        unsafe {
            Box::from_raw(cnn);
        }
    }
}

#[no_mangle]
pub extern "C" fn add_conv_layer_MyCNN(cnn: *mut MyCNN, num_filters: usize, kernel_size: usize, stride: usize, activation: *const c_char, seed: u64) {
    let cnn = unsafe { &mut *cnn };

    let c_str = unsafe { CStr::from_ptr(activation) };
    let activation_str = c_str.to_str().unwrap();

    let activation = match activation_str {
        "ReLU" => Activation::ReLU,
        "Sigmoid" => Activation::Sigmoid,
        "Tanh" => Activation::Tanh,
        _ => panic!("Unknown activation function"),
    };

    let seed_option = if seed == 0 { None } else { Some(seed) };
    cnn.add_conv_layer(num_filters, kernel_size, stride, activation, seed_option);
}


#[no_mangle]
pub extern "C" fn add_maxpooling_layer_MyCNN(cnn: *mut MyCNN, pool_size: usize) {
    let cnn = unsafe { &mut *cnn };
    cnn.add_maxpooling_layer(pool_size);
}

#[no_mangle]
pub extern "C" fn add_dense_layer_MyCNN(cnn: *mut MyCNN, output_size: usize, activation: *const c_char, dropout: f64, seed: u64) {
    let cnn = unsafe { &mut *cnn };

    let c_str = unsafe { CStr::from_ptr(activation) };
    let activation_str = c_str.to_str().unwrap();

    let activation = match activation_str {
        "ReLU" => Activation::ReLU,
        "Sigmoid" => Activation::Sigmoid,
        "Tanh" => Activation::Tanh,
        _ => panic!("Unknown activation function"),
    };

    let seed_option = if seed == 0 { None } else { Some(seed) };
    cnn.add_dense_layer(output_size, activation, dropout, seed_option);
}

#[no_mangle]
pub extern "C" fn set_optimizer_SGD(cnn: *mut MyCNN, learning_rate: f64) {
    let cnn = unsafe { &mut *cnn };
    cnn.set_optimizer(OptimizerAlg::SGD(learning_rate))
}

#[no_mangle]
pub extern "C" fn train_MyCNN(
    cnn: *mut MyCNN,
    X_train_ptr: *const c_double,
    y_train_ptr: *const c_double,
    X_test_ptr: *const c_double,
    y_test_ptr: *const c_double,
    num_train_samples: usize,
    num_test_samples: usize,
    input_dim_0: usize,
    input_dim_1: usize,
    input_dim_2: usize,
    output_size: usize,
    epochs: usize,
    batch_size: usize,
    seed: u64,
) {
    let cnn = unsafe { &mut *cnn };

    let X_train = unsafe { 
        slice::from_raw_parts(X_train_ptr, num_train_samples * input_dim_0 * input_dim_1 * input_dim_2)
            .chunks(input_dim_0 * input_dim_1 * input_dim_2)
            .map(|x| x.chunks(input_dim_1 * input_dim_2).map(|y| y.chunks(input_dim_2).map(|z| z.to_vec()).collect()).collect())
            .collect::<Vec<Vec<Vec<Vec<f64>>>>>()
    };

    let y_train = unsafe { slice::from_raw_parts(y_train_ptr, num_train_samples * output_size) }
        .chunks(output_size)
        .map(|x| x.to_vec())
        .collect::<Vec<Vec<f64>>>();

    let X_test = unsafe { 
        slice::from_raw_parts(X_test_ptr, num_test_samples * input_dim_0 * input_dim_1 * input_dim_2)
            .chunks(input_dim_0 * input_dim_1 * input_dim_2)
            .map(|x| x.chunks(input_dim_1 * input_dim_2).map(|y| y.chunks(input_dim_2).map(|z| z.to_vec()).collect()).collect())
            .collect::<Vec<Vec<Vec<Vec<f64>>>>>()
    };

    let y_test = unsafe { slice::from_raw_parts(y_test_ptr, num_test_samples * output_size) }
        .chunks(output_size)
        .map(|x| x.to_vec())
        .collect::<Vec<Vec<f64>>>();
    
    cnn.train(X_train, y_train, X_test, y_test, epochs, batch_size);
}

#[no_mangle]
pub extern "C" fn predict_MyCNN(
    cnn: *mut MyCNN,
    input_ptr: *const c_double,
    num_samples: usize,
    input_dim_0: usize,
    input_dim_1: usize,
    input_dim_2: usize
) -> *mut c_double {
    let cnn = unsafe { &mut *cnn };

    let inputs = unsafe { 
        slice::from_raw_parts(input_ptr, num_samples * input_dim_0 * input_dim_1 * input_dim_2)
            .chunks(input_dim_0 * input_dim_1 * input_dim_2)
            .map(|x| x.chunks(input_dim_1 * input_dim_2).map(|y| y.chunks(input_dim_2).map(|z| z.to_vec()).collect()).collect())
            .collect::<Vec<Vec<Vec<Vec<f64>>>>>()
    };

    // print_image(&inputs[0]);

    let mut predictions = Vec::new();
    for input in inputs {
        let pred = cnn.predict(input);
        let argmax_pred = argmax(&pred);

        // Imprimer le résultat en fonction de l'argmax
        match argmax_pred {
            0 => println!("Happy"),
            1 => println!("Neutral"),
            2 => println!("Sad"),
            _ => println!("Unknown")
        }
         // Effacer l'affichage après avoir imprimé les prédictions
        print!("\x1B[2J\x1B[1;1H"); // Séquence d'échappement ANSI pour effacer l'écran et repositionner le curseur
        predictions.extend(pred);
    }

    let predictions_ptr = predictions.as_mut_ptr();
    std::mem::forget(predictions);
    predictions_ptr
}

#[no_mangle]
pub extern "C" fn save_MyCNN(mlp: *mut MyCNN, path: *const c_char) -> c_int {
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
pub extern "C" fn load_MyCNN(path: *const c_char) -> *mut MyCNN {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match MyCNN::load(path_str) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}