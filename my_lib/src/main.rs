use std::path::{Path, PathBuf};

mod utils;
use mlp::MyMLP;

use crate::utils::*;

mod layers;
use crate::layers::*;

mod models;
use crate::models::*;

mod activation;
use crate::activation::*;

mod optimizer;
use crate::optimizer::*;

mod loss;
use crate::loss::*;

mod data;
use crate::data::*;

// fn main(){
//     let pred = vec![0.1, 0.6 ,0.8];
//     let label = vec![0., 0., 1.];

//     let acc = (argmax(&pred) == argmax(&label)) as i32;

//     println!("acc = {:?}", acc);
// }

// fn main() {
//     let seed = Some(42);

//     let dataset_path = Path::new("../dataset_upgrade/");
//     let (X_train, y_train, X_test, y_test) = load_dataset(dataset_path);

//     println!("Loaded {} training images", X_train.len());
//     println!("Loaded {} test images", X_test.len());
//     let (X_train, y_train, X_test, y_test) = shuffle_datasets(X_train, y_train, X_test, y_test, seed);

//     print_image(&X_train[0]);
//     println!("label = {:?}", y_train[0]);

//     let mut model = MyCNN::new((48,48,3));

//     model.add_conv_layer(32, 3, 1, Activation::Tanh, seed);
//     model.add_maxpooling_layer(2);

//     model.add_dense_layer(32, Activation::Tanh, 0.2, seed); 
//     model.add_dense_layer(3, Activation::Tanh, 0., seed);

//     model.train(X_train.clone(), y_train.clone(), X_test.clone(), y_test.clone(), 1000, 10);
//     // let mut model = MyCNN::load("../saved_models/CNN_models/CNN_76/best_model/best_model.json").unwrap();

//     // let path_str = String::from("img.png");
//     // let path = Path::new(&path_str);
//     // let img = load_image(path).unwrap();
//     // println!("{:?}", (img.len(), img[0].len(), img[0][0].len()));
    
//     // print_image(&img);

//     // let pred = model.predict(img);
//     // println!("pred = {:?}", pred);
// }

// fn main() {
//     let seed = Some(42);

//     let dataset_path = Path::new("../dataset_upgrade/");
//     let (X_train, y_train, X_test, y_test) = load_dataset(dataset_path);

//     println!("Loaded {} training images", X_train.len());
//     println!("Loaded {} test images", X_test.len());
//     let (X_train, y_train, X_test, y_test) = shuffle_datasets(X_train, y_train, X_test, y_test, seed);

//     print_image(&X_train[0]);
//     println!("label = {:?}", y_train[0]);

//     let (X_train, X_test) = (flatten_images(X_train), flatten_images(X_test));

//     let img = X_train[0].clone();
//     println!("flatten img shape = {:?}", img.len());

//     let input_size = img.len();

//     let mut mlp = MyNewMLP::new(input_size);
//     mlp.compile(OptimizerAlg::SGD(0.001), Loss::MSE);

//     // Ajouter des couches
//     mlp.add_dense_layer(16, Activation::Tanh, 0.0, seed);
//     mlp.add_dense_layer(32, Activation::Tanh, 0.0, seed);
//     mlp.add_dense_layer(3, Activation::Tanh, 0.0, seed);

//     let input = X_train[0].clone();
//     // mlp.forward(input, true);

//     // // Entraîner le modèle
//     mlp.train(X_train.clone(), y_train.clone(), X_test.clone(), y_test.clone(), 1000, 1);

//     // let mut model = MyNewMLP::load("mlp.json").unwrap();

//     // let path_str = String::from("neutral.png");
//     // let path = Path::new(&path_str);
//     // let img = load_image(path).unwrap();
//     // println!("{:?}", (img.len(), img[0].len(), img[0][0].len()));
    
//     // print_image(&img);
//     // let flatten_img = flatten_images(vec![img])[0].clone();

//     // let pred = model.predict(flatten_img);
//     // println!("pred = {:?}", pred);
// }

// fn main(){
//     let seed = Some(42);

//     let dataset_path = Path::new("../dataset_upgrade/");
//     let (X_train, y_train, X_test, y_test) = load_dataset(dataset_path);

//     println!("Loaded {} training images", X_train.len());
//     println!("Loaded {} test images", X_test.len());
//     let (X_train, y_train, X_test, y_test) = shuffle_datasets(X_train, y_train, X_test, y_test, seed);

//     print_image(&X_train[0]);
//     println!("label = {:?}", y_train[0]);

//     let (X_train, X_test) = (flatten_images(X_train), flatten_images(X_test));

//     let img = X_train[0].clone();
//     println!("flatten img shape = {:?}", img.len());

//     let input_size = img.len();
//     let mut model = MyLinearModel::new(input_size, 3, seed);
//     model.train(X_train, y_train, X_test, y_test, 0.001, 1000, true);

//     // let mut model = MyLinearModel::load("linear.json").unwrap();

//     // let path_str = String::from("neutral.png");
//     // let path = Path::new(&path_str);
//     // let img = load_image(path).unwrap();
//     // println!("{:?}", (img.len(), img[0].len(), img[0][0].len()));
    
//     // print_image(&img);

//     // let pred = model.predict(flatten_images(vec![img]), true);
//     // println!("pred = {:?}", pred);
// }

// fn main() {
//     let seed = Some(42);

//     let dataset_path = Path::new("../dataset_upgrade/");
//     let (X_train, y_train, X_test, y_test) = load_dataset(dataset_path);

//     println!("Loaded {} training images", X_train.len());
//     println!("Loaded {} test images", X_test.len());
//     let (X_train, y_train, X_test, y_test) = shuffle_datasets(X_train, y_train, X_test, y_test, seed);

//     print_image(&X_train[0]);
//     println!("label = {:?}", y_train[0]);

//     let mut X_train_flatten = vec![];
//     for i in 0..X_train.len(){
//         X_train_flatten.push(flatten(X_train[i].clone()));
//     }

//     let mut X_test_flatten = vec![];
//     for i in 0..X_test.len(){
//         X_test_flatten.push(flatten(X_test[i].clone()));
//     }

//     let input_size = X_train[0].len() * X_train[0][0].len() * X_train[0][0][0].len();

//     // Définir l'architecture du MLP (3 couches : input, hidden, output)
//     let npl = vec![input_size, 4, 3];
    
//     // Créer le modèle MLP avec une graine spécifique pour la reproductibilité
//     let mut mlp = MyMLP::new(npl, seed);

//     // Utiliser les mêmes données pour le test (pour simplifier l'exemple)
//     let X_test = X_train.clone();
//     let y_test = y_train.clone();

//     // Paramètres d'entraînement
//     let alpha = 0.001;
//     let nb_iter = 10_000;
//     let is_classification = true;

//     // Entraîner le modèle
//     mlp.train(&X_train_flatten, &y_train, &X_test_flatten, &y_test, alpha, nb_iter, is_classification);
// }


// fn main() {
//     // Définir les paramètres pour MyRBF
//     let centers = vec![
//         vec![0.0, 0.0], 
//         vec![0.0, 1.0], 
//         vec![1.0, 0.0], 
//         vec![1.0, 1.0]
//     ]; // Centres pour le XOR
//     let y_dim = 1; // Dimension de sortie
//     let gamma = 1.0; // Exemple de valeur gamma
//     let seed = Some(42); // Optionnel, pour la reproductibilité

//     // Créer une instance de MyRBF
//     let mut rbf = MyRBF::new(centers, y_dim, gamma, seed);

//     // Définir les ensembles d'entraînement et de test pour le XOR
//     let X_train = vec![
//         vec![0.0, 0.0], 
//         vec![0.0, 1.0], 
//         vec![1.0, 0.0], 
//         vec![1.0, 1.0]
//     ]; // Entrées du XOR
//     let y_train = vec![
//         vec![-1.0], 
//         vec![1.0], 
//         vec![1.0], 
//         vec![-1.0]
//     ]; // Sorties du XOR
//     let X_test = X_train.clone(); // Utiliser les mêmes données pour le test
//     let y_test = y_train.clone(); // Utiliser les mêmes étiquettes pour le test

//     // Entraîner le modèle
//     let learning_rate = 0.01; // Exemple de taux d'apprentissage
//     let epochs = 1000; // Exemple de nombre d'époques
//     let is_classification = true; // Tâche de classification

//     rbf.train(X_train.clone(), y_train.clone(), X_test.clone(), y_test.clone(), learning_rate, epochs, is_classification);

//     // Faire des prédictions
//     let predictions = rbf.predict(X_test.clone(), is_classification);
//     println!("Predictions: {:?}", predictions);
// }


// use std::fs::File;
// use std::io::Write;

// // Fonction de convolution
// // Fonction de convolution
// fn convolve(image: &Vec<Vec<Vec<f64>>>, kernel: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
//     let (image_height, image_width, image_depth) = (image.len(), image[0].len(), image[0][0].len());
//     let (kernel_height, kernel_width, kernel_depth) = (kernel.len(), kernel[0].len(), kernel[0][0].len());
//     assert_eq!(image_depth, kernel_depth, "L'image et le kernel doivent avoir la même profondeur");

//     let output_height = image_height - kernel_height + 1;
//     let output_width = image_width - kernel_width + 1;
//     let mut output = vec![vec![0.0; output_width]; output_height];

//     for i in 0..output_height {
//         for j in 0..output_width {
//             let mut sum = 0.0;
//             for ki in 0..kernel_height {
//                 for kj in 0..kernel_width {
//                     for kd in 0..kernel_depth {
//                         sum += image[i + ki][j + kj][kd] * kernel[ki][kj][kd];
//                     }
//                 }
//             }
//             output[i][j] = sum;
//         }
//     }
//     output
// }

// // Fonction pour sauvegarder les résultats de la convolution
// fn save_output(output: &Vec<Vec<f64>>, path: &str) {
//     let mut file = File::create(path).expect("Failed to create output file");
//     for row in output {
//         let line = row.iter()
//             .map(|v| v.to_string())
//             .collect::<Vec<String>>()
//             .join(" ");
//         writeln!(file, "{}", line).expect("Failed to write to output file");
//     }
// }

// fn main() {
//     // Charger le modèle à partir d'un fichier
//     let model_path = "../saved_models/CNN_models/2024-07-23_01-56-15/metrics.json";
//     let cnn = MyCNN::load(model_path).expect("Failed to load model");

//     // Charger une image d'entrée
//     let image_path = "../sad.png";
//     let image = load_image(Path::new(image_path)).expect("Failed to load image");

//     // Appliquer les convolutions de la première couche sur l'image
//     let kernels = match &cnn.layers[0].layer {
//         LayerType::Conv(conv_layer) => &conv_layer.kernels,
//         _ => panic!("La première couche n'est pas une couche de convolution"),
//     };

//     // Appliquer chaque kernel à l'image et sauvegarder les résultats
//     for (i, kernel) in kernels.iter().enumerate() {
//         let output = convolve(&image, kernel);
//         save_output(&output, &format!("../conv_output/output_kernel_{}.txt", i));
//     }
// }

fn main() {
    let seed = Some(42);

    let dataset_path = Path::new("../dataset_upgrade/");
    let (X_train, y_train, X_test, y_test) = load_dataset(dataset_path);

    println!("Loaded {} training images", X_train.len());
    println!("Loaded {} test images", X_test.len());
    let (X_train, y_train, X_test, y_test) = shuffle_datasets(X_train, y_train, X_test, y_test, seed);

    print_image(&X_train[0]);
    println!("label = {:?}", y_train[0]);

    let mut X_train_flatten = vec![];
    for i in 0..X_train.len(){
        X_train_flatten.push(flatten(X_train[i].clone()));
    }

    let mut X_test_flatten = vec![];
    for i in 0..X_test.len(){
        X_test_flatten.push(flatten(X_test[i].clone()));
    }

    let mut centers = vec![];
    for i in 0..1000{
        centers.push(X_train_flatten[i].clone());
    }

    // Initialiser le modèle
    let mut rbf = MyRBF::new(centers, 3, 0.01, seed);

    // Définir les hyperparamètres d'entraînement
    let learning_rate = 0.1;
    let epochs = 1000;
    let is_classification = true;

    // Entraîner le modèle
    rbf.train(X_train_flatten, y_train, X_test_flatten, y_test, learning_rate, epochs, is_classification);

    // Sauvegarder le modèle
    // rbf.save("rbf_model.json").expect("Failed to save the model");

    // // Charger le modèle
    // let loaded_rbf = MyRBF::load("rbf_model.json").expect("Failed to load the model");

    // // Vérifier les prédictions avec le modèle chargé
    // let loaded_predictions = loaded_rbf.predict(vec![vec![0.3, 0.3], vec![0.7, 0.7]], is_classification);
    // for (i, pred) in loaded_predictions.iter().enumerate() {
    //     println!("Loaded model prediction for input {}: {:?}", i, pred);
    // }
}