use std::cmp::{max, min};
extern crate nalgebra;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::fs::create_dir_all;

pub fn chunk_vector(input: Vec<f64>, chunk_size: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();
    let mut chunk: Vec<f64> = Vec::new();

    for (i, num) in input.iter().enumerate() {
        chunk.push(*num);
        if (i + 1) % chunk_size == 0 || i == input.len() - 1 {
            result.push(chunk.clone());
            chunk.clear();
        }
    }

    result
}

pub fn matrix_from_2d_vec(data: &Vec<Vec<f64>>) -> DMatrix<f64> {
    let rows = data.len();
    let cols = data[0].len();
    let mut matrix = DMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            matrix[(i, j)] = data[i][j];
        }
    }
    matrix
}

pub fn reshape2D(input: Vec<f64>, reshape_shape: (usize, usize)) -> Vec<Vec<f64>> {
    let (rows, cols) = reshape_shape;
    let total_elements = rows * cols;
    
    // Vérifie si la taille du vecteur d'entrée correspond aux dimensions souhaitées
    if input.len() != total_elements {
        panic!("La taille du vecteur d'entrée ne correspond pas aux dimensions souhaitées");
    }

    // Crée un nouveau vecteur 2D pour stocker le résultat
    let mut output = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            output[i][j] = input[i * cols + j];
        }
    }

    output
}

pub fn reshape3D(input: Vec<f64>, reshape_shape: (usize, usize, usize)) -> Vec<Vec<Vec<f64>>> {
    let (dim1, dim2, dim3) = reshape_shape;
    let total_elements = dim1 * dim2 * dim3;

    // Vérifie si la taille du vecteur d'entrée correspond aux dimensions souhaitées
    if input.len() != total_elements {
        panic!("La taille du vecteur d'entrée ne correspond pas aux dimensions souhaitées");
    }

    // Crée un nouveau vecteur 3D pour stocker le résultat
    let mut output = vec![vec![vec![0.0; dim3]; dim2]; dim1];

    for i in 0..dim1 {
        for j in 0..dim2 {
            for k in 0..dim3 {
                output[i][j][k] = input[i * dim2 * dim3 + j * dim3 + k];
            }
        }
    }

    output
}


pub fn reshape4D(input: Vec<f64>, reshape_shape: (usize, usize, usize, usize)) -> Vec<Vec<Vec<Vec<f64>>>> {
    let (dim1, dim2, dim3, dim4) = reshape_shape;
    let total_elements = dim1 * dim2 * dim3 * dim4;

    // Vérifie si la taille du vecteur d'entrée correspond aux dimensions souhaitées
    if input.len() != total_elements {
        panic!("La taille du vecteur d'entrée ne correspond pas aux dimensions souhaitées");
    }

    // Crée un nouveau vecteur 4D pour stocker le résultat
    let mut output = vec![vec![vec![vec![0.0; dim4]; dim3]; dim2]; dim1];

    for i in 0..dim1 {
        for j in 0..dim2 {
            for k in 0..dim3 {
                for l in 0..dim4 {
                    output[i][j][k][l] = input[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l];
                }
            }
        }
    }

    output
}


pub fn print_matrix(vec: &Vec<Vec<f64>>) {
    println!("[");
    for inner_vec in vec {
        print!("  [");
        for (i, val) in inner_vec.iter().enumerate() {
            if i != 0 {
                print!(", ");
            }
            print!("{:.2}", val); // Affiche les valeurs avec deux chiffres après la virgule
        }
        println!("],");
    }
    println!("]");
}

pub fn print_3d(vec: &Vec<Vec<Vec<f64>>>) {
    println!("[");
    for matrix in vec {
        println!("  [");
        for row in matrix {
            print!("    [");
            for (i, val) in row.iter().enumerate() {
                if (i != 0) {
                    print!(", ");
                }
                print!("{:.2}", val);
            }
            println!("],");
        }
        println!("  ],");
    }
    println!("]");
}

pub fn gauss_kernel(x: &Vec<f64>, c: &Vec<f64>, gamma: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - c[i]).powi(2);
    }

    (-sum / (2.0 * gamma.powi(2))).exp()
}


pub fn zeros_padding(input: Vec<Vec<f64>>, padding_height: usize, padding_width: usize) -> Vec<Vec<f64>> {
    let input_height = input.len();
    let input_width = input[0].len();
    let new_height = input_height + 2 * padding_height;
    let new_width = input_width + 2 * padding_width;

    let mut padded_input = vec![vec![0.0; new_width]; new_height];

    for i in 0..input_height {
        for j in 0..input_width {
            padded_input[i + padding_height][j + padding_width] = input[i][j];
        }
    }

    padded_input
}

pub fn generate_random_4d_tensor(shape: (usize, usize, usize, usize)) -> Vec<Vec<Vec<Vec<f64>>>> {
    let (d1, d2, d3, d4) = shape;
    let mut rng = rand::thread_rng();

    let mut tensor = vec![
        vec![
            vec![
                vec![0.0; d4];
                d3
            ];
            d2
        ];
        d1
    ];

    for i in 0..d1 {
        for j in 0..d2 {
            for k in 0..d3 {
                for l in 0..d4 {
                    tensor[i][j][k][l] = rng.gen_range(0.0..1.0);
                }
            }
        }
    }

    tensor
}

pub fn generate_random_2d_tensor(shape: (usize, usize)) -> Vec<Vec<f64>> {
    let (d1, d2) = shape;
    let mut rng = rand::thread_rng();

    let mut tensor = vec![vec![0.0; d2]; d1];

    for i in 0..d1 {
        for j in 0..d2 {
            tensor[i][j] = rng.gen_range(0.0..1.0);
        }
    }

    tensor
}

pub fn print_image(image: &Vec<Vec<Vec<f64>>>) {
    // Assuming the image is grayscale and has only one channel.
    for row in image {
        for pixel in row {
            // Assuming the pixel has one channel.
            let intensity = pixel[0];
            // Convert intensity to character for display
            let char = if intensity > 0.8 {
                '#'
            } else if intensity > 0.6 {
                'O'
            } else if intensity > 0.4 {
                '*'
            } else if intensity > 0.2 {
                '.'
            } else {
                ' '
            };
            print!("{}", char);
        }
        println!();
    }
}

pub fn flatten_images(x: Vec<Vec<Vec<Vec<f64>>>>) -> Vec<Vec<f64>> {
    x.into_iter()
        .map(|images| {
            images
                .into_iter()
                .flat_map(|image| {
                    image.into_iter().flatten().collect::<Vec<f64>>()
                })
                .collect()
        })
        .collect()
}

pub fn flatten(input: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    let mut output = Vec::new();

    for matrix in input {
        for row in matrix {
            for &value in row.iter() {
                output.push(value);
            }
        }
    }

    output
}

pub fn reshape(input: Vec<f64>, shape: (usize, usize, usize)) -> Vec<Vec<Vec<f64>>> {
    let (d1, d2, d3) = shape;
    assert_eq!(input.len(), d1 * d2 * d3, "The size of input does not match the specified shape");

    let mut output = vec![vec![vec![0.0; d3]; d2]; d1];
    let mut idx = 0;

    for i in 0..d1 {
        for j in 0..d2 {
            for k in 0..d3 {
                output[i][j][k] = input[idx];
                idx += 1;
            }
        }
    }

    output
}
pub fn print_non_zero_elements(array: &Vec<Vec<Vec<f64>>>) {
    for (i, layer) in array.iter().enumerate() {
        for (j, row) in layer.iter().enumerate() {
            for (k, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    println!("Element at ({}, {}, {}): {}", i, j, k, value);
                }
            }
        }
    }
}

pub fn print_non_zero_elements2(array: &Vec<Vec<f64>>) {
    for (i, row) in array.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                println!("Element at ({}, {}): {}", i, j, value);
            }
        }
    }
}

pub fn print_non_zero_elements1(array: &Vec<f64>) {
    for (i, &value) in array.iter().enumerate() {
        if value != 0.0 {
            println!("Element at ({}): {}", i, value);
        }
    }
}

pub fn print_non_zero_elements4(array: &Vec<Vec<Vec<Vec<f64>>>>) {
    for (w, mat) in array.iter().enumerate() {
        for (x, layer) in mat.iter().enumerate() {
            for (y, row) in layer.iter().enumerate() {
                for (z, &value) in row.iter().enumerate() {
                    if value != 0.0 {
                        println!("Element at ({}, {}, {}, {}): {}", w, x, y, z, value);
                    }
                }
            }
        }
    }
}

pub fn argmax(input:&Vec<f64>) -> usize{
    let mut max = input[0];
    let mut max_index = 0;
    for i in 1..input.len(){
        if input[i] > max{
            max = input[i];
            max_index = i;
        }
    }
    max_index
}

pub fn current_datetime_str() -> String {
    let now = chrono::Local::now();
    now.format("%Y-%m-%d_%H-%M-%S").to_string()
}

pub fn create_model_dirs(base_dir: &str, datetime_str: &str) -> (String, String, String) {
    let model_dir = format!("{}/{}", base_dir, datetime_str);
    let epoch_dir = format!("{}/epochs", model_dir);
    let best_model_dir = format!("{}/best_model", model_dir);

    create_dir_all(&epoch_dir).expect("Failed to create epoch directory");
    create_dir_all(&best_model_dir).expect("Failed to create best model directory");

    (model_dir, epoch_dir, best_model_dir)
}