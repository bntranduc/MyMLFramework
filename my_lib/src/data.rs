use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;
use image::io::Reader as ImageReader;
use image::GenericImageView;
use image::ImageError;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};

pub fn load_dataset(dataset_path: &Path) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>, Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>) {
    let emotions = vec!["happy", "neutral", "sad"];
    let mut emotion_to_label = HashMap::new();
    for (i, &emotion) in emotions.iter().enumerate() {
        emotion_to_label.insert(emotion, i);
    }

    let num_classes = emotions.len();

    let mut X_train = Vec::new();
    let mut y_train = Vec::new();
    let mut X_test = Vec::new();
    let mut y_test = Vec::new();

    for entry in fs::read_dir(dataset_path.join("train")).unwrap() {
        let entry = entry.unwrap();
        let emotion = entry.file_name().into_string().unwrap();
        let label = *emotion_to_label.get(emotion.as_str()).unwrap();

        for image_entry in WalkDir::new(entry.path()) {
            let image_entry = image_entry.unwrap();
            if image_entry.file_type().is_file() {
                match load_image(&image_entry.path()) {
                    Ok(image) => {
                        X_train.push(image);
                        y_train.push(one_hot_encode(label, num_classes));
                    }
                    Err(e) => {
                        eprintln!("Failed to load image {:?}: {}", image_entry.path(), e);
                    }
                }
            }
        }
    }

    for entry in fs::read_dir(dataset_path.join("test")).unwrap() {
        let entry = entry.unwrap();
        let emotion = entry.file_name().into_string().unwrap();
        let label = *emotion_to_label.get(emotion.as_str()).unwrap();

        for image_entry in WalkDir::new(entry.path()) {
            let image_entry = image_entry.unwrap();
            if image_entry.file_type().is_file() {
                match load_image(&image_entry.path()) {
                    Ok(image) => {
                        X_test.push(image);
                        y_test.push(one_hot_encode(label, num_classes));
                    }
                    Err(e) => {
                        eprintln!("Failed to load image {:?}: {}", image_entry.path(), e);
                    }
                }
            }
        }
    }

    (X_train, y_train, X_test, y_test)
}

pub fn shuffle_datasets(
    mut X_train: Vec<Vec<Vec<Vec<f64>>>>, 
    mut y_train: Vec<Vec<f64>>, 
    mut X_test: Vec<Vec<Vec<Vec<f64>>>>, 
    mut y_test: Vec<Vec<f64>>,
    seed: Option<u64>
) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>, Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>) {

    // Utilisation de la seed si elle est fournie
    let mut rng: StdRng = match seed {
        Some(s) => SeedableRng::seed_from_u64(s),
        None => SeedableRng::from_entropy(),
    };

    let mut train_data: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)> = X_train.into_iter().zip(y_train.into_iter()).collect();
    train_data.shuffle(&mut rng);
    let (shuffled_X_train, shuffled_y_train): (Vec<_>, Vec<_>) = train_data.into_iter().unzip();

    let mut test_data: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)> = X_test.into_iter().zip(y_test.into_iter()).collect();
    test_data.shuffle(&mut rng);
    let (shuffled_X_test, shuffled_y_test): (Vec<_>, Vec<_>) = test_data.into_iter().unzip();

    (shuffled_X_train, shuffled_y_train, shuffled_X_test, shuffled_y_test)
}

pub fn load_image(path: &Path) -> Result<Vec<Vec<Vec<f64>>>, ImageError> {
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_exact(48, 48, image::imageops::FilterType::Nearest);
    let img = img.to_rgb8();

    let rows = img.height() as usize;
    let cols = img.width() as usize;
    let mut array = vec![vec![vec![0.0; 3]; cols]; rows];

    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f64, pixel[1] as f64, pixel[2] as f64);
        array[y as usize][x as usize][0] = r/255.;
        array[y as usize][x as usize][1] = g/255.;
        array[y as usize][x as usize][2] = b/255.;
    }

    Ok(array)
}

fn one_hot_encode(label: usize, num_classes: usize) -> Vec<f64> {
    let mut encoding = vec![0.0; num_classes];
    encoding[label] = 1.0;
    encoding
}
