import numpy as np
import pandas as pd
import os
import requests

csv_dataset_url_path = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
csv_destination = "car_fuel_efficiency_2026.csv"
origin_column_name = "origin"
origin = "Asia"
horsepower_column_name = "horsepower"

def download_dataset(url: str, filename: str):
    current_directory = os.getcwd()
    destination = os.path.join(current_directory, filename)

    # 3. Check if the file already exists
    if os.path.exists(destination):
        print(f"Skipping download: '{filename}' already exists in {current_directory}")
    else:
        print(f"File not found. Downloading '{filename}' to {current_directory}...")

        try:
            # 4. Download and write the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Throws an error if the URL is broken (404, 500, etc.)

            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print("Download complete successfully!")

        except Exception as e:
            print(f"An error occurred during the download: {e}")

def get_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def question_1() -> str:
    return pd.__version__

def question_2(dataset: pd.DataFrame) -> int:
    return dataset.shape[0]

def question_3(dataset: pd.DataFrame) -> int:
    fuel_type_column_name = "fuel_type"
    return len(dataset[fuel_type_column_name].unique())

def question_4(dataset: pd.DataFrame) -> int:
    num_cols_missing_values = 0
    for col in dataset.isnull().sum():
        if col > 0:
            num_cols_missing_values += 1
    return num_cols_missing_values

def question_5(dataset: pd.DataFrame) -> float:
    max_efficiency_by_origin = dataset.groupby(origin_column_name).fuel_efficiency_mpg.max()
    return max_efficiency_by_origin[origin]

def question_6(dataset: pd.DataFrame) -> str:
    median_horsepower = dataset[horsepower_column_name].median()

    # ASSUMPTION: mode and most commonly occurring value are the same
    horsepower_mode = dataset[horsepower_column_name].mode()[0]

    no_null_median_horsepower = dataset[horsepower_column_name].fillna(horsepower_mode).median()

    if no_null_median_horsepower > median_horsepower:
        return "Yes, it increased"
    elif no_null_median_horsepower < median_horsepower:
        return "Yes, it decreased"
    else:
        return "No"

def question_7(dataset: pd.DataFrame) -> float:
    vehicle_weight_column_name = "vehicle_weight"
    model_year_column_name = "model_year"
    x = dataset.query("origin == 'Asia'")[[vehicle_weight_column_name, model_year_column_name]][0:7].to_numpy()

    x_transpose = x.T
    xtx = np.matmul(x_transpose, x)
    xtx_inverse = np.linalg.inv(xtx)

    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

    w = np.matmul(xtx_inverse, x_transpose) * y
    return w.sum()


download_dataset(csv_dataset_url_path, csv_destination)
dataset = get_dataset(csv_destination)
print(f"Question 1: {question_1()}")
print(f"Question 2: {question_2(dataset)}")
print(f"Question 3: {question_3(dataset)}")
print(f"Question 4: {question_4(dataset)}")
print(f"Question 5: {question_5(dataset)}")
print(f"Question 6: {question_6(dataset)}")
print(f"Question 7: {question_7(dataset)}")
