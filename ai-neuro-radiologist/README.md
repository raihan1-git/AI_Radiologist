# AI Neuro Radiologist Application

## Overview
The AI Neuro Radiologist application is designed to assist in the analysis and interpretation of neuroimaging data using advanced machine learning techniques. This project leverages deep learning models to automate the detection and classification of various neurological conditions from MRI scans.

## Project Structure
The project is organized into the following directories:

- **data/**: Contains all data-related files.
  - **raw/**: Original .nii.gz files, such as the BraTS dataset.
  - **processed/**: Preprocessed or cached volumes ready for model training.
  - **models/**: Saved models after training.

- **notebooks/**: Jupyter notebooks for exploratory data analysis and model development.
  - **exploration.ipynb**: For initial exploratory data analysis (EDA).
  - **model-development.ipynb**: For quick prototyping and model development.

- **scripts/**: Python scripts for data processing, training, and evaluation.
  - **preprocess.py**: Preprocessing raw data and preparing it for training.
  - **train.py**: Instantiating models and starting the training process.
  - **evaluate.py**: Evaluating trained models on test data.

- **src/**: Source code for the core machine learning package.
  - **app.py**: Main entry point for the core machine learning package.
  - **data_loader.py**: Functions and classes for loading and processing datasets.
  - **models.py**: Defines the neural network architectures used in the application.
  - **utils.py**: Utility functions that support various operations in the project.

- **streamlit_app/**: Streamlit web application files.
  - **main.py**: Entry point for the Streamlit web application.
  - **config.py**: Configuration settings for the Streamlit application.

- **requirements.txt**: Lists the project dependencies required for installation.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-neuro-radiologist
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place the raw neuroimaging data in the `data/raw` directory.
   - Run the preprocessing script to prepare the data for training:
     ```
     python scripts/preprocess.py
     ```

4. Train the model:
   ```
   python scripts/train.py
   ```

5. Evaluate the model:
   ```
   python scripts/evaluate.py
   ```

6. Run the Streamlit application:
   ```
   streamlit run streamlit_app/main.py
   ```

## Usage
Follow the instructions in the notebooks for exploratory data analysis and model development. Use the scripts for preprocessing, training, and evaluating models. The Streamlit application provides an interactive interface for visualizing results and making predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you would like to add.

## License
This project is licensed under the MIT License. See the LICENSE file for details.