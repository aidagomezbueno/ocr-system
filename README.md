# OCR Passport Information Extraction

This project demonstrates the use of an OCR system to extract information from passport images using the EasyOCR library and various preprocessing and postprocessing techniques.

## Prerequisites

- Conda
- Python 3.12.4
- GPU (optional, but recommended for faster processing)

## Installation

1. **Extract the Zip Folder**

    Extract the zip folder containing the project files.

    ```bash
    unzip ocr-passport-extraction.zip
    cd ocr-passport-extraction
    ```

2. **Create and Activate the Conda Environment**

    Create the Conda environment from the `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    conda activate ocr-env
    ```

3. **Install Any Additional Dependencies**

    Ensure all necessary dependencies are installed:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

1. **Prepare Your Images**

    Place your passport images either in `./sample_images/` or `./images` directory. Ensure that its structure is as follows:

    ```
    ocr-passport-extraction/
    ├── sample_images/
    │   ├── Canada.jpg
    │   ├── Iceland.jpg
    │   └── ...
    ├── images/
    │   ├── Australia.jpeg
    │   ├── Australia.jpg
    │   └── ...
    ├── ground_truth_data.json
    ├── ocr_system.ipynb
    ├── requirements.txt
    ├── environment.yml
    └── README.md
    ```

2. **Run the Notebook**

    To run the code and see the results, open the Jupyter Notebook directly from Jupyter Notebook:

    ```bash
    jupyter notebook ocr_system.ipynb
    ```

    Or in your favorite IDE, such as Visual Studio Code-the one used in this case to come up with the OCR system, that allows for a better utilization of the physical resources as well as different python libraries needed for its development.

    Once the notebook is open, run all cells by selecting `Kernel` > `Restart & Run All`.

3. **View Results**

    The notebook will process each image both in the `./sample_images` and `./images` directories, extract the relevant information, and compare it with the ground truth data. The results, including accuracy and processing time, will be displayed within the notebook and presented in a JSON formatted string.

## Notes

- Make sure your `ground_truth_data.json` file is properly formatted and contains the correct ground truth data for each image.
- GPU acceleration is highly recommended for faster processing. If you encounter any issues with CUDA, ensure that your environment is correctly set up to utilize the GPU.

# How to Run the Streamlit App

1. Ensure you are in the project directory and your virtual environment is activated.

2. Run the Streamlit app:
    ```sh
    streamlit run ocr_module.py
    ```

3. Open your web browser and go to the URL provided by Streamlit, typically `http://localhost:8501`.

## Usage

1. On the Streamlit app page, upload an image of a passport by clicking the "Choose an image..." button.

2. The app will process the image and display the extracted and verified information on the page.

## Notes

- This app uses EasyOCR to extract text from passport images and performs various verifications and corrections on the extracted data.
- Make sure you have a stable internet connection as the app fetches country data from a REST API.

## Troubleshooting

- If you encounter any issues with the installation or running the app, ensure that all dependencies are installed correctly.
- For GPU support with EasyOCR, ensure you have the necessary CUDA libraries installed.