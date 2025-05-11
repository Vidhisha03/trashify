# Garbage Classifier

This project is a garbage classification system that uses a machine learning model to classify images of garbage into different categories.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using the following command:
   ```
   pip install -r requirements.txt
   ```
4. Run the application using the command:
   ```
   python app.py
   ```
5. Open your browser and go to [http://localhost:5000/](http://localhost:5000/).

## Usage

- Upload an image of garbage to classify it into categories like cardboard, glass, metal, paper, plastic, etc.

## Project Structure

- `app.py`: The main application file.
- `model/`: Contains the machine learning model and related files.
- `static/`: Contains static files like images and CSS.
- `templates/`: Contains HTML templates for the web application.
- `uploads/`: Directory for storing uploaded images.

## Requirements

The project requires the following Python packages:
- Pillow
- pathlib
- numpy
- matplotlib
- torchvision

These packages are listed in the `requirements.txt` file.