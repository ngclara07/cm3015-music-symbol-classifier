# CM3015 Music Symbol Classifier

A Streamlit web application for classifying musical notation symbols using a trained Support Vector Machine (SVM) model developed for the CM3015 Machine Learning and Neural Networks midterm coursework.

## Project Overview

The application performs single-image classification across ten musical symbol classes:

- Quarter note
- Half note
- Whole note
- Quarter rest
- Whole rest
- Treble clef
- Bass clef
- Sharp
- Flat
- Natural

The classifier uses a Support Vector Machine with an RBF kernel.

Uploaded images are:

1. Converted to grayscale
2. Cropped around non-white content
3. Centred on a square white background
4. Resized to 64 × 64 pixels
5. Normalised to the range [0, 1]
6. Flattened into a 4096-dimensional feature vector
7. Classified using the trained SVM model

## Project Structure

```text
cm3015-music-symbol-classifier/
├── .streamlit/
│   └── config.toml
├── saved_models/
│   └── music_symbol_svm_bundle.joblib
├── .gitignore
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## Local Setup

Create and activate a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
``` 

Install the required packages:

```powershell
python -m pip install -r requirements.txt
``` 

Run the Streamlit application:

```powershell
streamlit run streamlit_app.py
``` 

Then open:

> http://localhost:8501

## Dependencies

The application uses:

* Streamlit
* NumPy
* SciPy
* scikit-learn
* joblib
* Pillow

## Model

The deployed model is stored at:

`saved_models/music_symbol_svm_bundle.joblib`

## Deployment

The application is intended for deployment using Streamlit Community Cloud.

## Academic Context

This repository contains the deployment application for a submitted CM3015 midterm coursework project on classical machine learning for music-symbol image classification under controlled perturbations.
