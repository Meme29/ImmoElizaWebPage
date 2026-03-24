# ImmoElizaWebPage

## 📖 Description
This API allows users to send requests containing information about a house.
The user fills out a form with various features (such as location, size, number of rooms, etc.), and the API processes this data to return an estimated price prediction based on a trained machine learning model.

## ⚙️ Requirements

How to use the requirements file :
pip install -r requirements.txt

## 📥 Installation

git clone project

## ▶️ Usage

uvicorn main:app --reload

## 🗂️ Project structure

📦 project  
├── 🚀 main.py                # API FastAPI (entry point)   
│  
├── 📂 data                  
│   ├── distances.csv         # Computed distances between cities and 5 big cities of Belgium 
│   └── postal_codes.csv      # List of postal codes and Locality names 
│  
├── 📂 pages                 # Frontend (HTML)  
│   └── homepage.html  
│  
├── 📂 preprocessing         # Pre-processing of data  
│   ├── preprocessor.py  
│   └── distances.py  
│  
├── 📂 predicts              # Predictions  
│   └── predictor.py  
│  
├── 📂 models                # Trained models  
│   ├── random_forest_model.pkl  
│   └── ...  


## 🤝 Contributing


## ✍️ Author / Credits
Team 6 💪
