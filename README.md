# ❤️ Heart Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-green?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange?style=for-the-badge&logo=scikit-learn)

**A Machine Learning-powered web application for predicting heart disease risk using patient health data**

[⭐ Star on GitHub](https://github.com/widushan/Heart-Disease-Prediction) | [📖 Documentation](#-features) | [🚀 Quick Start](#-quick-start)

</div>

---

## 📋 Overview

This project is a **Heart Disease Prediction System** that leverages Machine Learning algorithms to assess the risk of heart disease based on various health parameters. The system uses a Random Forest classifier trained on medical data to provide accurate predictions through an intuitive web interface.

## ✨ Features

- 🎯 **Accurate Predictions**: Powered by Random Forest Machine Learning model
- 🌐 **Web Interface**: User-friendly Flask-based web application
- 📊 **Multiple Health Parameters**: Analyzes 15+ health indicators including:
  - Age, Gender, BMI
  - Blood Pressure (Systolic & Diastolic)
  - Cholesterol levels
  - Smoking habits
  - Diabetes status
  - Heart rate & Glucose levels
- 🔒 **Scaled Data Processing**: Normalized input features for better accuracy
- 💻 **Easy to Use**: Simple form-based input system

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Random Forest Classifier
- **Data Processing**: NumPy, Pandas
- **Model Storage**: Pickle

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/widushan/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5000`
   <img width="1916" height="737" alt="Image" src="https://github.com/user-attachments/assets/3b491478-6dfc-43bb-8d67-0afacfcfae56" />
   
   - Fill in the health parameters
   <img width="1917" height="736" alt="Image" src="https://github.com/user-attachments/assets/0c031d6b-7f00-42c5-a986-015c96bae2fe" />
   
   - Get instant predictions!
   <img width="1914" height="798" alt="Image" src="https://github.com/user-attachments/assets/428fa24e-5cfa-419d-a0f6-8b3ced7a47af" />

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                 # Flask application
├── Models/                # Trained ML models
│   ├── rf_classifier.pkl  # Random Forest model
│   └── scaler.pkl         # Feature scaler
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── heart_disease.csv      # Dataset
└── README.md              # Project documentation
```

## 🎯 How It Works

1. User enters health parameters through the web form
2. Data is preprocessed and scaled using the trained scaler
3. Random Forest model analyzes the features
4. Prediction result is displayed (Heart Disease / No Heart Disease)

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 15 health-related parameters
- **Output**: Binary classification (0 = No Heart Disease, 1 = Heart Disease)

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📝 License

This project is open source and available for educational purposes.

## 🔗 Links

- **GitHub Repository**: [https://github.com/widushan/Heart-Disease-Prediction](https://github.com/widushan/Heart-Disease-Prediction)

---

<div align="center">

**Made with ❤️ using Python & Machine Learning**

⭐ Star this repo if you find it helpful!

</div>
