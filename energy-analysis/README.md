 ⚡ Energy Analysis

This folder contains my individual contribution to the **ICT4Building** project, focused on building energy analysis, energy-performance evaluation, and consumption prediction.

---

 🎯 My Contribution

I independently developed the **energy-analysis** component, covering:

- 🔥 **Energy Signature Analysis** for heating and cooling consumption
- 📊 **Energy-performance comparison** before and after optimization
- 📈 **Regression analysis** and R² evaluation
- 🌞 **Multivariate energy analysis** using temperature difference (ΔT), GHI, and energy consumption
- 🧠 **Multivariate LSTM-based prediction** of heating energy consumption
- 📉 **Data visualization** and exploration of energy-consumption patterns

---

🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| 🐍 Programming | Python |
| 📊 Data Analysis | Pandas, NumPy |
| 📈 Visualization | Matplotlib |
| 📐 Statistical Analysis | Scikit-learn, Linear Regression, R² |
| 🤖 Machine Learning | TensorFlow / Keras, LSTM |
| ⏱️ Time Series | Time-series aggregation and analysis |

---

 🔍 Main Analysis

 🔥 1. Energy Signature Analysis

The analysis explores the relationship between **temperature difference (ΔT)** and heating/cooling energy consumption at different time resolutions.

 🏢 2. Before vs After Optimization

Energy signatures and regression results are compared **before and after building optimization** to examine changes in energy-consumption behavior.

 🌞 3. Multivariate Energy Analysis

Multiple variables are analyzed together, including:

- 🌡️ Temperature difference (ΔT)
- ☀️ Global Horizontal Irradiance (GHI)
- ❄️ Cooling energy consumption

Regression analysis and visualizations are used to explore their relationships.

 🧠 4. Multivariate LSTM Prediction

A **multivariate LSTM model** was developed to predict heating energy consumption using environmental and temporal features.

Model performance was evaluated using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **R²** — Coefficient of Determination

---

 📁 Project Structure

```text
energy-analysis/
├── data/
├── images/
├── energy_signature.py
└── README.md

📌 Note

This work was developed as part of a group academic project.

The energy-analysis component represents my individual contribution, including the analysis, regression-based evaluation, visualizations, and LSTM prediction work described above.
