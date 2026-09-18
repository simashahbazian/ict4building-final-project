 🏢 ICT in Building Design

This repository contains the work developed for the **ICT in Building Design** group project at **Politecnico di Torino**.

The project focuses on building energy simulation, analysis, and optimization for an office building in Paris.

 👩‍💻 My Contribution

As part of the group project, I independently developed the `energy-analysis` module, focusing on **building energy signatures, energy-performance comparison, and energy consumption prediction**.

The work includes:

 ⚡ Deriving energy signatures from time-series data at **hourly, daily, and weekly** levels
 📊 Comparing **heating and cooling behavior before and after optimization** using regression analysis and R²
 🔮 Developing **multivariate LSTM models** to predict heating energy consumption using outdoor temperature, GHI, and hourly information
 📈 Visualizing energy signatures, regression results, and actual vs. predicted energy consumption

 📁 Contribution Structure

The main implementation is contained in:

`energy-analysis/energy_signature.py`

Supporting figures and outputs are available in:

`energy-analysis/images/`

 🔬 Energy Signature Analysis

The energy signature analysis examines the relationship between building energy consumption and temperature-related variables.

Separate analyses were performed for:

 🔥 Heating energy
 ❄️ Cooling energy
 ⏱️ Hourly data
 📅 Daily aggregated data
 📆 Weekly aggregated data

Linear regression was used to analyze the relationship between `ΔT` and heating/cooling energy consumption.

The analysis also compares the energy behavior of the building **before and after optimization**.

 📊 Performance Comparison

The analysis evaluates heating and cooling behavior before and after optimization using regression-based comparisons.

For selected analyses, the coefficient of determination (**R²**) is calculated at:

 Hourly level
 Daily level
 Weekly level

For cooling analysis, additional variables such as **GHI (Global Horizontal Irradiance)** and hour information were incorporated into the regression analysis.

 🤖 Energy Consumption Prediction

A multivariate **Long Short-Term Memory (LSTM)** model was developed for heating energy prediction.

The model uses:

 🌡️ Outdoor temperature
 ☀️ Global Horizontal Irradiance (GHI)
 🕐 Hour of day

to predict:

 🔥 Heating energy consumption (`heating_kWh`)

A 24-timestep sequence window was used for the prediction task.

Model performance was evaluated using:

 MAE
 MSE
 R²

Actual and predicted heating consumption were also visualized for comparison.

 🛠️ Tools & Technologies

 🐍 Python
 🐼 Pandas
 🔢 NumPy
 📊 Matplotlib
 📐 Scikit-learn
 🧠 TensorFlow / Keras
 📈 Linear Regression
 🤖 LSTM
 📊 R², MAE, MSE

 📌 Project Scope

This repository is a **group project**. The `energy-analysis` directory represents my individual contribution to the project and should be considered separately from the other project components.
