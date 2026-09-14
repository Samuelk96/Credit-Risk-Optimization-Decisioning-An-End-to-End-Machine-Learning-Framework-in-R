# 💳 Credit Risk & Decisioning Framework (End-to-End Analytics)

[![R Version](https://img.shields.io/badge/R-4.3%2B-blue.svg)](https://www.r-project.org/)
[![Framework](https://img.shields.io/badge/tidymodels-Ecosystem-orange.svg)](https://www.tidymodels.org/)
[![XAI](https://img.shields.io/badge/Explainability-DALEX%20%2F%20SHAP-green.svg)](https://dalex.drwhy.ai/)
[![Domain](https://img.shields.io/badge/Domain-Consumer%20Credit%20Risk-red.svg)]()

## 📌 Resumen del Proyecto
Este repositorio contiene la implementación de un marco de trabajo (*end-to-end*) para la predicción de impago (*Default Prediction*) y optimización de políticas de concesión de crédito al consumo. 

El marco combina la interpretabilidad y gobernanza regulatoria de los **Scorecards analíticos (WOE / IV)** con la potencia discriminatoria de modelos de **Gradient Boosting (XGBoost)**, cerrando el ciclo técnico con la optimización financiera del punto de corte (*Cut-off*) para maximizar el margen de utilidad de la cartera.

---

## 🏗️ Arquitectura del Repositorio

```text
credit-risk-ml-framework-r/
├── scripts/
│   ├── 01_eda_and_data_cleaning.R        # EDA, limpieza y profiling de variables
│   ├── 02_feature_engineering_woe.R      # Fine/Coarse binning, WOE e Information Value (IV)
│   ├── 03_model_training_and_tuning.R    # Baseline Logístico vs XGBoost (Tuning con 5-fold CV)
│   ├── 04_model_evaluation_and_xai.R     # Gobernanza (Gini, KS, PSI) y Explicabilidad (SHAP via DALEX)
│   └── 05_business_strategy_cutoff.R    # Simulación financiera, Cut-off óptimo y Escalamiento a Score (300-850)
├── data/
│   ├── raw/                              # Datasets originales (o rutina sintética Script 01)
│   └── processed/                        # Artefactos serializados (.rds) entre scripts
└── outputs/
    ├── figures/                          # Gráficas generadas (ROC, SHAP, Cut-off, Score dist)
    └── reports/                          # Tablas resumen de gobernanza y estrategia
![Curva ROC](outputs/figures/04_roc_curves.png)
![Optimizacion Cutoff](outputs/figures/05_cutoff_optimization.png)
