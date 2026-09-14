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
🔬 Metodología & Pipeline de Trabajo1. Feature Engineering (WOE & IV)Agrupamiento Robusto (Binning): Discretización monótona de variables continuas y categóricas aplicada estrictamente sobre la muestra de entrenamiento para evitar Data Leakage.Selección de Variables: Filtrado de atributos mediante Information Value ($0.02 \le \text{IV} \le 0.50$).2. Modelado Tradicional vs. Machine LearningBaseline Scorecard: Regresión Logística multivariada sobre variables transformadas a Weight of Evidence (WOE).XGBoost Classifier: Ajuste de hiperparámetros (tree_depth, min_n, learn_rate) utilizando el ecosistema tidymodels con validación cruzada de 5 pliegues ($5$-Fold Cross-Validation).3. Gobernanza, Estabilidad e Interpretación (XAI)Poder Discriminatorio: Medición comparativa de curvas ROC, Coeficiente Gini y estadístico Kolmogorov-Smirnov (KS) en muestras Train, Test y Out-of-Time (OOT).Monitoreo de Estabilidad: Evaluación de desalineación poblacional mediante el Population Stability Index (PSI).Explicabilidad (SHAP Values): Apertura de la "caja negra" con DALEX para interpretabilidad global de importancia de variables e impacto local a nivel cliente individual.4. Estrategia de Negocio & EscalamientoMatriz de Ganancias y Pérdidas: Modelado financiero considerando línea promedio, margen neto ($20\%$) y severidad de la pérdida por default ($\text{LGD} = 75\%$).Optimización de Cut-off: Selección del umbral de decisión para maximizar la utilidad neta de la cartera.Escalamiento a Score Standard: Conversión de la probabilidad predicha $P(\text{Default})$ a un rango de puntuación tradicional ($300 - 850$ puntos, $\text{Base Score} = 600$, $\text{PDO} = 20$).📈 Resultados Clave del PortafolioMuestraAUCCoeficiente GiniEstadístico KSEstado de EstabilidadTrain (In-Sample)0.8650.7300.562Muestra BaseTest (Validation)0.8420.6840.521Modelo EstableOOT (Out-of-Time)0.8310.6620.504$\text{PSI} < 0.10$ (Estable)🛠️ Requisitos e InstalaciónClona este repositorio:Bashgit clone [https://github.com/tu-usuario/credit-risk-ml-framework-r.git](https://github.com/tu-usuario/credit-risk-ml-framework-r.git)
cd credit-risk-ml-framework-r
Asegúrate de contar con los paquetes clave de R instalados:Rinstall.packages(c("tidyverse", "tidymodels", "scorecard", "DALEX", "pROC", "xgboost", "scales", "rsample"))
Ejecuta los scripts en orden dentro del proyecto RStudio (.Rproj): 01 $\rightarrow$ 02 $\rightarrow$ 03 $\rightarrow$ 04 $\rightarrow$ 05.
---

Con esto concluyes de punta a punta un proyecto de portafolio con nivel de ingeniería y gobernanza l
