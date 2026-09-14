# ==============================================================================
# SCRIPT 04: EVALUACIÓN DE MODELOS, GOBERNANZA Y EXPLICABILIDAD (XAI - SHAP)
# Proyecto: Credit Risk Optimization & Decisioning Framework
# Autor: Gaspar Samuel Salazar Salgado
# Objetivo: 
#   1. Cargar los modelos entrenados (Script 03) y los datos WOE (Script 02).
#   2. Generar predicciones para las muestras Train, Test y Out-of-Time (OOT).
#   3. Calcular el poder discriminatorio: Coeficiente Gini, Kolmogorov-Smirnov (KS) y ROC-AUC.
#   4. Evaluar la estabilidad poblacional mediante el Population Stability Index (PSI).
#   5. Aplicar Explainable AI (XAI) con valores SHAP (global y local) via DALEX.
#   6. Exportar métricas y artefactos visuales de gobernanza.
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 1: CARGA DE LIBRERÍAS Y CONFIGURACIÓN
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse) # Manipulación de datos y visualizaciones
  library(tidymodels)# Evaluaciones y predicciones estándar
  library(scorecard) # Cálculo de KS, ROC y PSI estandarizados
  library(DALEX)     # Framework para Explainable AI (XAI)
  library(pROC)      # Análisis de curvas ROC y Gini nativo
  library(scales)    # Formato numérico y porcentajes
})

# Directorios de salida
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/reports", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# PASO 2: CARGA DE MODELOS Y DATOS
# ------------------------------------------------------------------------------
data_bundle_path   <- "data/processed/data_woe_bundle.rds"
models_bundle_path <- "outputs/models/trained_models_bundle.rds"

if (!file.exists(data_bundle_path) || !file.exists(models_bundle_path)) {
  stop("[ERROR] Faltan archivos requeridos de los Scripts 02 o 03.")
}

woe_bundle    <- readRDS(data_bundle_path)
models_bundle <- readRDS(models_bundle_path)

# Muestras procesadas
train_df <- woe_bundle$train
test_df  <- woe_bundle$test
oot_df   <- woe_bundle$oot

# Modelos ajustados
logit_fit <- models_bundle$logit_model
xgb_fit   <- models_bundle$xgb_model

cat("=== MODELOS Y DATOS CARGADOS CON ÉXITO ===\n")

# ------------------------------------------------------------------------------
# PASO 3: GENERACIÓN DE PREDICCIONES (PROBABILIDAD DE DEFAULT)
# ------------------------------------------------------------------------------
cat("\n=== GENERANDO PREDICCIONES EN MUESTRAS TRAIN, TEST Y OOT ===\n")

# Extraer probabilidades de impago P(Y = 1) para XGBoost
train_preds <- predict(xgb_fit, train_df, type = "prob")$.pred_1
test_preds  <- predict(xgb_fit, test_df, type = "prob")$.pred_1
oot_preds   <- predict(xgb_fit, oot_df, type = "prob")$.pred_1

# Dataframes consolidados con Target numérico y Probabilidad Predicha
df_eval_train <- tibble(target = as.numeric(as.character(train_df$target)), pred = train_preds)
df_eval_test  <- tibble(target = as.numeric(as.character(test_df$target)), pred = test_preds)
df_eval_oot   <- tibble(target = as.numeric(as.character(oot_df$target)), pred = oot_preds)

# ------------------------------------------------------------------------------
# PASO 4: PODER DISCRIMINATORIO Y CURVAS ROC (COMPATIBLE CON PROC)
# ------------------------------------------------------------------------------
cat("\n=== CALCULANDO MÉTRICAS DE PODER DISCRIMINATORIO ===\n")

roc_train <- roc(df_eval_train$target, df_eval_train$pred, quiet = TRUE)
roc_test  <- roc(df_eval_test$target,  df_eval_test$pred,  quiet = TRUE)
roc_oot   <- roc(df_eval_oot$target,   df_eval_oot$pred,   quiet = TRUE)

calc_metrics_from_roc <- function(roc_obj, sample_name) {
  auc_val  <- as.numeric(auc(roc_obj))
  gini_val <- 2 * auc_val - 1
  ks_stat  <- max(roc_obj$sensitivities + roc_obj$specificities - 1)
  
  tibble(
    Muestra = sample_name,
    AUC = round(auc_val, 4),
    Gini = round(gini_val, 4),
    KS = round(ks_stat, 4)
  )
}

metrics_summary <- bind_rows(
  calc_metrics_from_roc(roc_train, "Train (In-Sample)"),
  calc_metrics_from_roc(roc_test,  "Test (Validation)"),
  calc_metrics_from_roc(roc_oot,   "OOT (Out-of-Time)")
)

print(metrics_summary)
saveRDS(metrics_summary, "outputs/reports/governance_metrics.rds")

# Generar la gráfica ROC procesando manualmente las coordenadas
roc_df <- bind_rows(
  tibble(fpr = 1 - roc_train$specificities, tpr = roc_train$sensitivities, Muestra = "Train"),
  tibble(fpr = 1 - roc_test$specificities,  tpr = roc_test$sensitivities,  Muestra = "Test"),
  tibble(fpr = 1 - roc_oot$specificities,   tpr = roc_oot$sensitivities,   Muestra = "OOT")
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = Muestra)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Train" = "#2A9D8F", "Test" = "#E76F51", "OOT" = "#264653")) +
  labs(
    title = "Curvas ROC por Muestra (Evaluación de Sobreajuste)",
    subtitle = "Modelo XGBoost - TransUnion Risk Framework",
    x = "1 - Especificidad (Falsos Positivos)",
    y = "Sensibilidad (Verdaderos Positivos)",
    color = "Muestra"
  ) +
  theme_minimal()

ggsave("outputs/figures/04_roc_curves.png", p_roc, width = 7, height = 5)

# ------------------------------------------------------------------------------
# PASO 5: EVALUACIÓN DE ESTABILIDAD POBLACIONAL (PSI)
# ------------------------------------------------------------------------------
cat("\n=== EVALUANDO ESTABILIDAD POBLACIONAL (PSI TRAIN VS. OOT) ===\n")

psi_result <- perf_psi(
  score = list(train = df_eval_train$pred, oot = df_eval_oot$pred),
  label = list(train = df_eval_train$target, oot = df_eval_oot$target)
)

cat("PSI global entre Train y OOT:", round(psi_result$psi$psi, 4), "\n")

# ------------------------------------------------------------------------------
# PASO 6: EXPLICABILIDAD CON SHAP VALUES (XAI - DALEX)
# ------------------------------------------------------------------------------
cat("\n=== CONFIGURANDO EXPLICADOR DALEX PARA VALORES SHAP ===\n")

x_train <- train_df %>% select(-target)
y_train <- as.numeric(as.character(train_df$target))

# Función predictora personalizada para workflows de tidymodels
predict_xgb_dalex <- function(model, newdata) {
  predict(model, newdata, type = "prob")$.pred_1
}

# Creación del Explicador DALEX
explainer_xgb <- explain(
  model = xgb_fit,
  data = x_train,
  y = y_train,
  predict_function = predict_xgb_dalex,
  label = "XGBoost Credit Risk",
  verbose = FALSE
)

# 1. IMPORTANCIA GLOBAL DE VARIABLES (Definiendo loss_root_mean_square explícitamente)
cat("Calculando Importancia Global con DALEX...\n")
vip_global <- model_parts(
  explainer = explainer_xgb, 
  loss_function = loss_root_mean_square
)

p_shap_global <- plot(vip_global) +
  labs(
    title = "Importancia Global de Variables (Feature Importance)",
    subtitle = "Perdida de rendimiento al permutar variables (RMSE)"
  ) +
  theme_minimal()

ggsave("outputs/figures/04_shap_global_importance.png", p_shap_global, width = 8, height = 5)

# 2. EXPLICABILIDAD LOCAL (SHAP para cuenta individual)
cat("Calculando SHAP Values para una cuenta individual...\n")
single_customer <- x_train[1, , drop = FALSE]

shap_local <- predict_parts(
  explainer = explainer_xgb,
  new_observation = single_customer,
  type = "shap"
)

p_shap_local <- plot(shap_local) +
  labs(
    title = "Explicabilidad Local (SHAP Breakdown) - Cuenta #1001",
    subtitle = "Contribución de cada variable a la probabilidad predicha"
  ) +
  theme_minimal()

ggsave("outputs/figures/04_shap_local_account.png", p_shap_local, width = 8, height = 5)

# ------------------------------------------------------------------------------
# PASO 7: EXPORTAR RESULTADOS DE EXPLICABILIDAD Y EVALUACIÓN
# ------------------------------------------------------------------------------
evaluation_bundle <- list(
  metrics_table = metrics_summary,
  psi_summary   = psi_result$psi,
  explainer     = explainer_xgb
)

output_path <- "outputs/reports/evaluation_bundle.rds"
saveRDS(evaluation_bundle, output_path)

cat(paste0("\n[PROCESO COMPLETADO] Script 04 ejecutado con éxito."))
cat(paste0("\nReporte de evaluación guardado en: ", output_path, "\n"))