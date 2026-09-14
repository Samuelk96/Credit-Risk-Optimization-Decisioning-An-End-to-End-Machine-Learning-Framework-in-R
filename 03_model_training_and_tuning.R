# ==============================================================================
# SCRIPT 03: ENTRENAMIENTO Y OPTIMIZACIÓN DE MODELOS (BASELINE VS. XGBOOST)
# Proyecto: Credit Risk Optimization & Decisioning Framework
# Autor: Gaspar Samuel Salazar Salgado
# Objetivo: 
#   1. Cargar el bundle de datos transformados a WOE del Script 02.
#   2. Entrenar un modelo de Regresión Logística (Scorecard tradicional).
#   3. Construir una receta y flujo de trabajo en tidymodels para XGBoost.
#   4. Optimizar hiperparámetros de XGBoost via Cross-Validation (k-fold CV).
#   5. Ajustar el modelo final ganador y exportar artefactos para evaluación.
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 1: CARGA DE LIBRERÍAS Y CONFIGURACIÓN
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse) # Manipulación de datos
  library(tidymodels) # Framework estándar para modelado y ML
  library(xgboost)    # Motor para Gradient Boosting
  library(vip)        # Importancia de variables
})

# Aseguramos la existencia de directorios de salida
dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# PASO 2: CARGA DE DATOS DESDE EL SCRIPT 02
# ------------------------------------------------------------------------------
data_bundle_path <- "data/processed/data_woe_bundle.rds"

if (!file.exists(data_bundle_path)) {
  stop("[ERROR] No se encuentra 'data_woe_bundle.rds'. Ejecuta el Script 02 primero.")
}

woe_bundle <- readRDS(data_bundle_path)

# Extraemos las muestras y preparamos el target como Factor (Requisito de tidymodels)
train_data <- woe_bundle$train %>%
  mutate(target = factor(target, levels = c("1", "0"))) # "1" = Bad (Evento de interés)

test_data  <- woe_bundle$test %>%
  mutate(target = factor(target, levels = c("1", "0")))

cat("=== DATOS DE ENTRENAMIENTO CARGADOS ===\n")
cat("Registros Train:", nrow(train_data), "| Variables:", ncol(train_data) - 1, "\n\n")

# ------------------------------------------------------------------------------
# PASO 3: MODELO 1 - BASELINE DE REGRESIÓN LOGÍSTICA (SCORECARD TRADICIONAL)
# ------------------------------------------------------------------------------
cat("=== ENTRENANDO MODELO BASELINE (REGRESIÓN LOGÍSTICA) ===\n")

# Especificación del modelo logístico
logit_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Ajuste directo sobre los datos transformados con WOE
logit_fit <- logit_spec %>%
  fit(target ~ ., data = train_data)

# Resumen de coeficientes
cat("\n--- Coeficientes de Regresión Logística ---\n")
print(summary(logit_fit$fit))

# ------------------------------------------------------------------------------
# PASO 4: MODELO 2 - MACHINE LEARNING AVANZADO (XGBOOST EN TIDYMODELS)
# ------------------------------------------------------------------------------
cat("\n=== CONFIGURANDO PIPELINE DE XGBOOST ===\n")

# 1. Definición de la Receta (Preprocessing Recipe)
xgb_recipe <- recipe(target ~ ., data = train_data) %>%
  step_zv(all_predictors()) # Elimina variables con varianza cero si existieran

# 2. Especificación del Modelo XGBoost dejando hiperparámetros para Tuning
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),        # Profundidad del árbol
  min_n = tune(),             # Mínimo de observaciones por nodo
  loss_reduction = tune(),    # Gamma (reducción de pérdida)
  learn_rate = tune()         # Learning rate (eta)
) %>%
  set_engine("xgboost", eval_metric = "logloss") %>%
  set_mode("classification")

# 3. Creación del Workflow
xgb_workflow <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

# ------------------------------------------------------------------------------
# PASO 5: OPTIMIZACIÓN DE HIPERPARÁMETROS (CROSS-VALIDATION TUNING)
# ------------------------------------------------------------------------------
cat("\n=== INICIANDO TUNING DE HIPERPARÁMETROS CON 5-FOLD CV ===\n")

set.seed(42)
# Configuración de 5-Fold Cross Validation estratificado
cv_folds <- vfold_cv(train_data, v = 5, strata = target)

# Malla de búsqueda aleatoria (Random Grid Search) de 15 combinaciones
xgb_grid <- grid_random(
  tree_depth(range = c(3, 8)),
  min_n(range = c(10, 50)),
  loss_reduction(range = c(-10, 1.5), trans = log10_trans()),
  learn_rate(range = c(-3, -1), trans = log10_trans()),
  size = 15
)

# Ejecución del tuning optimizando por área bajo la curva ROC (roc_auc)
tune_results <- xgb_workflow %>%
  tune_grid(
    resamples = cv_folds,
    grid = xgb_grid,
    metrics = metric_set(roc_auc, pr_auc),
    control = control_grid(save_pred = TRUE, verbose = FALSE)
  )

# Selección de la mejor combinación de hiperparámetros
best_hyperparams <- tune_results %>%
  select_best(metric = "roc_auc")

cat("\n--- Mejores Hiperparámetros Encontrados ---\n")
print(best_hyperparams)

# ------------------------------------------------------------------------------
# PASO 6: AJUSTE FINAL DEL MODELO XGBOOST (FINAL FIT)
# ------------------------------------------------------------------------------
cat("\n=== AJUSTANDO MODELO XGBOOST FINAL CON MEJORES HIPERPARÁMETROS ===\n")

# Finalizamos el workflow con la mejor combinación
final_xgb_workflow <- xgb_workflow %>%
  finalize_workflow(best_hyperparams)

# Entrenamos el modelo final con toda la muestra de Train
final_xgb_fit <- final_xgb_workflow %>%
  fit(data = train_data)

# ------------------------------------------------------------------------------
# PASO 7: GUARDADO DE ARTEFACTOS Y MODELOS
# ------------------------------------------------------------------------------
# Exportamos los modelos entrenados para consumirlos en el Script 04 (Evaluación y XAI)
models_bundle <- list(
  logit_model = logit_fit,
  xgb_model   = final_xgb_fit,
  tuning_results = tune_results,
  best_params = best_hyperparams
)

output_path <- "outputs/models/trained_models_bundle.rds"
saveRDS(models_bundle, output_path)

cat(paste0("\n[PROCESO COMPLETADO] Script 03 ejecutado con éxito."))
cat(paste0("\nModelos guardados en: ", output_path, "\n"))