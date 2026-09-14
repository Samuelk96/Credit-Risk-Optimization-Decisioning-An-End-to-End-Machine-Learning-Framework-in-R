# ==============================================================================
# SCRIPT 02: INGENIERÍA DE ATRIBUTOS, WOE & SELECCIÓN POR INFORMATION VALUE (IV)
# Proyecto: Credit Risk Optimization & Decisioning Framework
# Autor: Gaspar Samuel Salazar Salgado
# Objetivo: 
#   1. Cargar el dataset analítico limpio creado en el Script 01.
#   2. Dividir la muestra en Entrenamiento (Train), Prueba (Test) y Out-of-Time (OOT).
#   3. Realizar el agrupamiento de variables (Binning) continuas y categóricas.
#   4. Calcular el Weight of Evidence (WOE) y el Information Value (IV) por variable.
#   5. Filtrar variables predictivas según su capacidad de discriminación de riesgo.
#   6. Transformar el dataset a valores WOE y exportar para la fase de modelado.
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 1: CARGA DE LIBRERÍAS Y CONFIGURACIÓN
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse) # Manipulación de datos y visualización
  library(scorecard) # Paquete estándar en R para binning, cálculo de WOE e IV
  library(rsample)   # Para la partición robusta de muestras (Train / Test)
  library(ggplot2)   # Renderizado de gráficos de soporte
  library(scales)    # Formato de ejes y porcentajes
})

# Aseguramos la existencia de directorios de salida
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# PASO 2: CARGA DEL DATASET DESDE EL SCRIPT 01
# ------------------------------------------------------------------------------
clean_data_path <- "data/processed/data_cleaned.rds"

if (!file.exists(clean_data_path)) {
  stop("[ERROR] No se encuentra el archivo 'data_cleaned.rds'. Ejecuta primero el Script 01.")
}

df_clean <- readRDS(clean_data_path)

cat("=== DATASET CARGADO CORRECTAMENTE ===\n")
cat("Total de observaciones:", nrow(df_clean), "\n")
cat("Total de columnas:", ncol(df_clean), "\n\n")

# ------------------------------------------------------------------------------
# PASO 3: DIVISIÓN DE MUESTRAS (TRAIN, TEST Y OUT-OF-TIME / OOT)
# ------------------------------------------------------------------------------
set.seed(42) # Semilla para reproducibilidad

# 1. Separación de la muestra Out-of-Time (OOT) usando la fecha de corte
oot_cutoff_date <- max(df_clean$snapshot_date) - months(3)

df_in_time <- df_clean %>% filter(snapshot_date < oot_cutoff_date)
df_oot     <- df_clean %>% filter(snapshot_date >= oot_cutoff_date)

# 2. Partición Stratified de In-Time en Train (75%) y Test (25%)
split_strat <- initial_split(df_in_time, prop = 0.75, strata = target)
df_train    <- training(split_strat)
df_test     <- testing(split_strat)

cat("=== DISTRIBUCIÓN DE MUESTRAS ===\n")
cat("Muestra de Entrenamiento (Train):", nrow(df_train), "registros | Tasa Bad:", percent(mean(df_train$target)), "\n")
cat("Muestra de Prueba (Test)       :", nrow(df_test),  "registros | Tasa Bad:", percent(mean(df_test$target)), "\n")
cat("Muestra Fuera de Tiempo (OOT)  :", nrow(df_oot),   "registros | Tasa Bad:", percent(mean(df_oot$target)), "\n\n")

# Seleccionar solo las variables numéricas/categóricas de interés + target
variables_to_keep <- c(
  "target", "age", "monthly_income", "revolving_utilization", 
  "num_delinquencies_30_59d", "debt_ratio", "num_open_credit_lines", 
  "num_bureau_inquiries_6m", "vintage_months"
)

df_train_model <- df_train %>% select(all_of(variables_to_keep))
df_test_model  <- df_test  %>% select(all_of(variables_to_keep))
df_oot_model   <- df_oot   %>% select(all_of(variables_to_keep))

# ------------------------------------------------------------------------------
# PASO 4: AGRUPAMIENTO EN BINS (FINE & COARSE BINNING) Y CÁLCULO DE WOE
# ------------------------------------------------------------------------------
cat("=== CALCULANDO BINNING Y WEIGHT OF EVIDENCE (WOE) ===\n")

# Calculamos las reglas de agrupamiento ÚNICAMENTE sobre la muestra de Train
# para evitar fugas de información (Data Leakage) hacia Test u OOT.
bins <- woebin(
  dt = df_train_model, 
  y = "target",
  min_perc_fine = 0.02,   # Mínimo 2% de datos por bin fino
  min_perc_coarse = 0.05, # Mínimo 5% de datos por grupo final
  stop_limit = 0.1
)

# Guardamos la estructura de bins
saveRDS(bins, "data/processed/binning_rules.rds")

# ------------------------------------------------------------------------------
# PASO 5: EVALUACIÓN DEL INFORMATION VALUE (IV) Y FILTRADO DE VARIABLES
# ------------------------------------------------------------------------------
# Calculamos la tabla resumen de IV utilizando la función iv() nativa de scorecard
iv_table <- iv(df_train_model, y = "target") %>%
  as_tibble() %>%
  arrange(desc(info_value))

cat("\n=== TABLA DE INFORMATION VALUE (IV) POR VARIABLE ===\n")
print(iv_table)

# Visualización del Information Value por Variable
p_iv <- iv_table %>%
  ggplot(aes(x = reorder(variable, info_value), y = info_value, fill = info_value >= 0.02)) +
  geom_col() +
  geom_hline(yintercept = 0.02, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 0.50, linetype = "dashed", color = "darkorange", size = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#2A9D8F", "FALSE" = "#E76F51"), guide = "none") +
  labs(
    title = "Ranking de Selección por Information Value (IV)",
    subtitle = "Variables con IV >= 0.02 seleccionadas para modelado (Línea roja = Umbral mínimo)",
    x = "Atributos del Buró",
    y = "Information Value (IV)"
  ) +
  theme_minimal()

ggsave("outputs/figures/02_information_value_ranking.png", p_iv, width = 8, height = 5)

# Filtrado formal de variables que cumplen el criterio (0.02 <= IV <= 0.50)
selected_vars <- iv_table %>%
  filter(info_value >= 0.02 & info_value <= 0.50) %>%
  pull(variable)

cat("\nVariables seleccionadas para el modelo (", length(selected_vars), "de", nrow(iv_table), "):\n")
cat(paste("-", selected_vars), sep = "\n")

# ------------------------------------------------------------------------------
# PASO 6: TRANSFORMACIÓN DE LOS DATASETS A VALORES WOE (WOEBIN_PLY)
# ------------------------------------------------------------------------------
cat("\n=== TRANSFORMANDO DATASETS A VALORES WOE ===\n")

df_train_woe <- woebin_ply(df_train_model %>% select(all_of(c("target", selected_vars))), bins)
df_test_woe  <- woebin_ply(df_test_model  %>% select(all_of(c("target", selected_vars))), bins)
df_oot_woe   <- woebin_ply(df_oot_model   %>% select(all_of(c("target", selected_vars))), bins)

# ------------------------------------------------------------------------------
# PASO 7: GUARDADO DE ARTEFACTOS Y PROCESADOS
# ------------------------------------------------------------------------------
woe_data_bundle <- list(
  train = df_train_woe,
  test  = df_test_woe,
  oot   = df_oot_woe,
  selected_features = selected_vars,
  iv_summary = iv_table
)

output_path <- "data/processed/data_woe_bundle.rds"
saveRDS(woe_data_bundle, output_path)

cat(paste0("\n[PROCESO COMPLETADO] Script 02 ejecutado con éxito."))
cat(paste0("\nBundle de datos WOE guardado en: ", output_path, "\n"))