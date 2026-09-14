# ==============================================================================
# SCRIPT 05: ESTRATEGIA DE NEGOCIO, OPTIMIZACIÓN DE CUT-OFF Y SCORECARD
# Proyecto: Credit Risk Optimization & Decisioning Framework
# Autor: Gaspar Samuel Salazar Salgado
# Objetivo:
#   1. Cargar las predicciones del modelo XGBoost / Logit y métricas del Script 04.
#   2. Definir parámetros financieros de negocio (línea promedio, margen, loss severity).
#   3. Simular escenarios de Cut-off para maximizar la utilidad neta de la cartera.
#   4. Escalar las probabilidades predichas a una puntuación estándar (Scorecard 300-850).
#   5. Exportar la matriz de decisión y el reporte final de negocio.
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 1: CARGA DE LIBRERÍAS Y CONFIGURACIÓN
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse) # Manipulación de datos y visualización
  library(scorecard) # Escalamiento de score y scorecards
  library(scales)    # Formato numérico en moneda y porcentajes
  library(knitr)     # Formato de tablas integrables
})

# Asegurar directorios de salida
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/reports", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# PASO 2: CARGA DE DATOS Y PREDICCIONES
# ------------------------------------------------------------------------------
data_bundle_path   <- "data/processed/data_woe_bundle.rds"
models_bundle_path <- "outputs/models/trained_models_bundle.rds"

if (!file.exists(data_bundle_path) || !file.exists(models_bundle_path)) {
  stop("[ERROR] Faltan archivos de datos o modelos de pasos previos.")
}

woe_bundle    <- readRDS(data_bundle_path)
models_bundle <- readRDS(models_bundle_path)

test_df  <- woe_bundle$test
xgb_fit  <- models_bundle$xgb_model

# Generar dataset de decisión
test_eval <- test_df %>%
  mutate(
    target_num = as.numeric(as.character(target)),
    prob_default = predict(xgb_fit, test_df, type = "prob")$.pred_1
  )

cat("=== DATOS DE PRUEBA CARGADOS PARA OPTIMIZACIÓN DE NEGOCIO ===\n")

# ------------------------------------------------------------------------------
# PASO 3: DEFINICIÓN DE PARÁMETROS FINANCIEROS Y SIMULACIÓN DE CUT-OFF
# ------------------------------------------------------------------------------
# Supuestos de Negocio (Ajustables según el producto crediticio):
avg_ticket    <- 50000   # Línea de crédito promedio ($50,000 MXN)
interest_rate <- 0.28    # Tasa de interés promedio anual (28%)
funding_cost  <- 0.08    # Costo de fondeo + operacional (8%)
net_margin    <- interest_rate - funding_cost # Margen neto (20%)
lgd           <- 0.75    # Loss Given Default / Pérdida en caso de impago (75%)

cat("\n--- PARÁMETROS FINANCIEROS DE LA SIMULACIÓN ---\n")
cat("Monto Promedio por Crédito:", dollar(avg_ticket), "\n")
cat("Ganancia Neta por Crédito Bueno (TN):", dollar(avg_ticket * net_margin), "\n")
cat("Pérdida Neta por Crédito Malo (FN):", dollar(-avg_ticket * lgd), "\n")

# Simular evaluando umbrales de corte de 0.01 a 0.99
thresholds <- seq(0.01, 0.99, by = 0.01)

simulation_results <- map_dfr(thresholds, function(cutoff) {
  df <- test_eval %>%
    mutate(decision = ifelse(prob_default < cutoff, "Aprobado", "Rechazado"))
  
  # Matriz de confusión
  tp <- sum(df$decision == "Rechazado" & df$target_num == 1) # Malos bien rechazados
  tn <- sum(df$decision == "Aprobado"  & df$target_num == 0) # Buenos aprobados (Generan utilidad)
  fp <- sum(df$decision == "Rechazado" & df$target_num == 0) # Buenos rechazados (Costo de oportunidad)
  fn <- sum(df$decision == "Aprobado"  & df$target_num == 1) # Malos aprobados (Pérdida por default)
  
  total_accounts <- nrow(df)
  approval_rate  <- (tn + fn) / total_accounts
  bad_rate_approved <- ifelse((tn + fn) > 0, fn / (tn + fn), 0)
  
  # Utilidad Financiera Total
  profit_goods <- tn * (avg_ticket * net_margin)
  loss_bads    <- fn * (avg_ticket * lgd)
  net_profit   <- profit_goods - loss_bads
  
  tibble(
    cutoff = cutoff,
    approval_rate = approval_rate,
    bad_rate_approved = bad_rate_approved,
    profit_goods = profit_goods,
    loss_bads = loss_bads,
    net_profit = net_profit
  )
})

# Identificar el punto de corte óptimo (Máxima utilidad neta)
optimal_row <- simulation_results %>% arrange(desc(net_profit)) %>% slice(1)

cat("\n=== UMBRAL DE CORTE ÓPTIMO ENCONTRADO ===\n")
cat("Cut-off Óptimo (PD Máxima):", round(optimal_row$cutoff, 4), "\n")
cat("Tasa de Aprobación Estimada:", percent(optimal_row$approval_rate, accuracy = 0.1), "\n")
cat("Bad Rate en Aprobados:", percent(optimal_row$bad_rate_approved, accuracy = 0.1), "\n")
cat("Utilidad Neta Estimada (Test):", dollar(optimal_row$net_profit), "\n")

# ------------------------------------------------------------------------------
# PASO 4: VISUALIZACIÓN DE LA CURVA DE GANANCIA NETAS
# ------------------------------------------------------------------------------
p_profit <- ggplot(simulation_results, aes(x = cutoff, y = net_profit)) +
  geom_line(color = "#2A9D8F", linewidth = 1.2) +
  geom_vline(xintercept = optimal_row$cutoff, linetype = "dashed", color = "#E76F51", linewidth = 1) +
  annotate("text", x = optimal_row$cutoff + 0.05, y = optimal_row$net_profit * 0.9,
           label = paste0("Cut-off Óptimo: ", round(optimal_row$cutoff, 2), 
                          "\nProfit: ", dollar(optimal_row$net_profit)),
           color = "#E76F51", fontface = "bold", hjust = 0) +
  scale_y_continuous(labels = dollar_format()) +
  scale_x_continuous(labels = percent_format()) +
  labs(
    title = "Optimización del Umbral de Decisión (Cut-off)",
    subtitle = "Maximización de Utilidad Neta considerando Pérdida por Default vs. Margen",
    x = "Umbral Máximo de Probabilidad de Default Aceptada",
    y = "Utilidad Neta Total ($ MXN)"
  ) +
  theme_minimal()

ggsave("outputs/figures/05_cutoff_optimization.png", p_profit, width = 8, height = 5)

# ------------------------------------------------------------------------------
# PASO 5: ESCALAMIENTO A SCORECARD ESTÁNDAR (300 - 850 PUNTOS)
# ------------------------------------------------------------------------------
# Fórmula Estándar de Scorecard:
# Score = Target_Score + Slope * log(Odds)
# Configuración: Base Score = 600 puntos para Odds 50:1, PDO (Points to Double Odds) = 20

target_score_base <- 600
target_odds_base  <- 50
pdo               <- 20

factor <- pdo / log(2)
offset <- target_score_base - (factor * log(target_odds_base))

# Transformar probabilidades predichas a Score Crediticio
test_eval <- test_eval %>%
  mutate(
    odds = (1 - prob_default) / prob_default,
    score = round(offset + factor * log(odds)),
    score = pmax(pmin(score, 850), 300) # Acotar entre 300 y 850
  )

# Calcular el Score equivalente al Cut-off óptimo
optimal_odds <- (1 - optimal_row$cutoff) / optimal_row$cutoff
cutoff_score <- round(offset + factor * log(optimal_odds))

cat("\n=== TRADUCCIÓN A SCORECREDITICIO ===\n")
cat("Score de Corte para Aprobación (Cut-off Score):", cutoff_score, "puntos\n")
cat("Regla de Decisión: Aprobar si Score >=", cutoff_score, "puntos.\n")

# Gráfica de Distribución de Score y Punto de Corte
p_score_dist <- ggplot(test_eval, aes(x = score, fill = factor(target_num))) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = cutoff_score, linetype = "dashed", color = "black", linewidth = 1) +
  scale_fill_manual(values = c("0" = "#2A9D8F", "1" = "#E76F51"), labels = c("Bueno (0)", "Malo (1)")) +
  annotate("text", x = cutoff_score + 10, y = 0.005, 
           label = paste0("Punto de Corte: ", cutoff_score, " pts"), fontface = "bold", hjust = 0) +
  labs(
    title = "Distribución de Score Crediticio y Cut-off de Aprobación",
    subtitle = "Separación entre Clientes Buenos y Malos en la Muestra de Prueba",
    x = "Credit Score (300 - 850)",
    y = "Densidad",
    fill = "Estado"
  ) +
  theme_minimal()

ggsave("outputs/figures/05_score_distribution.png", p_score_dist, width = 8, height = 5)

# ------------------------------------------------------------------------------
# PASO 6: GUARDAR ARTEFACTOS Y RESUMEN ESTRATÉGICO
# ------------------------------------------------------------------------------
business_strategy_summary <- list(
  optimal_cutoff_prob  = optimal_row$cutoff,
  optimal_cutoff_score = cutoff_score,
  financial_simulation = simulation_results,
  scored_test_data     = test_eval
)

saveRDS(business_strategy_summary, "outputs/reports/business_strategy_summary.rds")

cat("\n[PROCESO COMPLETADO] Script 05 ejecutado exitosamente.")
cat("\nResultados de estrategia y scorecards exportados a 'outputs/reports/' y 'outputs/figures/'\n")