# ==============================================================================
# SCRIPT 01:INGESTA DE DATOS, LIMPIEZA Y ANÁLISIS EXPLORATORIO (EDA)
# Proyecto: Credit Risk Optimization & Decisioning Framework
# Autor: Gaspar Samuel Salazar Salgado
# Objetivo: 
#   1. Cargar la información cruda del buró/portafolio de crédito.
#   2. Definir la variable objetivo de negocio (Target / Bad Definition).
#   3. Identificar y tratar valores nulos, atípicos (outliers) e inconsistencias.
#   4. Generar diagnósticos iniciales del portafolio (EDA).
#   5. Exportar el dataset limpio (ADS) para la fase de Weight of Evidence (WOE).
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 1: CARGA DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ------------------------------------------------------------------------------
# Utilizamos suppressPackageStartupMessages para mantener la consola limpia de advertencias.
suppressPackageStartupMessages({
  library(tidyverse) # Ecosistema principal para manipulación de datos (dplyr) y gráficos (ggplot2)
  library(janitor)   # Herramienta para estandarizar nombres de columnas a formato snake_case
  library(naniar)    # Herramienta especializada en evaluar patrones de datos faltantes (missing values)
  library(skimr)     # Proporciona resúmenes estadísticos rápidos y detallados de los datasets
  library(scales)    # Ayuda a dar formato financiero (porcentajes, monedas) a los ejes de los gráficos
})

# Creamos las carpetas de salida si no existen en la estructura del proyecto.
# Esto garantiza que el script no falle al intentar guardar los resultados.
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# PASO 2: INGESTA DE DATOS Y ESTANDARIZACIÓN DE FORMATOS
# ------------------------------------------------------------------------------
# Definimos la ruta donde debe residir el archivo crudo extraído del Data Warehouse o Buró.
raw_data_path <- "data/raw/credit_bureau_data.csv"

# Lógica de respaldo: Si el archivo real aún no existe en la carpeta local, 
# el script genera automáticamente un dataset sintético realista para pruebas.
if (!file.exists(raw_data_path)) {
  
  message("[INFO] Archivo de datos no encontrado. Generando muestra sintética del buró...")
  
  set.seed(42) # Fijamos semilla para asegurar la reproducibilidad de los resultados aleatorios
  n <- 5000    # Tamaño de la muestra (5,000 cuentas de crédito)
  
  df_raw <- tibble(
    account_id = 1000 + 1:n,                                            # Identificador único de la cuenta
    age = sample(18:75, n, replace = TRUE),                              # Edad del acreditado
    monthly_income = rlnorm(n, meanlog = 8.5, sdlog = 0.8),             # Ingreso mensual (Distribución Lognormal)
    revolving_utilization = runif(n, 0, 1.2),                           # Porcentaje de uso de la línea de crédito
    num_delinquencies_30_59d = rpois(n, lambda = 0.3),                  # Conteo de moras leves (30-59 días)
    num_delinquencies_90d = rpois(n, lambda = 0.1),                     # Conteo de moras graves (>=90 días)
    debt_ratio = runif(n, 0.05, 1.5),                                   # Ratio de endeudamiento (Deuda / Ingreso)
    num_open_credit_lines = rpois(n, lambda = 6),                       # Número de líneas de crédito activas
    num_bureau_inquiries_6m = rpois(n, lambda = 1.5),                   # Inquiries/Consultas al buró en los últimos 6 meses
    vintage_months = sample(6:60, n, replace = TRUE),                   # Antigüedad de la cuenta en meses
    snapshot_date = sample(seq(as.Date('2024-01-01'),                   # Fecha de corte de la información
                               as.Date('2025-12-01'), 
                               by="month"), n, replace = TRUE)
  )
} else {
  # Lectura del archivo CSV real en caso de estar disponible
  df_raw <- read_csv(raw_data_path, show_col_types = FALSE)
}

# Estandarizamos los encabezados de las columnas (ej: "Monthly Income" -> "monthly_income").
# Esto evita errores por espacios, caracteres especiales o mayúsculas.
df_clean <- df_raw %>% 
  clean_names()


# ------------------------------------------------------------------------------
# PASO 3: DEFINICIÓN DE LA VARIABLE OBJETIVO (TARGET / BAD DEFINITION)
# ------------------------------------------------------------------------------
# En riesgo crediticio, el "Target" o variable $Y$ define qué consideramos un "Mal Cliente" (Bad).
# Regla Estándar Financiera: 
#   - Target = 1 (Bad): Si el cliente registró una mora grave de 90 días o más.
#   - Target = 0 (Good): Si el cliente ha cumplido puntualmente con sus pagos.

df_clean <- df_clean %>%
  mutate(
    target = if_else(num_delinquencies_90d > 0, 1, 0),
    target_label = if_else(target == 1, "Bad (Default)", "Good (Paid)")
  )

# Calculamos y mostramos la tasa de morosidad (Bad Rate / Class Imbalance).
# Esto es crucial para entender el desbalance de clases antes de entrenar modelos de ML.
cat("\n=== RESUMEN DE DISTRIBUCIÓN DE LA VARIABLE OBJETIVO ===\n")
target_summary <- df_clean %>%
  count(target, target_label) %>%
  mutate(porcentaje = (n / sum(n)) * 100)

print(target_summary)


# ------------------------------------------------------------------------------
# PASO 4: DIAGNÓSTICO Y TRATAMIENTO DE VALORES NULOS Y OUTLIERS
# ------------------------------------------------------------------------------
cat("\n=== DIAGNÓSTICO DE VALORES FALTANTES (MISSING VALUES) ===\n")
# Analizamos qué porcentaje de información falta en cada variable del dataset.
missing_summary <- miss_var_summary(df_clean)
print(missing_summary)

# Generamos una gráfica sobre el mapa de faltantes para documentación de auditoría.
p_missing <- vis_miss(df_clean) +
  labs(
    title = "Mapa de Diagnóstico de Datos Faltantes",
    subtitle = "Análisis de completitud en variables del Buró de Crédito"
  ) +
  theme_minimal()

# Guardamos el mapa gráfico en la carpeta de salidas
ggsave("outputs/figures/01_missing_data_pattern.png", p_missing, width = 8, height = 5)

# APLICACIÓN DE REGLAS DE TRATAMIENTO (IMPUTACIÓN Y TRUNCADO):
# 1. Utilización revolvente > 100%: En términos de negocio, sobrepasar el 100% ocurre por comisiones,
#    pero para el modelo es recomendable topearlo/truncarlo al 1.0 (100%) para estabilizar distribuciones.
# 2. Imputación de Ingresos: Los ingresos nulos se imputan temporalmente con la mediana por robustez 
#    (en el Script 02, el método WOE manejará los nulos como una categoría explícita).

df_clean <- df_clean %>%
  mutate(
    revolving_utilization = if_else(revolving_utilization > 1, 1, revolving_utilization),
    monthly_income = if_else(is.na(monthly_income), median(monthly_income, na.rm = TRUE), monthly_income)
  )


# ------------------------------------------------------------------------------
# PASO 5: ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y GENERACIÓN DE INSIGHTS
# ------------------------------------------------------------------------------
cat("\n=== RESUMEN ESTADÍSTICO GENERAL DE LAS VARIABLES ===\n")
# Imprime métricas de dispersión, media, percentiles y nulos en consola
skim(df_clean)

# GRÁFICA 1: Distribución del Ingreso Mensual según Estatus de Impago (Good vs Bad)
# Comprobamos la hipótesis de si un menor ingreso correlaciona con una mayor tasa de incumplimiento.
p_income <- df_clean %>%
  ggplot(aes(x = target_label, y = monthly_income, fill = target_label)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.alpha = 0.3) +
  scale_y_log10(labels = dollar_format()) + # Escala logarítmica para manejar la asimetría del ingreso
  scale_fill_manual(values = c("#E63946", "#1D3557")) +
  labs(
    title = "Distribución del Ingreso Mensual por Estatus de Impago",
    x = "Estatus del Acreditado",
    y = "Ingreso Mensual (Escala Logarítmica)",
    fill = "Estatus"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Guardamos la gráfica en alta definición
ggsave("outputs/figures/01_income_by_target.png", p_income, width = 7, height = 5)

# GRÁFICA 2: Tasa de Incumplimiento según Consultas Recientes al Buró (Inquiries)
# En riesgo crediticio, múltiples consultas recientes indican desesperación por liquidez o apetito de riesgo alto.
p_inquiries <- df_clean %>%
  group_by(num_bureau_inquiries_6m) %>%
  summarise(
    total_cuentas = n(),
    tasa_default = mean(target)
  ) %>%
  filter(total_cuentas > 30) %>% # Filtramos categorías con muy pocos datos para evitar ruido estadístico
  ggplot(aes(x = factor(num_bureau_inquiries_6m), y = tasa_default)) +
  geom_col(fill = "#457B9D", alpha = 0.85) +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title = "Tasa de Incumplimiento según Consultas al Buró (Últimos 6 Meses)",
    x = "Número de Consultas Recientes",
    y = "Tasa de Impago (%)"
  ) +
  theme_minimal()

# Guardamos la gráfica de consultas
ggsave("outputs/figures/01_inquiries_vs_default.png", p_inquiries, width = 7, height = 5)


# ------------------------------------------------------------------------------
# PASO 6: GUARDADO Y EXPORTACIÓN DEL DATASET ANALÍTICO (ADS)
# ------------------------------------------------------------------------------
# Guardamos el dataframe procesado en formato binario .rds nativo de R.
# Ventaja: Mantiene los tipos de datos (factores, fechas, numéricos) exactamente 
# como los procesamos, sin perder formato como ocurriría con un CSV tradicional.

output_path <- "data/processed/data_cleaned.rds"
saveRDS(df_clean, output_path)

cat(paste0("\n[PROCESO COMPLETADO] Script 01 ejecutado con éxito."))
cat(paste0("\nDataset analítico guardado en: ", output_path, "\n"))