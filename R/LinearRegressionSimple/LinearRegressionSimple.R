# Inicio del script de regresión lineal simple

# Carga de librerias a utilizar en este ejemplo
library(readr)
library(dplyr)
library(ggplot2)
library(lattice)
library(caret)
library(broom)
library(rlang)

# Crear carpetas si no existen
dirs <- c("./reports", "./data/processed", "./plots", "./models")
invisible(lapply(dirs, function(d) if (!dir.exists(d)) dir.create(d, recursive = TRUE)))

# Lectura de conjunto de datos (dataset)

rute_csv <- "./data/raw/LinearRegressionSimple.csv"


dataframe_raw <- read_csv(rute_csv)

if (ncol(dataframe_raw) < 2) stop("El dataset debe tener al menos 2 columnas (y, x).")
names(dataframe_raw)[1:2] <- c("y", "x")

# En un entorno real en esta sección se realiza la limpieza y tranformación de los datos, para este ejemplo práctico los datos están limpios y completos solo se deben ordenar para un mejor trabajo

dataframe_processed <- dataframe_raw %>%
    mutate(Obs = row_number()) %>%
    select(Obs, y , x)%>%
    arrange(x)

# Encabezado del conjunto de datos ordenados y limpios
head(dataframe_processed)

# Datos procesados en archivo
write.csv(dataframe_processed, "./data/processed/LinearRegressionSimple_processed.csv", row.names = FALSE)

df <- dataframe_processed
# Gráfico exploratorio de los datos

plot_data <- ggplot(df, aes(x = x, y = y)) +
    geom_point(color = "blue", size = 3, alpha = 0.7) +
    labs(
        title = "Exploratory figure: x vs y", 
        x = "Predictor, x", 
        y = "Outcome, y"
        ) +
    theme_minimal(
        base_size = 14
    ) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

ggsave("./plots/exploratory_plot.png", plot = plot_data, width = 8, height = 8, units = "in", dpi = 300)
# División del conjunto de datos en entrenamiento y prueba

set.seed(123)

# Muestreo aleatorio

train_frac <- 0.8

n <- nrow(df)

train_index <- sample(seq_len(n), size = floor(train_frac * n), replace = FALSE)

# Al usar caret::createDataPartition (realiza estratificación en y)

# train_index <- createDataPartition(df$y, p = 0.7, list = FALSE)

train_data <- df[train_index, ]

test_data <- df[-train_index, ]

model <- lm(y ~ x, data = train_data)

# Guardar el modelo

saveRDS(model, "./models/linear_model_train.rds")

# Análisis de residuos y gráficos de residuos

resid_std <- rstandard(model)
fitted_vals <- fitted(model)

# Tabla de datos residuales
resid_df <- data.frame(Obs = train_data$Obs, fitted = fitted_vals, resid = resid(model), resid_std = resid_std, x = train_data$x)


# QQ-plot (Residuales estandarizados)
plot_qq <- ggplot(resid_df, aes(sample = resid_std))+
    stat_qq(color = "blue",  size = 3, alpha = 0.7)+
    stat_qq_line(color = "red", linetype = "dashed", size = 1)+
    labs(title = "QQ-plot standarized resids",
       x = "Theoric quantile",
       y = "Standarized resid quantile") +
    theme_minimal(base_size = 14) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

ggsave("./plots/qqplot_resid.png", plot = plot_qq, width = 8, height = 8, units = "in", dpi = 300)

# Residuales vs ajuste
plot_res_fitted <- ggplot(resid_df, aes(x = fitted, y = resid))+
    geom_point(color = "blue", size = 3, alpha = 0.7)+
    geom_hline(color = "red", yintercept = 0, linetype = "dashed") +
    labs(title = "Resids vs Fitted values",
        x = "Fitted values",
        y = "Resids") +
    theme_minimal(base_size = 14) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

ggsave("./plots/residuals_vs_fitted.png", plot = plot_res_fitted, width = 8, height = 8, units = "in", dpi = 300)

# Puntos de influencia y/o residuales grandes
cooks_dist <- cooks.distance(model)
cooks_df <- data.frame(Obs = train_data$Obs, cooks = cooks_dist)

plot_cooks <- ggplot(cooks_df, aes(x = Obs, y = cooks))+
    geom_bar(stat = "identity", color = "blue", fill = "blue", alpha=0.5)+
    labs(
        title = "Cook's distance",
        x = "Obs",
        y = "Cook's distance",
    )
    theme_minimal(base_size = 14) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

ggsave("./plots/cooks_distance.png", plot = plot_cooks, width = 10, height = 4, units = "in", dpi = 300)

plot_fit <- ggplot(train_data, aes(x = x, y = y)) +
    geom_point(color = "blue", size = 3) +
    geom_smooth(method = "lm", se = TRUE, size = 1.2, color = "red") +
    labs(
        title = "Linear regression graph (train data)",
        x = "x",
        y = "y"
    )+
    theme_minimal(base_size = 14) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

ggsave("./plots/fitted_plot.png", plot = plot_fit, width = 8, height = 8, units = "in", dpi = 300)

#  Evaluación del conjunto de prueba
pred_test <- predict(model, newdata = test_data)
test_eval <- data.frame(Obs = test_data$Obs, y_true = test_data$y, y_pred = pred_test)

# Métricas simples de la libreria caret

metrics <- postResample(pred = test_eval$y_pred, obs = test_eval$y_true)

rmse <- metrics["RMSE"]
r2_test <- metrics["Rsquared"]
mae <- metrics["MAE"]


# Las metricas también pueden calcularse manualmente de la siguiente manera:
# rmse <- sqrt(mean((test_eval$y_true - test_eval$y_pred)^2)) 
# mae <- mean(abs(test_eval$y_true - test_eval$y_pred)) 
# r2_test <- cor(test_eval$y_true, test_eval$y_pred)^2

# ---- Reporte (capturado en archivo) ----
report_file <- "./reports/reportlr.txt"
# Asegurar cierre de sink aunque ocurra error
zz <- file(report_file, open = "wt")
sink(zz)
sink(zz, type = "message")  # redirige también mensajes y warnings

on.exit({
  sink(type = "message")
  sink()
  close(zz)
}, add = TRUE)

cat("=== Descripción general del conjunto de datos ===\n\n")

# Tamaño del dataframe
cat(sprintf("Número de observaciones: %d\n", nrow(dataframe_raw)))
cat(sprintf("Número de variables: %d\n\n", ncol(dataframe_raw)))

# Nombres y tipos de variables
cat("Estructura de las variables:\n")
print(str(dataframe_raw))
cat("\n")

# Resumen estadístico
cat("Resumen estadístico de las variables numéricas:\n")
print(summary(dataframe_raw))
cat("\n")

# (Opcional) Estadísticas más detalladas con skimr
# library(skimr)
# print(skim(df))

cat("=== Fin de descripción general ===\n\n")

cat("=== Reporte de regresión lineal de entrenamiento ===\n\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("== Resumen del modelo (summary) ==\n")
print(summary(model))

cat("\n== ANOVA ==\n")
print(anova(model))

cat("\n== Coeficientes (broom::tidy) ==\n")
print(tidy(model))

cat("\n== Métricas generales ==\n")
print(glance(model))

cat("\n== Métricas en conjunto de prueba ==\n")
cat(sprintf("RMSE (test): %.5f\n", rmse))
cat(sprintf("MAE  (test): %.5f\n", mae))
cat(sprintf("R2   (test): %.5f\n", r2_test))

cat("\nArchivos generados:\n")
cat(" - Procesado CSV: ./data/processed/LinearRegressionSimple_processed.csv\n")
cat(" - Plots: ./plots/*.png\n")
cat(" - Modelo guardado: ./reports/linear_model_train.rds\n")

# Cerrar sink explícitamente (aunque on.exit se encargará)
sink(type = "message")
sink()
close(zz)
