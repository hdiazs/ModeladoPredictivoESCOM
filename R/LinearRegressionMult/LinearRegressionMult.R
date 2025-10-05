# Inicio del script de regresión lineal multiple

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
