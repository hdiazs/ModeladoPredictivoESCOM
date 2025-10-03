# Inicio del script de regresión lineal simple

# Si no estan instalados, instalar  install.packages(c("readr","data.table","readxl","openxlsx","tidyverse","broom","car","ggplot2","caret"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)

# Lectura de conjunto de datos (dataset)
rute_csv <- "./data/raw/LinearRegressionSimple.csv"

dataframe_raw <- read_csv(rute_csv)

# En un entorno real en esta sección se realiza la limpieza y tranformación de los datos, para este ejemplo práctico los datos están limpios y completos solo se deben ordenar para un mejor trabajo

dataframe_processed <- dataframe_raw %>%
    mutate(Obs = row_number()) %>%
    rename(
        x = 'Cases Stocked, x',
        y =  'Time, y (minutes)',
    )%>%
    select(Obs, y , x)%>%
    arrange(x)


# Encabezado del conjunto de datos ordenados y limpios
head(dataframe_processed)

# Datos procesados en archivo
write.csv(dataframe_processed, "./data/processed/LinearRegressionSimple_processed.csv", row.names = FALSE)

df <- dataframe_processed

# Gráfico exploratorio de los datos

ggplot(df, aes(x = x, y = y)) +
    geom_point(color = "black", size = 3, alpha = 0.7) +
    labs(
        title = "Exploratory figure: x vs y", 
        x = "predictor, x", 
        y = "outcome, y"
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

ggsave("./reports/exploratory_graph.png",
       plot = last_plot(),
       width = 10, height = 10, units = "in", dpi = 300)

# División del conjunto de datos en entrenamiento y prueba

set.seed(42)

train_index <- createDataPartition(df$y, p = 0.8, list = FALSE)

train_data <- df[train_index, ]

test_data <- df[-train_index, ]

model <- lm(y ~ x, data = train_data)

# Muestra los datos y métricas del entrenamiento

# Intercepto y coeficientes de predictores
coef(model)

#Residuos
resid_raw <- resid(model)

ggplot(data.frame(resid = resid_raw), aes(sample = resid)) +
    stat_qq(color = 'blue', size = 2, alpha = 0.7) +
    stat_qq_line(color = 'red', linetype = 'dashed', size = 1)+
    labs(title = "QQ-plot de residuos",
       x = "Cuantiles teóricos",
       y = "Cuantiles de los residuos") +
    theme_minimal(base_size = 14) +
    theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )

    ggsave("./reports/qqplot_resid.png",
       plot = last_plot(),
       width = 10, height = 10, units = "in", dpi = 300)


# Ajuste
ggplot(train_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red", size = 1.2) +
  labs(title = "Regresión lineal: Y vs X",
       x = "X",
       y = "Y") +
  theme_minimal(base_size = 14)+
  theme(
        aspect.ratio = 1,
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90")
    )
  ggsave("./reports/fitted_plot.png",
       plot = last_plot(),
       width = 10, height = 10, units = "in", dpi = 300)

# ANOVA
anova(model)

# Resumen
summary(model)
