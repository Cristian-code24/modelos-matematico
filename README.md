# 📈 NVIDIA AI Growth Modeler v2.0

> **Proyecto Académico:** Derivadas Parciales — Cálculo Multivariado  
> **Institución:** Universidad Nacional José Faustino Sánchez Carrión (UNJFSC)  
> **Carrera:** Estadística e Informática  
> **Autor:** Cristian Lucas ([Cristian-code24](https://github.com/Cristian-code24))

Este es un modelo interactivo desarrollado en **R** y **Shiny** para simular y analizar el crecimiento financiero de NVIDIA, fundamentado en conceptos de cálculo multivariado (derivadas parciales y vector gradiente) y un modelo de regresión cuadrática multivariable ajustado por mínimos cuadrados ordinarios (OLS).

---

## 🚀 Características Principales

*   **Modelo Matemático Avanzado:** Ajuste cuadrático multivariable con términos de interacción para predecir los ingresos totales ($z$) en función de la inversión en I+D ($x_1$), ingresos de Data Center/IA ($x_2$), inversión en IA (capex, $x_3$) y costos de producción ($x_4$).
*   **Visualización en 3D Interactiva:** Superficie de respuesta tridimensional que proyecta la relación entre las principales variables ($x_1, x_2, z$), mostrando los puntos históricos de datos reales de NVIDIA, el punto simulado actual y el **vector gradiente ($\nabla f$)** en tiempo real.
*   **Mapa de Curvas de Nivel:** Gráfico bidimensional interactivo con curvas de nivel donde se puede observar la perpendicularidad del vector gradiente y las líneas equipotenciales.
*   **Cálculo Analítico y Simulación:**
    *   Evaluación analítica y numérica del vector gradiente $\nabla f(x^0)$.
    *   Tasa de variación instantánea mediante derivadas parciales ($\partial z/\partial x_i$).
    *   Análisis de sensibilidad detallado (impacto de $+\$1\text{M}$ en cada variable).
*   **Ecuaciones en LaTeX:** Visualización limpia y profesional de los procedimientos y ecuaciones matemáticas renderizadas en tiempo real con MathJax.
*   **Diseño Premium:** Interfaz de usuario moderna con temática oscura (*slate*), tipografía personalizada (`Oxanium`, `Outfit`, `JetBrains Mono`), micro-animaciones, y bordes con degradados estilo NVIDIA.

---

## 🛠️ Instalación y Uso

Para ejecutar la aplicación localmente, asegúrate de tener instalado **R** y **RStudio**, luego sigue estos pasos:

### 1. Instalar dependencias (ejecutar una sola vez en la consola de R)
```R
install.packages(c("shiny", "plotly", "shinythemes", "shinyWidgets", "withMathJax"))
```

### 2. Ejecutar la aplicación
Abre el archivo `calculo.R` en tu entorno de desarrollo y presiona el botón **Run App** (o ejecuta el comando `shiny::runApp()` en la carpeta del proyecto).

---

## 📊 Estructura del Repositorio

```
├── calculo.R         # Código fuente de la aplicación Shiny (UI y Server)
├── .gitignore        # Archivo para excluir archivos temporales de R
├── README.md         # Documentación del proyecto (este archivo)
└── www/
    └── logo_uni.png  # Logo de la Universidad (UNJFSC)
```

---

## 📐 Marco Matemático

El modelo matemático ajustado por mínimos cuadrados ordinarios (OLS) está definido por:

$$z = f(x_1, x_2, x_3, x_4) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_{11} x_1^2 + \beta_{22} x_2^2 + \beta_{12} x_1 x_2$$

Donde:
*   $x_1$: Inversión en I+D.
*   $x_2$: Ingresos en Data Center / Inteligencia Artificial.
*   $x_3$: Inversión en IA (Capex).
*   $x_4$: Costos de Producción.
*   $z$: Ingresos Totales de NVIDIA.

El **Vector Gradiente ($\nabla f$)** apunta en la dirección de máximo crecimiento y está definido por:

$$\nabla f(x) = \begin{pmatrix} \frac{\partial z}{\partial x_1} \\ \frac{\partial z}{\partial x_2} \\ \frac{\partial z}{\partial x_3} \\ \frac{\partial z}{\partial x_4} \end{pmatrix}$$
