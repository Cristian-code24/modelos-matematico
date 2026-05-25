# ==============================================================================
#  NVIDIA AI GROWTH MODELER  v2.0
#  Proyecto: Derivadas Parciales — Cálculo Multivariado
#  Institución: Universidad Nacional José Faustino Sánchez Carrión (UNJFSC)
#  Carrera: Estadística e Informática
#
#  INSTRUCCIONES DE INSTALACIÓN (ejecutar UNA sola vez en la consola):
#    install.packages(c("shiny","plotly","shinythemes","shinyWidgets","withMathJax"))
#
#  ESTRUCTURA:
#    proyecto/
#    ├── app.R          ← este archivo
#    └── www/
#        └── logo_uni.png
# ==============================================================================

library(shiny)
library(plotly)
library(shinythemes)
library(shinyWidgets)

# ==============================================================================
#  SECCIÓN 1: DATOS HISTÓRICOS DE NVIDIA
#  x1 = Inversión en I+D          [Mill. USD]
#  x2 = Ingresos Data Center / IA [Mill. USD]
#  x3 = Inversión en IA (capex)   [Mill. USD]
#  x4 = Costos de Producción      [Mill. USD]
#  z  = Ingresos Totales          [Mill. USD]
# ==============================================================================

nvidia_data <- data.frame(
  trimestre = c("Q1 FY25","Q2 FY25","Q3 FY25","Q4 FY25",
                "Q1 FY26","Q2 FY26","Q3 FY26","Q4 FY26"),
  x1 = c(1508, 1823, 2294, 2725, 3115, 3596, 4030, 4641),
  x2 = c(22563,26272,30771,35580,39112,41096,51215,62314),
  x3 = c(3200, 3800, 4600, 5500, 6200, 7100, 8500, 10200),
  x4 = c(4100, 4800, 5500, 6200, 6900, 7500, 8800, 10400),
  z  = c(26044,30040,35082,39331,44062,46743,57006,68127)
)

# ==============================================================================
#  SECCIÓN 2: MODELO NO-LINEAL (CUADRÁTICO) — MÍNIMOS CUADRADOS
# ==============================================================================

# Ajuste robusto usando lm() para evitar el error de singularidad matricial
modelo_fit <- lm(z ~ x1 + x2 + x3 + x4 + I(x1^2) + I(x2^2) + x1:x2, data = nvidia_data)

# Extraer coeficientes de forma segura (reemplazando NA por 0)
b_raw <- coef(modelo_fit)
b_raw[is.na(b_raw)] <- 0

B0  <- b_raw["(Intercept)"]
B1  <- b_raw["x1"]
B2  <- b_raw["x2"]
B3  <- b_raw["x3"]
B4  <- b_raw["x4"]
B11 <- b_raw["I(x1^2)"]
B22 <- b_raw["I(x2^2)"]
B12 <- b_raw["x1:x2"]

# Predicciones y Métricas
z_hat <- predict(modelo_fit)
ss_res <- sum((nvidia_data$z - z_hat)^2)
ss_tot <- sum((nvidia_data$z - mean(nvidia_data$z))^2)
R2     <- if(ss_tot > 0) 1 - (ss_res / ss_tot) else 1

# Corrección de Grados de Libertad para evitar división por cero
df_res <- max(1, nrow(nvidia_data) - sum(b_raw != 0))
df_tot <- max(1, nrow(nvidia_data) - 1)
R2adj  <- 1 - (ss_res / df_res) / (ss_tot / df_tot)

# Función del modelo
f_model <- function(x1, x2, x3, x4) {
  B0 + B1*x1 + B2*x2 + B3*x3 + B4*x4 + B11*x1^2 + B22*x2^2 + B12*x1*x2
}

# Derivadas parciales en un punto
dpz_dx1 <- function(x1, x2) B1 + 2*B11*x1 + B12*x2
dpz_dx2 <- function(x1, x2) B2 + 2*B22*x2 + B12*x1
dpz_dx3 <- function()        B3
dpz_dx4 <- function()        B4

# Rangos para sliders y grilla 3D
X1_MIN <- 1200; X1_MAX <- 5500
X2_MIN <- 20000; X2_MAX <- 70000
X3_MIN <- 2500; X3_MAX <- 12000
X4_MIN <- 3500; X4_MAX <- 12000
N_GRID <- 50

# ==============================================================================
#  SECCIÓN 3: INTERFAZ DE USUARIO
# ==============================================================================

ui <- fluidPage(
  theme = shinytheme("slate"),
  
  tags$head(
    tags$link(
      href = "https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@300;400;500;600&display=swap",
      rel  = "stylesheet"
    ),
    tags$script(src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js",
                type = "text/javascript",
                async = NA),
    tags$style(HTML("
      :root {
        --g1: #76b900; --g2: #5a8f00; --g3: #3d6400;
        --dk: #080c10; --dk2: #0d1219; --dk3: #111820;
        --card: #141e2c; --card2: #192438;
        --border: #1e3048; --border2: #2a4060;
        --txt: #d8e8f0; --muted: #6a8099;
        --cyan: #00d4ff; --warn: #f5a623; --red: #ff4d6d;
        --mono: 'JetBrains Mono', monospace;
        --title: 'Oxanium', sans-serif;
        --body: 'Outfit', sans-serif;
      }
      *, *::before, *::after { box-sizing: border-box; }
      body { background: var(--dk) !important; color: var(--txt) !important; font-family: var(--body) !important; margin: 0; padding: 0; font-size: 13px; }
      .container-fluid { padding: 0 !important; }
      .hdr { background: linear-gradient(120deg, #040608 0%, #080f0a 60%, #060c04 100%); border-bottom: 2px solid var(--g1); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; min-height: 72px; position: relative; overflow: hidden; box-shadow: 0 4px 32px rgba(118,185,0,0.18); }
      .hdr::after { content:''; position: absolute; inset: 0; background: repeating-linear-gradient(90deg, transparent 0,transparent 80px, rgba(118,185,0,0.025) 80px,rgba(118,185,0,0.025) 81px); pointer-events:none; }
      .hdr-left { display:flex; align-items:center; gap:14px; z-index:1; }
      .logo-box { width:52px; height:52px; border-radius:8px; border:2px solid rgba(118,185,0,0.45); background:rgba(118,185,0,0.08); display:flex; align-items:center; justify-content:center; overflow:hidden; flex-shrink:0; }
      .logo-box img { width:100%; height:100%; object-fit:contain; padding:3px; }
      .logo-box span { font-family:var(--title); font-weight:800; font-size:.75rem; color:var(--g1); letter-spacing:1px; }
      .hdr-titles h1 { font-family:var(--title); font-size:1.55rem; font-weight:700; margin:0; letter-spacing:2.5px; color:#fff; text-shadow:0 0 24px rgba(118,185,0,0.5); }
      .hdr-titles h1 em { color:var(--g1); font-style:normal; }
      .hdr-titles p { margin:1px 0 0; font-family:var(--mono); font-size:.62rem; color:var(--muted); letter-spacing:.6px; }
      .hdr-right { display:flex; align-items:center; gap:10px; z-index:1; }
      .badge { background:rgba(118,185,0,0.1); border:1px solid rgba(118,185,0,0.35); border-radius:16px; padding:4px 12px; font-family:var(--mono); font-size:.6rem; color:var(--g1); letter-spacing:1.2px; white-space:nowrap; }
      .badge.cyan { background:rgba(0,212,255,.08); border-color:rgba(0,212,255,.3); color:var(--cyan); }
      .nv-logo { font-family:var(--title); font-weight:800; font-size:1.1rem; color:var(--g1); letter-spacing:2px; text-shadow:0 0 12px rgba(118,185,0,.6); }
      .layout { display:grid; grid-template-columns:310px 1fr; height:calc(100vh - 72px); overflow:hidden; }
      .sidebar { background:var(--dk2); border-right:1px solid var(--border); overflow-y:auto; padding:14px 12px 18px; display:flex; flex-direction:column; gap:12px; }
      ::-webkit-scrollbar { width:3px; }
      ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:2px; }
      .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:13px 14px; }
      .ct { font-family:var(--mono); font-size:.58rem; color:var(--g1); letter-spacing:2.5px; text-transform:uppercase; margin:0 0 11px; padding-bottom:7px; border-bottom:1px solid rgba(118,185,0,0.18); display:flex; align-items:center; gap:7px; }
      .ct .dot { width:6px; height:6px; background:var(--g1); border-radius:50%; box-shadow:0 0 5px var(--g1); flex-shrink:0; }
      .ct.cyan .dot { background:var(--cyan); box-shadow:0 0 5px var(--cyan); }
      .ct.cyan { color:var(--cyan); }
      .ct.warn .dot { background:var(--warn); box-shadow:0 0 5px var(--warn); }
      .ct.warn { color:var(--warn); }
      .sl-label { font-family:var(--mono); font-size:.64rem; color:var(--muted); margin-bottom:1px; letter-spacing:.3px; }
      .sl-label em { color:var(--txt); font-style:normal; font-weight:600; }
      .irs--shiny .irs-bar { background:var(--g1)!important; border:none!important; height:3px!important; }
      .irs--shiny .irs-line { background:#1a2a3a!important; border:none!important; height:3px!important; }
      .irs--shiny .irs-handle { background:var(--g1)!important; border:2px solid #fff!important; top:24px!important; width:16px!important; height:16px!important; box-shadow:0 0 8px rgba(118,185,0,.7)!important; }
      .irs--shiny .irs-from,.irs--shiny .irs-to,.irs--shiny .irs-single { background:var(--g1)!important; font-family:var(--mono)!important; font-size:.62rem!important; border-radius:3px!important; color:#000!important; font-weight:700!important; }
      .irs--shiny .irs-min,.irs--shiny .irs-max { color:var(--muted)!important; font-family:var(--mono)!important; font-size:.58rem!important; background:transparent!important; }
      .mgrid { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
      .mbox { background:var(--dk); border:1px solid var(--border); border-radius:7px; padding:9px 7px; text-align:center; position:relative; overflow:hidden; }
      .mbox::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
      .mbox.g::before { background:var(--g1); }
      .mbox.c::before { background:var(--cyan); }
      .mbox.w::before { background:var(--warn); }
      .mbox.r::before { background:var(--red); }
      .mbox.full { grid-column:1/-1; }
      .ml { font-family:var(--mono); font-size:.55rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; margin-bottom:3px; }
      .mv { font-family:var(--mono); font-size:.95rem; font-weight:700; line-height:1.2; }
      .mv.g { color:var(--g1); }
      .mv.c { color:var(--cyan); }
      .mv.w { color:var(--warn); }
      .mv.r { color:var(--red); }
      .ms { font-family:var(--mono); font-size:.55rem; color:var(--muted); margin-top:2px; }
      .meq { background:#060b12; border:1px solid #162030; border-radius:8px; padding:11px 13px; font-family:var(--mono); font-size:.65rem; line-height:1.9; color:#a0b8c8; }
      .meq .b0 { color:#ffb347; }
      .meq .b1 { color:var(--g1); }
      .meq .b2 { color:var(--cyan); }
      .meq .b3 { color:#b39ddb; }
      .meq .b4 { color:var(--red); }
      .meq .bq { color:var(--warn); }
      .meq .r2 { color:#a8ff78; }
      .dtab { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:.58rem; }
      .dtab th { background:rgba(118,185,0,.08); color:var(--g1); padding:4px 5px; text-align:center; border-bottom:1px solid var(--border); letter-spacing:.8px; }
      .dtab td { padding:3px 5px; text-align:right; color:var(--muted); border-bottom:1px solid rgba(30,48,72,.5); font-size:.57rem; }
      .dtab td:first-child { text-align:left; color:var(--txt); }
      .dtab tr:hover td { background:rgba(118,185,0,.03); color:var(--txt); }
      .right-col { display:flex; flex-direction:column; overflow:hidden; background:var(--dk); }
      .nav-tabs { background:var(--card) !important; border-bottom:1px solid var(--border) !important; padding:0 12px !important; display:flex !important; gap:4px !important; flex-shrink:0; }
      .nav-tabs > li > a { font-family:var(--mono) !important; font-size:.63rem !important; letter-spacing:1.2px !important; text-transform:uppercase !important; color:var(--muted) !important; background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; border-radius:0 !important; padding:10px 16px !important; transition:color .2s, border-color .2s; }
      .nav-tabs > li.active > a, .nav-tabs > li > a:hover { color:var(--g1) !important; border-bottom-color:var(--g1) !important; background:transparent !important; }
      .tab-content { flex:1; overflow:auto; background:var(--dk) !important; }
      .tab-pane { height:100%; }
      .math-panel { padding:20px 24px; max-width:900px; margin:0 auto; }
      .math-section { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin-bottom:16px; }
      .math-section h3 { font-family:var(--mono); font-size:.65rem; color:var(--g1); letter-spacing:2px; text-transform:uppercase; margin:0 0 14px; padding-bottom:8px; border-bottom:1px solid rgba(118,185,0,.2); }
      .math-block { background:#06080f; border-left:3px solid var(--g1); border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0; font-size:1em; overflow-x:auto; }
      .math-block.cyan { border-left-color:var(--cyan); }
      .math-block.warn { border-left-color:var(--warn); }
      .math-block.red  { border-left-color:var(--red); }
      .math-block.purple { border-left-color:#b39ddb; }
      .math-prose { font-family:var(--body); font-size:.77rem; color:#8a9db5; line-height:1.8; margin:8px 0; }
      .math-prose strong { color:var(--txt); }
      .sens-card { background:var(--dk2); border:1px solid var(--border2); border-radius:8px; padding:12px 14px; margin-top:10px; }
      .sens-row { display:grid; grid-template-columns:24px 1fr auto; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid rgba(30,48,72,.6); }
      .sens-row:last-child { border-bottom:none; }
      .sens-icon { font-size:1rem; text-align:center; }
      .sens-text { font-family:var(--body); font-size:.74rem; color:#8a9db5; line-height:1.5; }
      .sens-text strong { color:var(--txt); }
      .sens-value { font-family:var(--mono); font-size:.9rem; font-weight:700; white-space:nowrap; }
      .topbar { background:var(--card); border-bottom:1px solid var(--border); padding:7px 14px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
      .topbar h2 { font-family:var(--mono); font-size:.63rem; color:var(--muted); margin:0; letter-spacing:1.2px; text-transform:uppercase; }
      .chips { display:flex; gap:7px; }
      .chip { background:rgba(0,0,0,.3); border:1px solid var(--border); border-radius:10px; padding:2px 9px; font-family:var(--mono); font-size:.58rem; color:var(--muted); }
      .chip.g { border-color:rgba(118,185,0,.5); color:var(--g1); }
      .chip.c { border-color:rgba(0,212,255,.5); color:var(--cyan); }
      .chip.r { border-color:rgba(255,77,109,.5); color:var(--red); }
    "))
  ),
  
  div(class = "hdr",
      div(class = "hdr-left",
          div(class = "logo-box",
              tags$img(src = "logo_uni.png", alt = "UNJFSC",
                       onerror = "this.parentElement.innerHTML='<span>UNJFSC</span>'")
          ),
          div(class = "hdr-titles",
              tags$h1(HTML("NVIDIA <em>AI GROWTH</em> MODELER")),
              tags$p("Estadística e Informática — UNJFSC  |  Derivadas Parciales — Cálculo Multivariado")
          )
      ),
      div(class = "hdr-right",
          div(class = "badge", "MODELO CUADRÁTICO"),
          div(class = "badge cyan", paste0("R² = ", round(R2, 5))),
          div(class = "nv-logo", "NVIDIA")
      )
  ),
  
  div(class = "layout",
      div(class = "sidebar",
          div(class = "card",
              div(class = "ct", div(class="dot"), "Parámetros de Simulación"),
              div(class="sl-label", HTML("x₁ — <em>Inversión en I+D</em>  [Mill. USD]")),
              sliderInput("x1", NULL, X1_MIN, X1_MAX, 3000, 50, width="100%"),
              div(class="sl-label", HTML("x₂ — <em>Ingresos Data Center/IA</em>  [Mill. USD]")),
              sliderInput("x2", NULL, X2_MIN, X2_MAX, 39000, 200, width="100%"),
              div(class="sl-label", HTML("x₃ — <em>Inversión en IA (CapEx)</em>  [Mill. USD]")),
              sliderInput("x3", NULL, X3_MIN, X3_MAX, 6200, 100, width="100%"),
              div(class="sl-label", HTML("x₄ — <em>Costos de Producción</em>  [Mill. USD]")),
              sliderInput("x4", NULL, X4_MIN, X4_MAX, 6900, 100, width="100%")
          ),
          div(class = "card",
              div(class = "ct", div(class="dot"), "Modelo Cuadrático Ajustado"),
              div(class = "meq",
                  HTML(paste0(
                    "z = f(x₁,x₂,x₃,x₄) =<br>",
                    "&nbsp;<span class='b0'>",  sprintf("%.2f",B0), "</span><br>",
                    "+ <span class='b1'>",  sprintf("%.6f",B1), "</span>·x₁<br>",
                    "+ <span class='b2'>",  sprintf("%.6f",B2), "</span>·x₂<br>",
                    "+ <span class='b3'>",  sprintf("%.6f",B3), "</span>·x₃<br>",
                    "+ <span class='b4'>",  sprintf("%.6f",B4), "</span>·x₄<br>",
                    "+ <span class='bq'>",  sprintf("%.8f",B11),"</span>·x₁²<br>",
                    "+ <span class='bq'>",  sprintf("%.8f",B22),"</span>·x₂²<br>",
                    "+ <span class='bq'>",  sprintf("%.8f",B12),"</span>·x₁x₂<br><br>",
                    "<span class='r2'>R² = ", round(R2,7), " &nbsp;|&nbsp; R²adj = ", round(R2adj,7), "</span>"
                  ))
              )
          ),
          div(class = "card",
              div(class = "ct cyan", div(class="dot"), "Evaluación en f(x₀)"),
              div(class = "mgrid",
                  div(class="mbox g",
                      div(class="ml","∂z/∂x₁"),
                      div(class="mv g", textOutput("dpdx1")),
                      div(class="ms","∆ ingr. / ∆ I+D")
                  ),
                  div(class="mbox c",
                      div(class="ml","∂z/∂x₂"),
                      div(class="mv c", textOutput("dpdx2")),
                      div(class="ms","∆ ingr. / ∆ DC")
                  ),
                  div(class="mbox w",
                      div(class="ml","∂z/∂x₃"),
                      div(class="mv w", textOutput("dpdx3")),
                      div(class="ms","∆ ingr. / ∆ IA")
                  ),
                  div(class="mbox r",
                      div(class="ml","∂z/∂x₄"),
                      div(class="mv r", textOutput("dpdx4")),
                      div(class="ms","∆ ingr. / ∆ Costos")
                  ),
                  div(class="mbox warn full",
                      div(class="ml","Módulo |∇f|"),
                      div(class="mv w", textOutput("grad_mod")),
                      div(class="ms","magnitud del gradiente")
                  ),
                  div(class="mbox full", style="border-color:rgba(118,185,0,.4);",
                      div(class="ml","Ingreso Total Estimado  ẑ"),
                      div(class="mv", style="font-size:1.25rem;color:#fff;", textOutput("zpred")),
                      div(class="ms","millones de USD")
                  )
              )
          ),
          div(class = "card",
              div(class = "ct", div(class="dot"), "Serie Histórica NVIDIA"),
              tags$table(class="dtab",
                         tags$thead(tags$tr(
                           tags$th("Trim."),tags$th("I+D"),
                           tags$th("DC"),tags$th("Total")
                         )),
                         tags$tbody(lapply(seq_len(nrow(nvidia_data)), function(i) {
                           tags$tr(
                             tags$td(nvidia_data$trimestre[i]),
                             tags$td(formatC(nvidia_data$x1[i], format="d", big.mark=",")),
                             tags$td(formatC(nvidia_data$x2[i], format="d", big.mark=",")),
                             tags$td(formatC(nvidia_data$z[i],  format="d", big.mark=","))
                           )
                         }))
              )
          )
      ),
      
      div(class = "right-col",
          tabsetPanel(id = "tabs",
                      tabPanel("Superficie 3D",
                               div(class="topbar",
                                   tags$h2("Superficie de Respuesta + Curvas de Nivel + Vector Gradiente"),
                                   div(class="chips",
                                       div(class="chip c","● Datos reales"),
                                       div(class="chip r","◆ Punto simulado"),
                                       div(class="chip g","→ Gradiente ∇f")
                                   )
                               ),
                               plotlyOutput("plot3d", height="calc(100vh - 72px - 44px - 45px)", width="100%")
                      ),
                      tabPanel("Procedimiento Matemático",
                               div(class="math-panel",
                                   div(class="math-section",
                                       tags$h3("A. Modelo de Regresión No Lineal (Cuadrático)"),
                                       div(class="math-prose",
                                           "El modelo de regresión múltiple con términos cuadráticos e interacción ajustado por mínimos cuadrados ordinarios (OLS) tiene la forma:"
                                       ),
                                       div(class="math-block",
                                           uiOutput("latex_model")
                                       ),
                                       div(class="math-prose",
                                           HTML("donde los coeficientes se obtienen de la solución analítica <strong>β = (X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>z</strong>, que minimiza la suma de cuadrados de los residuos.")
                                       )
                                   ),
                                   div(class="math-section",
                                       tags$h3("B. Derivadas Parciales — Expresiones Generales"),
                                       div(class="math-prose",
                                           "Para un modelo cuadrático, las derivadas parciales NO son constantes (salvo ∂z/∂x₃ y ∂z/∂x₄). Dependen del punto evaluado:"
                                       ),
                                       div(class="math-block",        uiOutput("latex_dpx1")),
                                       div(class="math-block cyan",   uiOutput("latex_dpx2")),
                                       div(class="math-block purple", uiOutput("latex_dpx3")),
                                       div(class="math-block red",    uiOutput("latex_dpx4"))
                                   ),
                                   div(class="math-section",
                                       tags$h3("C. Evaluación Numérica en el Punto Seleccionado"),
                                       div(class="math-prose",
                                           "Sustituyendo los valores actuales de los deslizadores:"
                                       ),
                                       uiOutput("latex_eval")
                                   ),
                                   div(class="math-section",
                                       tags$h3("D. Vector Gradiente y Dirección de Máximo Crecimiento"),
                                       div(class="math-block warn", uiOutput("latex_grad")),
                                       div(class="math-prose",
                                           HTML("El gradiente <strong>∇f</strong> apunta en la dirección de <strong>máximo crecimiento</strong> de la función. Su módulo indica la tasa de cambio máxima en ese punto. Geométricamente, <strong>∇f es perpendicular a las curvas de nivel</strong> de la superficie de respuesta.")
                                       )
                                   ),
                                   div(class="math-section",
                                       tags$h3("E. Análisis de Sensibilidad — Punto por Punto"),
                                       div(class="math-prose",
                                           "Para los valores actuales de los parámetros, si cada variable aumenta <strong>una unidad ($1M)</strong> manteniendo las demás constantes:"
                                       ),
                                       uiOutput("sens_table")
                                   )
                               )
                      ),
                      tabPanel("Curvas de Nivel",
                               div(class="topbar",
                                   tags$h2("Mapa de Curvas de Nivel  z = f(x₁, x₂)  fijando x₃ y x₄"),
                                   div(class="chips",
                                       div(class="chip c","● Datos reales"),
                                       div(class="chip r","◆ Punto simulado")
                                   )
                               ),
                               plotlyOutput("plot_contour", height="calc(100vh - 72px - 44px - 45px)", width="100%")
                      )
          )
      )
  )
)

# ==============================================================================
#  SECCIÓN 4: SERVIDOR
# ==============================================================================

server <- function(input, output, session) {
  
  vals <- reactive({
    x1 <- input$x1; x2 <- input$x2; x3 <- input$x3; x4 <- input$x4
    z0  <- f_model(x1, x2, x3, x4)
    d1  <- dpz_dx1(x1, x2)
    d2  <- dpz_dx2(x1, x2)
    d3  <- dpz_dx3()
    d4  <- dpz_dx4()
    gmod <- sqrt(d1^2 + d2^2 + d3^2 + d4^2)
    list(x1=x1, x2=x2, x3=x3, x4=x4, z0=z0, d1=d1, d2=d2, d3=d3, d4=d4, gmod=gmod)
  })
  
  output$dpdx1 <- renderText({ sprintf("%.4f", vals()$d1) })
  output$dpdx2 <- renderText({ sprintf("%.4f", vals()$d2) })
  output$dpdx3 <- renderText({ sprintf("%.4f", vals()$d3) })
  output$dpdx4 <- renderText({ sprintf("%.4f", vals()$d4) })
  output$grad_mod <- renderText({ sprintf("%.4f", vals()$gmod) })
  output$zpred <- renderText({ paste0("$", formatC(vals()$z0, format="f", digits=1, big.mark=","), " M") })
  
  x1g <- seq(X1_MIN, X1_MAX, length.out = N_GRID)
  x2g <- seq(X2_MIN, X2_MAX, length.out = N_GRID)
  
  output$plot3d <- renderPlotly({
    v  <- vals()
    x3f <- v$x3; x4f <- v$x4
    
    zmat <- outer(x1g, x2g, FUN = function(a,b) f_model(a, b, x3f, x4f))
    
    g_scale <- (X1_MAX - X1_MIN) * 0.09
    gx1 <- v$d1; gx2 <- v$d2
    g_norm <- sqrt(gx1^2 + gx2^2)
    if (g_norm > 0) { gx1u <- gx1/g_norm; gx2u <- gx2/g_norm } else { gx1u<-1; gx2u<-0 }
    arrow_x1_end <- v$x1 + gx1u * g_scale
    arrow_x2_end <- v$x2 + gx2u * g_scale * (X2_MAX-X2_MIN)/(X1_MAX-X1_MIN)
    arrow_z_end  <- f_model(arrow_x1_end, arrow_x2_end, x3f, x4f)
    
    h_real <- paste0(
      "<b>", nvidia_data$trimestre, "</b><br>",
      "I+D: $", formatC(nvidia_data$x1, format="d", big.mark=","), "M<br>",
      "DC: $",  formatC(nvidia_data$x2, format="d", big.mark=","), "M<br>",
      "Total: $",formatC(nvidia_data$z,  format="d", big.mark=","), "M"
    )
    
    plot_ly() %>%
      add_surface(
        x = x1g, y = x2g, z = zmat,
        name = "Superficie f(x)",
        colorscale = list(
          list(0.00, "#020601"), list(0.20, "#091500"), list(0.40, "#1e4400"),
          list(0.65, "#4a8000"), list(0.85, "#68a800"), list(1.00, "#76b900")
        ),
        opacity = 0.80, showscale = FALSE,
        contours = list(
          z = list(show=TRUE, usecolormap=FALSE, color="rgba(118,185,0,0.65)", width=2, project=list(z=TRUE)),
          x = list(show=FALSE), y = list(show=FALSE)
        ),
        hovertemplate = paste0("<b>Superficie OLS</b><br>x₁ (I+D): $%{x:,.0f}M<br>x₂ (DC): $%{y:,.0f}M<br>ẑ: $%{z:,.0f}M<extra></extra>")
      ) %>%
      add_trace(
        type="scatter3d", mode="markers+text",
        x=nvidia_data$x1, y=nvidia_data$x2, z=nvidia_data$z,
        text=nvidia_data$trimestre, textposition="top center",
        textfont=list(color="#00d4ff", size=8, family="JetBrains Mono"),
        marker=list(size=5, color="#00d4ff", line=list(color="#fff",width=1.5), opacity=.95),
        name="Datos reales", hovertext=h_real, hovertemplate="%{hovertext}<extra></extra>"
      ) %>%
      add_trace(
        type="scatter3d", mode="markers",
        x=v$x1, y=v$x2, z=v$z0,
        marker=list(size=12, color="#ff4d6d", symbol="diamond", line=list(color="#fff",width=2.5), opacity=1),
        name="Punto simulado",
        hovertemplate=paste0("<b>◆ Simulación</b><br>x₁=$", formatC(v$x1,format="d",big.mark=","),"M<br>x₂=$", formatC(v$x2,format="d",big.mark=","),"M<br><b>ẑ=$", formatC(v$z0,format="f",digits=0,big.mark=","),"M</b><extra></extra>")
      ) %>%
      add_trace(
        type="scatter3d", mode="lines",
        x=c(v$x1,v$x1), y=c(v$x2,v$x2), z=c(min(zmat)*0.98, v$z0),
        line=list(color="#ff4d6d",width=2,dash="dot"),
        showlegend=FALSE, hoverinfo="skip"
      ) %>%
      add_trace(
        type="scatter3d", mode="lines+markers",
        x=c(v$x1, arrow_x1_end), y=c(v$x2, arrow_x2_end), z=c(v$z0, arrow_z_end),
        line=list(color="#f5a623", width=5),
        marker=list(size=c(0,9), color="#f5a623", symbol=c("circle","cone"), line=list(color="#fff",width=1)),
        name="Gradiente ∇f",
        hovertemplate=paste0("<b>Gradiente ∇f</b><br>∂z/∂x₁=",sprintf("%.4f",v$d1),"<br>∂z/∂x₂=",sprintf("%.4f",v$d2),"<br>|∇f|=",sprintf("%.4f",v$gmod),"<extra></extra>")
      ) %>%
      layout(
        paper_bgcolor="rgba(8,12,16,1)", plot_bgcolor ="rgba(8,12,16,1)",
        font=list(family="JetBrains Mono, monospace", color="#5a7080", size=10),
        scene=list(
          bgcolor="rgba(8,12,16,1)",
          xaxis=list(title=list(text="x₁ — I+D [Mill. USD]", font=list(color="#76b900",size=11,family="Oxanium")), tickfont=list(color="#3a5060",size=8,family="JetBrains Mono"), gridcolor="rgba(30,48,72,.5)", zerolinecolor="#1e3048", backgroundcolor="rgba(6,10,16,.8)", showbackground=TRUE),
          yaxis=list(title=list(text="x₂ — Data Center [Mill. USD]", font=list(color="#00d4ff",size=11,family="Oxanium")), tickfont=list(color="#3a5060",size=8,family="JetBrains Mono"), gridcolor="rgba(30,48,72,.5)", zerolinecolor="#1e3048", backgroundcolor="rgba(6,10,18,.8)", showbackground=TRUE),
          zaxis=list(title=list(text="z — Ingresos Totales [Mill. USD]", font=list(color="#d8e8f0",size=11,family="Oxanium")), tickfont=list(color="#3a5060",size=8,family="JetBrains Mono"), gridcolor="rgba(30,48,72,.5)", zerolinecolor="#1e3048", backgroundcolor="rgba(8,10,18,.8)", showbackground=TRUE),
          camera=list(eye=list(x=1.5,y=-1.5,z=0.8), center=list(x=0,y=0,z=-.1)),
          aspectratio=list(x=1.1,y=1.1,z=0.7), dragmode="turntable"
        ),
        legend=list(bgcolor="rgba(14,20,32,.92)", bordercolor="#1e3048", borderwidth=1, font=list(color="#a0b8c8",size=10,family="JetBrains Mono"), x=0.01, y=0.98),
        margin=list(l=0,r=0,t=0,b=0)
      ) %>%
      config(displayModeBar=TRUE, modeBarButtonsToRemove=c("toImage","sendDataToCloud","select3d"), displaylogo=FALSE, locale="es")
  })
  
  output$plot_contour <- renderPlotly({
    v <- vals()
    zmat2 <- outer(x1g, x2g, FUN=function(a,b) f_model(a,b,v$x3,v$x4))
    
    plot_ly() %>%
      add_contour(
        x=x1g, y=x2g, z=t(zmat2),
        contours=list(coloring="heatmap", showlabels=TRUE, labelfont=list(family="JetBrains Mono",size=9,color="#fff"), start=min(zmat2), end=max(zmat2), size=(max(zmat2)-min(zmat2))/14),
        colorscale=list(list(0.0,"#04080c"),list(0.15,"#071a04"),list(0.35,"#1a4400"),list(0.60,"#3d8000"),list(0.80,"#62a800"),list(1.0,"#76b900")),
        showscale=TRUE, colorbar=list(tickfont=list(family="JetBrains Mono",size=9,color="#6a8099"), title=list(text="z [Mill.USD]",font=list(color="#76b900",size=10)), bgcolor="rgba(14,20,32,.8)", bordercolor="#1e3048"),
        hovertemplate=paste0("x₁=$%{x:,.0f}M<br>x₂=$%{y:,.0f}M<br>ẑ=$%{z:,.0f}M<extra></extra>")
      ) %>%
      add_trace(
        type="scatter", mode="markers+text", x=nvidia_data$x1, y=nvidia_data$x2, text=nvidia_data$trimestre, textposition="top right", textfont=list(color="#00d4ff",size=8,family="JetBrains Mono"), marker=list(size=8,color="#00d4ff",line=list(color="#fff",width=1.5)), name="Datos reales", hovertemplate="<b>%{text}</b><br>x₁=$%{x:,.0f}M<br>x₂=$%{y:,.0f}M<extra></extra>"
      ) %>%
      add_trace(
        type="scatter", mode="markers", x=v$x1, y=v$x2, marker=list(size=14,color="#ff4d6d",symbol="diamond", line=list(color="#fff",width=2)), name="Punto simulado", hovertemplate=paste0("<b>Simulación</b><br>x₁=$", formatC(v$x1,format="d",big.mark=","),"M<br>x₂=$", formatC(v$x2,format="d",big.mark=","),"M<extra></extra>")
      ) %>%
      add_annotations(
        x=v$x1, y=v$x2, ax=v$x1 + v$d1/abs(v$d1+.001)*300, ay=v$x2 + v$d2/abs(v$d2+.001)*5000, xref="x", yref="y", axref="x", ayref="y", arrowhead=3, arrowsize=1.8, arrowwidth=2.5, arrowcolor="#f5a623", showarrow=TRUE, text="∇f", font=list(color="#f5a623",size=10,family="JetBrains Mono"), standoff=8
      ) %>%
      layout(
        paper_bgcolor="rgba(8,12,16,1)", plot_bgcolor ="rgba(8,12,16,1)", font=list(family="JetBrains Mono",color="#5a7080",size=10),
        xaxis=list(title=list(text="x₁ — Inversión en I+D [Mill. USD]", font=list(color="#76b900",size=11,family="Oxanium")), tickfont=list(color="#4a6070",size=9), gridcolor="rgba(30,48,72,.3)", zerolinecolor="#1e3048"),
        yaxis=list(title=list(text="x₂ — Ingresos Data Center/IA [Mill. USD]", font=list(color="#00d4ff",size=11,family="Oxanium")), tickfont=list(color="#4a6070",size=9), gridcolor="rgba(30,48,72,.3)", zerolinecolor="#1e3048"),
        legend=list(bgcolor="rgba(14,20,32,.9)",bordercolor="#1e3048", borderwidth=1,font=list(color="#a0b8c8",size=10)),
        margin=list(l=60,r=20,t=20,b=60)
      ) %>%
      config(displayModeBar=TRUE, displaylogo=FALSE, locale="es")
  })
  
  output$latex_model <- renderUI({
    withMathJax(HTML(paste0(
      "$$z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\beta_3 x_3 + \\beta_4 x_4",
      " + \\beta_{11} x_1^2 + \\beta_{22} x_2^2 + \\beta_{12} x_1 x_2$$",
      "$$\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^\\top \\mathbf{X})^{-1}\\mathbf{X}^\\top\\mathbf{z}$$",
      "$$R^2 = ", round(R2,7), "\\qquad R^2_{\\text{adj}} = ", round(R2adj,7), "$$"
    )))
  })
  
  output$latex_dpx1 <- renderUI({
    withMathJax(HTML(paste0(
      "$$\\frac{\\partial z}{\\partial x_1} = \\beta_1 + 2\\beta_{11}\\,x_1 + \\beta_{12}\\,x_2 = ",
      sprintf("%.6f",B1)," + 2(",sprintf("%.8f",B11),")x_1 + (",sprintf("%.8f",B12),")x_2$$"
    )))
  })
  output$latex_dpx2 <- renderUI({
    withMathJax(HTML(paste0(
      "$$\\frac{\\partial z}{\\partial x_2} = \\beta_2 + 2\\beta_{22}\\,x_2 + \\beta_{12}\\,x_1 = ",
      sprintf("%.6f",B2)," + 2(",sprintf("%.8f",B22),")x_2 + (",sprintf("%.8f",B12),")x_1$$"
    )))
  })
  output$latex_dpx3 <- renderUI({
    withMathJax(HTML(paste0("$$\\frac{\\partial z}{\\partial x_3} = \\beta_3 = ",sprintf("%.6f",B3),"$$")))
  })
  output$latex_dpx4 <- renderUI({
    withMathJax(HTML(paste0("$$\\frac{\\partial z}{\\partial x_4} = \\beta_4 = ",sprintf("%.6f",B4),"$$")))
  })
  
  output$latex_eval <- renderUI({
    v <- vals()
    withMathJax(HTML(paste0(
      "$$\\text{Punto:}\\quad x_1^0 = ", formatC(v$x1,format="d",big.mark=","), ",\\; x_2^0 = ", formatC(v$x2,format="d",big.mark=","), ",\\; x_3^0 = ", formatC(v$x3,format="d",big.mark=","), ",\\; x_4^0 = ", formatC(v$x4,format="d",big.mark=","), "$$",
      "$$\\frac{\\partial z}{\\partial x_1}\\bigg|_{x^0} = ", sprintf("%.6f",B1)," + 2(",sprintf("%.8f",B11),")(",formatC(v$x1,format="d"), ") + (",sprintf("%.8f",B12),")(",formatC(v$x2,format="d"), ") = \\boxed{",sprintf("%.4f",v$d1),"}$$",
      "$$\\frac{\\partial z}{\\partial x_2}\\bigg|_{x^0} = ", sprintf("%.6f",B2)," + 2(",sprintf("%.8f",B22),")(",formatC(v$x2,format="d"), ") + (",sprintf("%.8f",B12),")(",formatC(v$x1,format="d"), ") = \\boxed{",sprintf("%.4f",v$d2),"}$$",
      "$$\\frac{\\partial z}{\\partial x_3}\\bigg|_{x^0} = \\beta_3 = \\boxed{",sprintf("%.4f",v$d3),"}$$",
      "$$\\frac{\\partial z}{\\partial x_4}\\bigg|_{x^0} = \\beta_4 = \\boxed{",sprintf("%.4f",v$d4),"}$$",
      "$$\\hat{z}(x^0) = \\boxed{\\$", formatC(v$z0,format="f",digits=1,big.mark=","),"\\text{ M}}$$"
    )))
  })
  
  output$latex_grad <- renderUI({
    v <- vals()
    withMathJax(HTML(paste0(
      "$$\\nabla f(x^0) = \\begin{pmatrix}", sprintf("%.4f",v$d1)," \\\\ ", sprintf("%.4f",v$d2)," \\\\ ", sprintf("%.4f",v$d3)," \\\\ ", sprintf("%.4f",v$d4), "\\end{pmatrix}",
      "\\qquad |\\nabla f| = \\sqrt{", sprintf("%.4f",v$d1),"^2 + ", sprintf("%.4f",v$d2),"^2 + ", sprintf("%.4f",v$d3),"^2 + ", sprintf("%.4f",v$d4),"^2} = \\boxed{", sprintf("%.4f",v$gmod),"}$$"
    )))
  })
  
  output$sens_table <- renderUI({
    v <- vals()
    items <- list(
      list(icon="🟢", color="#76b900", label="<strong>Inversión en I+D (x₁)</strong>", text=paste0("Si I+D aumenta <strong>$1M</strong> (de $", formatC(v$x1,format="d",big.mark=","),"M a $", formatC(v$x1+1,format="d",big.mark=","), "M), manteniendo x₂, x₃, x₄ constantes, el ingreso total cambiará en:"), val=sprintf("%+.4f M USD",v$d1)),
      list(icon="🔵", color="#00d4ff", label="<strong>Ingresos Data Center (x₂)</strong>", text=paste0("Si Data Center aumenta <strong>$1M</strong> (de $", formatC(v$x2,format="d",big.mark=","),"M a $", formatC(v$x2+1,format="d",big.mark=","), "M), manteniendo x₁, x₃, x₄ constantes, el ingreso total cambiará en:"), val=sprintf("%+.4f M USD",v$d2)),
      list(icon="🟣", color="#b39ddb", label="<strong>Inversión IA — CapEx (x₃)</strong>", text=paste0("Si la inversión en IA aumenta <strong>$1M</strong>, manteniendo las demás constantes, el ingreso total cambiará en:"), val=sprintf("%+.4f M USD",v$d3)),
      list(icon="🔴", color="#ff4d6d", label="<strong>Costos de Producción (x₄)</strong>", text=paste0("Si los costos de producción aumentan <strong>$1M</strong>, manteniendo las demás constantes, el ingreso total cambiará en:"), val=sprintf("%+.4f M USD",v$d4))
    )
    div(class="sens-card",
        lapply(items, function(it) {
          div(class="sens-row", div(class="sens-icon", it$icon), div(class="sens-text", HTML(paste0(it$label,"<br>",it$text))), div(class="sens-value", style=paste0("color:",it$color,";"), it$val))
        })
    )
  })
}

# ==============================================================================
#  SECCIÓN 5: EJECUTAR
# ==============================================================================
shinyApp(ui = ui, server = server)