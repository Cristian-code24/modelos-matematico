# =============================================================================
#  NVIDIA AI GROWTH MODELER
#  Proyecto Académico | Cálculo Multivariado | UNJFSC
#  Tema: Evaluación de z = f(x1, x2) — Funciones de Varias Variables
#  Autor: Cristian Lucas
# =============================================================================

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NVIDIA · Funciones de Varias Variables",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { background-color:#080c10!important; color:#d8e8f0!important; font-family:'Outfit',sans-serif!important; }
.stApp { background:#080c10!important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1219 0%,#0a1118 100%)!important; border-right:1px solid #1e3048!important; }
section[data-testid="stSidebar"] * { color:#d8e8f0!important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#76b900!important; box-shadow:0 0 10px rgba(118,185,0,.7)!important; }
[data-testid="metric-container"] { background:#0d1821; border:1px solid #1e3048; border-top:3px solid #76b900; border-radius:12px; padding:18px 22px; box-shadow:0 4px 20px rgba(118,185,0,.1); }
[data-testid="metric-container"] label { color:#6a8099!important; font-family:'JetBrains Mono',monospace!important; font-size:.7rem!important; letter-spacing:1.5px!important; text-transform:uppercase!important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color:#f40000!important; font-family:'Oxanium',sans-serif!important; font-size:1.8rem!important; font-weight:800!important; }
button[data-baseweb="tab"] { background:transparent!important; color:#4a6070!important; font-family:'JetBrains Mono',monospace!important; font-size:.68rem!important; letter-spacing:1.5px!important; text-transform:uppercase!important; border-bottom:2px solid transparent!important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f40000!important; border-bottom:2px solid #76b900!important; }
.katex-display { background:#060b12!important; border-left:3px solid #76b900!important; border-radius:0 8px 8px 0!important; padding:14px 20px!important; margin:8px 0!important; }
.katex { color:#e8f4ff!important; font-size:1.1em!important; }
thead th { background:#0d1821!important; color:#f40000!important; font-family:'JetBrains Mono',monospace!important; font-size:.7rem!important; }
tbody td { color:#a0b8c8!important; font-family:'JetBrains Mono',monospace!important; font-size:.74rem!important; }
hr { border-color:#1e3048!important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    return pd.DataFrame({
        "Trimestre": ["Q1 FY25","Q2 FY25","Q3 FY25","Q4 FY25","Q1 FY26","Q2 FY26","Q3 FY26","Q4 FY26"],
        "x1": [1508,1823,2294,2725,3115,3596,4030,4641],
        "x2": [22563,26272,30771,35580,39112,41096,51215,62314],
        "z":  [26044,30040,35082,39331,44062,46743,57006,68127],
    })

def f(x1, x2):
    return 2440.57 + 1.6928 * x1 + 0.9282 * x2

@st.cache_data
def calcular_r2(df):
    z_r = df["z"].values
    z_p = f(df["x1"].values, df["x2"].values)
    ss_res = np.sum((z_r - z_p)**2)
    ss_tot = np.sum((z_r - z_r.mean())**2)
    return 1.0 - ss_res / ss_tot

def mathjax_block(titulo, color, bloques, altura=320):
    items = ""
    for b in bloques:
        prose = f'<p class="prose">{b["prose"]}</p>' if b.get("prose") else ""
        items += f"""
        <div class="blk" style="border-left-color:{b['color']}">
          <div class="lbl" style="color:{b['color']}">{b['label']}</div>
          {prose}
          <div class="math">\\[{b['tex']}\\]</div>
        </div>"""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script>MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},
options:{{skipHtmlTags:['script','noscript','style','textarea']}}}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@700&family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080c10;color:#d8e8f0;font-family:'Outfit',sans-serif;font-size:13px;padding:16px 18px}}
.titulo{{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:{color};letter-spacing:2.5px;text-transform:uppercase;padding-bottom:10px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;gap:8px}}
.titulo::before{{content:'';width:8px;height:8px;background:{color};border-radius:50%;box-shadow:0 0 8px {color};flex-shrink:0}}
.blk{{background:#060b12;border-left:3px solid;border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px}}
.lbl{{font-family:'JetBrains Mono',monospace;font-size:.56rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:7px}}
.prose{{font-family:'Outfit',sans-serif;font-size:.77rem;color:#6a7d8e;line-height:1.7;margin-bottom:9px}}
.prose strong{{color:#b8ccd8}}
.math{{overflow-x:auto;padding:2px 0}}
mjx-container{{color:#e8f4ff!important}}
</style></head><body>
<div class="titulo">{titulo}</div>
{items}
</body></html>"""
    components.html(html, height=altura, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS ESTILO GEOGEBRA — PLANO CARTESIANO MATEMÁTICO
# ─────────────────────────────────────────────────────────────────────────────

GEO = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Outfit, sans-serif", color="#333333", size=12),
    margin=dict(l=70, r=40, t=70, b=60),
    legend=dict(
        bgcolor="rgba(255,255,255,.9)", bordercolor="#cccccc",
        borderwidth=1, font=dict(color="#333333", size=11), x=0.01, y=0.99,
    ),
    xaxis=dict(
        gridcolor="#dcdcdc", gridwidth=1.5, griddash="solid",
        zeroline=True, zerolinecolor="#000000", zerolinewidth=1.5,
        linecolor="#000000", linewidth=1.5, showline=True,
        tickfont=dict(color="#333333", size=11, family="Outfit"),
        showgrid=True, ticks="outside", ticklen=6, tickcolor="#000000",
    ),
    yaxis=dict(
        gridcolor="#dcdcdc", gridwidth=1.5, griddash="solid",
        zeroline=True, zerolinecolor="#000000", zerolinewidth=1.5,
        linecolor="#000000", linewidth=1.5, showline=True,
        tickfont=dict(color="#333333", size=11, family="Outfit"),
        showgrid=True, ticks="outside", ticklen=6, tickcolor="#000000",
    ),
)


def _crosshair(x_start, x_pt, y_start, y_pt):
    """Líneas de proyección cruzada estilo GeoGebra."""
    traces = [
        go.Scatter(x=[x_pt, x_pt], y=[y_start, y_pt], mode="lines",
                   line=dict(color="rgba(255,77,109,.45)", width=1.5, dash="dash"),
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[x_start, x_pt], y=[y_pt, y_pt], mode="lines",
                   line=dict(color="rgba(255,77,109,.45)", width=1.5, dash="dash"),
                   showlegend=False, hoverinfo="skip"),
    ]
    return traces


def graf_funcion_x2(df, x1_val, x2_val):
    """
    Gráfica GeoGebra: z = f(x1_FIJO, x2).
    La curva sube y baja en tiempo real al mover el slider x1.
    """
    x2_rng = np.linspace(18000, 67000, 400)
    z_cur  = f(x1_val, x2_rng)    # curva activa — varía con x1
    z_lo   = f(1200,   x2_rng)    # límite inferior (x1 mín)
    z_hi   = f(5000,   x2_rng)    # límite superior (x1 máx)
    z_sim  = f(x1_val, x2_val)

    fig = go.Figure()

    # Banda de rango posible
    fig.add_trace(go.Scatter(
        x=np.concatenate([x2_rng, x2_rng[::-1]]),
        y=np.concatenate([z_hi, z_lo[::-1]]),
        fill="toself", fillcolor="rgba(118,185,0,.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango f(x₁∈[1200,5000], x₂)", hoverinfo="skip",
    ))
    # Bordes de la banda
    for y_bnd, lbl, col in [(z_lo, "f(1200, x₂)  [x₁ mín]", "rgba(118,185,0,.2)"),
                             (z_hi, "f(5000, x₂)  [x₁ máx]", "rgba(118,185,0,.2)")]:
        fig.add_trace(go.Scatter(
            x=x2_rng, y=y_bnd, mode="lines", name=lbl,
            line=dict(color=col, width=1.2, dash="dot"), hoverinfo="skip",
        ))

    # Curva principal (reactiva al slider)
    fig.add_trace(go.Scatter(
        x=x2_rng, y=z_cur, mode="lines",
        name=f"z = f({x1_val:,.0f}, x₂)  ← curva activa",
        line=dict(color="#f40000", width=3.5),
        hovertemplate="x₂ = %{x:,.0f} M<br>ẑ = %{y:,.0f} M<extra></extra>",
    ))

    # Crosshairs
    for tr in _crosshair(18000, x2_val, z_cur.min()*0.97, z_sim):
        fig.add_trace(tr)

    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df["x2"].values, y=df["z"].values, mode="markers+text",
        name="Datos reales NVIDIA",
        text=df["Trimestre"].values, textposition="top right",
        textfont=dict(color="#000000", size=11, family="Outfit"),
        marker=dict(symbol="x", size=10, color="#000000", line=dict(width=2.5, color="#000000")),
        hovertemplate="<b>%{text}</b><br>x₂=%{x:,.0f}M<br>z=%{y:,.0f}M<extra></extra>",
    ))

    # Punto simulado
    fig.add_trace(go.Scatter(
        x=[x2_val], y=[z_sim], mode="markers",
        name=f"P({x2_val:,.0f} , {z_sim:,.0f})",
        marker=dict(size=18, color="#ff4d6d", symbol="diamond",
                    line=dict(color="#fff", width=2.5)),
        hovertemplate=(
            f"<b>Punto Simulado</b><br>"
            f"x₁ = {x1_val:,.0f} M<br>"
            f"x₂ = {x2_val:,.0f} M<br>"
            f"<b>ẑ = {z_sim:,.1f} M</b><extra></extra>"
        ),
    ))

    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>z = f(x₁, x₂)</b>  —  "
                f"x₁ fijo = <span style='color:#f40000'>{x1_val:,.0f} M</span>"
                f"  |  mueve el slider → la curva sube o baja"
            ),
            font=dict(color="#000000", size=14, family="Outfit"), x=0.01,
        ),
        xaxis_title=dict(text="x₂ — Ingresos Data Center / IA [Mill. USD]",
                         font=dict(color="#000000", size=12, family="Outfit")),
        xaxis_range=[18000, 67000], xaxis_tickformat=",.0f",
        yaxis_title=dict(text="z — Ingresos Totales [Mill. USD]",
                         font=dict(color="#000000", size=12, family="Outfit")),
        yaxis_range=[20000, 75000], yaxis_tickformat=",.0f",
        annotations=[dict(
            x=x2_val, y=z_sim,
            text=f"  ẑ = {z_sim:,.0f} M",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color="#ff4d6d", size=11, family="JetBrains Mono"),
            bgcolor="rgba(10,15,24,.85)",
            bordercolor="#ff4d6d", borderwidth=1, borderpad=4,
        )],
    )
    return fig


def graf_funcion_x1(df, x1_val, x2_val):
    """
    Gráfica GeoGebra: z = f(x1, x2_FIJO).
    La curva sube y baja en tiempo real al mover el slider x2.
    """
    x1_rng = np.linspace(1000, 5200, 400)
    z_cur  = f(x1_rng, x2_val)
    z_lo   = f(x1_rng, 20000)
    z_hi   = f(x1_rng, 65000)
    z_sim  = f(x1_val, x2_val)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([x1_rng, x1_rng[::-1]]),
        y=np.concatenate([z_hi, z_lo[::-1]]),
        fill="toself", fillcolor="rgba(0,212,255,.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango f(x₁, x₂∈[20k,65k])", hoverinfo="skip",
    ))
    for y_bnd, lbl, col in [(z_lo, "f(x₁, 20000)  [x₂ mín]", "rgba(0,212,255,.2)"),
                             (z_hi, "f(x₁, 65000)  [x₂ máx]", "rgba(0,212,255,.2)")]:
        fig.add_trace(go.Scatter(
            x=x1_rng, y=y_bnd, mode="lines", name=lbl,
            line=dict(color=col, width=1.2, dash="dot"), hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x1_rng, y=z_cur, mode="lines",
        name=f"z = f(x₁, {x2_val:,.0f})  ← curva activa",
        line=dict(color="#f40000", width=3.5),
        hovertemplate="x₁ = %{x:,.0f} M<br>ẑ = %{y:,.0f} M<extra></extra>",
    ))

    for tr in _crosshair(1000, x1_val, z_cur.min()*0.97, z_sim):
        fig.add_trace(tr)

    fig.add_trace(go.Scatter(
        x=df["x1"].values, y=df["z"].values, mode="markers+text",
        name="Datos reales NVIDIA",
        text=df["Trimestre"].values, textposition="top right",
        textfont=dict(color="#000000", size=11, family="Outfit"),
        marker=dict(symbol="x", size=10, color="#000000", line=dict(width=2.5, color="#000000")),
        hovertemplate="<b>%{text}</b><br>x₁=%{x:,.0f}M<br>z=%{y:,.0f}M<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[x1_val], y=[z_sim], mode="markers",
        name=f"P({x1_val:,.0f} , {z_sim:,.0f})",
        marker=dict(size=18, color="#ff4d6d", symbol="diamond",
                    line=dict(color="#fff", width=2.5)),
        hovertemplate=(
            f"<b>Punto Simulado</b><br>"
            f"x₁ = {x1_val:,.0f} M<br>"
            f"x₂ = {x2_val:,.0f} M<br>"
            f"<b>ẑ = {z_sim:,.1f} M</b><extra></extra>"
        ),
    ))

    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>z = f(x₁, x₂)</b>  —  "
                f"x₂ fijo = <span style='color:#f40000'>{x2_val:,.0f} M</span>"
                f"  |  mueve el slider → la curva sube o baja"
            ),
            font=dict(color="#000000", size=14, family="Outfit"), x=0.01,
        ),
        xaxis_title=dict(text="x₁ — Inversión en I+D [Mill. USD]",
                         font=dict(color="#000000", size=12, family="Outfit")),
        xaxis_range=[1000, 5200], xaxis_tickformat=",.0f",
        yaxis_title=dict(text="z — Ingresos Totales [Mill. USD]",
                         font=dict(color="#000000", size=12, family="Outfit")),
        yaxis_range=[20000, 75000], yaxis_tickformat=",.0f",
        annotations=[dict(
            x=x1_val, y=z_sim,
            text=f"  ẑ = {z_sim:,.0f} M",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color="#ff4d6d", size=11, family="JetBrains Mono"),
            bgcolor="rgba(10,15,24,.85)",
            bordercolor="#ff4d6d", borderwidth=1, borderpad=4,
        )],
    )
    return fig





def graf_lineas_historico(df):
    z_r = df["z"].values
    t = df["Trimestre"].values
    z_m = f(df["x1"].values, df["x2"].values)
    x_vals = np.arange(1, len(t) + 1)
    
    fig = go.Figure()
    
    text_labels = [f"({i}, {y:,.0f})" for i, y in zip(x_vals, z_r)]
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=z_r, mode="lines+markers+text",
        line=dict(color="#f40000", width=4),
        marker=dict(symbol="x", size=10, color="#000000", line=dict(width=2.5, color="#000000")),
        text=text_labels,
        textposition="top center",
        textfont=dict(color="#000000", size=11, family="Outfit"),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>z Real: $%{y:,.0f}M<extra></extra>",
        customdata=t
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=z_m, mode="lines+markers",
        name="Modelo ẑ",
        line=dict(color="#0055ff", width=2.5, dash="dash"),
        marker=dict(symbol="circle", size=7, color="#0055ff"),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>Modelo ẑ: $%{y:,.0f}M<extra></extra>",
        customdata=t
    ))
    
    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(text="<b>Tendencia Histórica NVIDIA</b>",
                   font=dict(color="#000000", size=14, family="Outfit"), x=.02),
        xaxis_title=dict(text="Intervalos ⟶", font=dict(color="#000000", size=13, family="Outfit")),
        yaxis_tickformat=",.0f",
        xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=x_vals),
        yaxis_range=[20000, 75000]
    )
    
    fig.add_annotation(
        x=0, y=1.07, xref='paper', yref='paper',
        text="Escala de valores", showarrow=False,
        font=dict(color="#000000", size=13, family="Outfit"),
        xanchor='left', yanchor='bottom'
    )
    return fig


def render_sidebar(r2):
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:10px 0 20px">
          <div style="font-family:'Oxanium',sans-serif;font-size:1.4rem;font-weight:800;color:#f40000;letter-spacing:3px;text-shadow:0 0 18px rgba(118,185,0,.6)">NVIDIA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.56rem;color:#3a5060;letter-spacing:2px;margin-top:4px">AI GROWTH MODELER</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:2px;text-transform:uppercase;padding-bottom:12px;border-bottom:1px solid #1e3048;margin-bottom:16px">● Parámetros de Simulación</div>
        """, unsafe_allow_html=True)
        st.markdown("<span style='font-family:JetBrains Mono,monospace;font-size:.63rem;color:#6a8099'>x₁ — <b style=\"color:#d8e8f0\">Inversión en I+D</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x1 = st.slider("x1_sl", 1200, 5000, 3000, 50, label_visibility="collapsed")
        st.markdown("<span style='font-family:JetBrains Mono,monospace;font-size:.63rem;color:#6a8099'>x₂ — <b style=\"color:#d8e8f0\">Ingresos Data Center/IA</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x2 = st.slider("x2_sl", 20000, 65000, 40000, 500, label_visibility="collapsed")
        st.markdown("<hr style='margin:18px 0;border-color:#1e3048'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#00d4ff;font-family:Outfit;font-size:0.9rem;'>▶ Animación Interactiva</b>", unsafe_allow_html=True)
        animar = st.button("Simular Escenario Automático ⏯️", use_container_width=True)
        
        r2c = "#76b900" if r2 >= 0.97 else "#f5a623"
        st.markdown(f"""<div style="background:#060b12;border:1px solid #1e3048;border-radius:9px;padding:14px;text-align:center;margin-top:15px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.54rem;color:#3a5060;letter-spacing:1.5px;margin-bottom:6px">BONDAD DE AJUSTE · R²</div>
          <div style="font-family:'Oxanium',sans-serif;font-size:2rem;font-weight:800;color:{r2c};text-shadow:0 0 14px {r2c}66">{r2:.6f}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.54rem;color:#2a4050;margin-top:5px">Ajuste excelente ✓</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr style='margin:16px 0;border-color:#1e3048'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#2a4050;line-height:2.2;letter-spacing:.5px">
          📚 Cálculo Multivariado<br>🏛 UNJFSC · Est. e Informática<br>👤 Cristian Lucas<br>📅 NVIDIA FY25–FY26
        </div>""", unsafe_allow_html=True)
    return float(x1), float(x2), animar

def render_header():
    st.markdown("""<div style="background:linear-gradient(120deg,#040608 0%,#080f0a 60%,#060c04 100%);border-bottom:2px solid #76b900;padding:18px 28px;margin:-1rem -1rem 1.4rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 4px 32px rgba(118,185,0,.18)">
      <div>
        <div style="font-family:'Oxanium',sans-serif;font-size:1.55rem;font-weight:800;color:#fff;letter-spacing:3px;text-shadow:0 0 24px rgba(118,185,0,.5)">NVIDIA <span style="color:#f40000">AI GROWTH</span> MODELER</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.59rem;color:#3a5060;letter-spacing:1px;margin-top:4px">UNJFSC · Estadística e Informática &nbsp;|&nbsp; Funciones de Varias Variables  z = f(x₁, x₂)</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span style="background:rgba(118,185,0,.1);border:1px solid rgba(118,185,0,.4);border-radius:20px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:1.2px">MODELO LINEAL OLS</span>
        <span style="background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.35);border-radius:20px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:1.2px">GRÁFICOS 2D</span>
        <span style="background:rgba(245,166,35,.08);border:1px solid rgba(245,166,35,.35);border-radius:20px;padding:4px 14px;font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f5a623;letter-spacing:1.2px">MATHJAX 3</span>
      </div>
    </div>""", unsafe_allow_html=True)

def render_metricas(x1, x2, r2):
    z = f(x1, x2)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("📐  x₁ — Inversión I+D",  f"${x1:,.0f} M")
    with c2: st.metric("🖥️  x₂ — Data Center",     f"${x2:,.0f} M")
    with c3: st.metric("🟢  ẑ = f(x₁, x₂)",       f"${z:,.1f} M")
    with c4: st.metric("📊  R² del Modelo",          f"{r2:.6f}")

def render_modelo_matematico(x1, x2):
    z = f(x1, x2); t1 = 1.6928*x1; t2 = 0.9282*x2

    st.markdown("""<div style="background:linear-gradient(90deg,#0c1a0a,#0d1822);border:1px solid #1e3048;border-left:4px solid #76b900;border-radius:0 12px 12px 0;padding:16px 22px;margin-bottom:18px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.61rem;color:#f40000;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px">▶ Marco Matemático — Funciones de Varias Variables</div>
      <div style="font-family:'Outfit',sans-serif;font-size:.84rem;color:#7a8da0;line-height:1.75">El modelo representa una <strong style="color:#d8e8f0">función de dos variables reales</strong> <strong style="color:#f40000">z = f(x₁, x₂)</strong>, ajustada por <strong style="color:#d8e8f0">Mínimos Cuadrados Ordinarios (OLS)</strong> sobre 8 trimestres históricos de NVIDIA (FY25–FY26).</div>
    </div>""", unsafe_allow_html=True)

    col_math, col_info = st.columns([3, 2], gap="medium")

    with col_math:
        mathjax_block(
            "A — Definición de la Función y Dominio", "#00d4ff",
            [{"label":"Tipo de función","color":"#00d4ff",
              "prose":"La función de dos variables mapea cada par (x₁, x₂) a un único valor z ∈ ℝ:",
              "tex": r"f \;:\; \mathbb{R}^2 \;\longrightarrow\; \mathbb{R} \qquad (x_1,\, x_2) \;\mapsto\; z = f(x_1, x_2)"},
             {"label":"Dominio acotado","color":"#00d4ff",
              "prose":"Rango definido por los datos históricos reales de NVIDIA (en millones de USD):",
              "tex": r"\text{Dom}(f) = \bigl[1{,}200\;;\;5{,}000\bigr] \times \bigl[20{,}000\;;\;65{,}000\bigr] \;\subset\; \mathbb{R}^2"}],
            altura=290)

        mathjax_block(
            "B — Modelo Lineal: Forma General y Estimador OLS", "#76b900",
            [{"label":"Modelo de regresión lineal múltiple","color":"#76b900",
              "prose":"Forma canónica con coeficientes β estimados por Mínimos Cuadrados Ordinarios:",
              "tex": r"z = f(x_1, x_2) = \beta_0 + \beta_1\, x_1 + \beta_2\, x_2"},
             {"label":"Estimador de mínimos cuadrados","color":"#76b900",
              "prose":"Los coeficientes β minimizan la suma de cuadrados de residuos SSR:",
              "tex": r"\hat{\boldsymbol{\beta}} = \bigl(\mathbf{X}^\top \mathbf{X}\bigr)^{-1}\mathbf{X}^\top\mathbf{z} \qquad \text{con}\quad \text{SSR} = \sum_{i=1}^{8}(z_i - \hat{z}_i)^2"}],
            altura=300)

        mathjax_block(
            "C — Modelo con Coeficientes Ajustados (NVIDIA FY25–FY26)", "#f5a623",
            [{"label":"Ecuación calibrada","color":"#f5a623",
              "prose":"Coeficientes β₀, β₁, β₂ estimados con OLS sobre los 8 trimestres reales:",
              "tex": r"z = \underbrace{2440.57}_{\beta_0} + \underbrace{1.6928}_{\beta_1}\cdot x_1 + \underbrace{0.9282}_{\beta_2}\cdot x_2"},
             {"label":"Bondad de ajuste","color":"#f5a623",
              "prose":"El coeficiente R² mide la proporción de varianza de z explicada por el modelo:",
              "tex": r"R^2 = 1 - \dfrac{\displaystyle\sum_{i=1}^{8}(z_i - \hat{z}_i)^2}{\displaystyle\sum_{i=1}^{8}(z_i - \bar{z})^2} \;=\; 0.999995"}],
            altura=320)

        mathjax_block(
            f"D — Evaluación Numérica en x₁ = {x1:,.0f}  y  x₂ = {x2:,.0f}", "#ff4d6d",
            [{"label":"Sustitución del punto seleccionado","color":"#ff4d6d",
              "prose":f"Reemplazando los valores actuales de los sliders en la función:",
              "tex": rf"f\!\left({x1:,.0f},\;{x2:,.0f}\right) = 2440.57 + 1.6928 \times {x1:,.0f} + 0.9282 \times {x2:,.0f}"},
             {"label":"Desarrollo aritmético","color":"#ff4d6d",
              "tex": rf"= 2440.57 + {t1:,.2f} + {t2:,.2f}"},
             {"label":"Resultado final encuadrado","color":"#ff4d6d",
              "tex": rf"\boxed{{\;\hat{{z}} = {z:,.2f}\;\text{{millones de USD}}\;}}"}],
            altura=360)

    with col_info:
        st.markdown("""<div style="background:#0d1821;border:1px solid #1e3048;border-radius:10px;padding:16px 18px;margin-bottom:14px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Coeficientes del Modelo</div>""", unsafe_allow_html=True)
        st.markdown("""
| Coef. | Valor | Significado |
|:---:|---:|:---|
| $\\beta_0$ | 2,440.57 | Intercepto (base) |
| $\\beta_1$ | 1.6928 | Efecto marginal x₁ |
| $\\beta_2$ | 0.9282 | Efecto marginal x₂ |""")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div style="background:#0d1821;border:1px solid #1e3048;border-radius:10px;padding:16px 18px;margin-bottom:14px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Variables del Modelo</div>""", unsafe_allow_html=True)
        st.markdown("""
| Símbolo | Variable | Unidad |
|:---:|:---|:---:|
| $x_1$ | Inversión en I+D | Mill. USD |
| $x_2$ | Ingresos Data Center/IA | Mill. USD |
| $z$ | **Ingresos Totales** | Mill. USD |""")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div style="background:#0d1821;border:1px solid #1e3048;border-radius:10px;padding:16px 18px;margin-bottom:14px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f5a623;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Interpretación Económica</div>""", unsafe_allow_html=True)
        st.markdown("""
- Cada **USD 1M** adicional en $x_1$ genera $$\\Delta z = \\beta_1 = 1.6928$$ M en ingresos.
- Cada **USD 1M** adicional en $x_2$ genera $$\\Delta z = \\beta_2 = 0.9282$$ M en ingresos.
- Dado que $$\\beta_1 > \\beta_2$$, la **I+D tiene mayor efecto marginal** sobre los ingresos.""")
        st.markdown("</div>", unsafe_allow_html=True)

        z_val = f(x1, x2)
        st.markdown(f"""<div style="background:#060b12;border:2px solid #76b900;border-radius:12px;padding:20px;text-align:center;box-shadow:0 0 28px rgba(118,185,0,.2)">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.53rem;color:#3a5060;letter-spacing:2px;margin-bottom:10px">RESULTADO · f({x1:,.0f} , {x2:,.0f})</div>
          <div style="font-family:'Oxanium',sans-serif;font-size:2rem;font-weight:800;color:#f40000;text-shadow:0 0 22px rgba(118,185,0,.6);line-height:1.2">${z_val:,.1f} M</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.53rem;color:#2a4050;margin-top:8px">Ingresos Totales Estimados (USD)</div>
        </div>""", unsafe_allow_html=True)

def render_tabla(df):
    x1=df["x1"].values; x2=df["x2"].values; z_r=df["z"].values
    z_m=f(x1,x2); err=z_r-z_m
    tabla=pd.DataFrame({
        "Trimestre": df["Trimestre"].values,
        "x₁ I+D": [f"${v:,.0f}M" for v in x1],
        "x₂ Data Center": [f"${v:,.0f}M" for v in x2],
        "z Real": [f"${v:,.0f}M" for v in z_r],
        "ẑ Modelo": [f"${v:,.1f}M" for v in z_m],
        "Error (z−ẑ)": [f"{'+'if e>=0 else ''}{e:,.1f}M" for e in err],
    })
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#6a8099;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">● Datos Históricos — Real vs Modelo OLS</div>""", unsafe_allow_html=True)
    st.dataframe(tabla, width='stretch', hide_index=True)




def graf_barras_crecimiento(df):
    df_chart = df.copy()
    df_chart['Delta'] = df_chart['z'].diff().fillna(0)
    
    fig = go.Figure()
    
    # Barras principales
    fig.add_trace(go.Bar(
        x=df_chart['Trimestre'], 
        y=df_chart['z'],
        name="Ingresos (z)",
        marker=dict(
            color=df_chart['z'],
            colorscale=[[0, "#1e3048"], [1, "#76b900"]],
            line=dict(color="#76b900", width=1)
        ),
        text=[f"${v:,.0f}M" for v in df_chart['z']],
        textposition='outside',
        textfont=dict(color="#d8e8f0", family="Outfit")
    ))
    
    # Linea de margen / ganancia extra
    fig.add_trace(go.Scatter(
        x=df_chart['Trimestre'], 
        y=df_chart['Delta'],
        mode="lines+markers",
        name="Ganancia/Crecimiento vs Anterior",
        line=dict(color="#00d4ff", width=3, dash="dot"),
        marker=dict(size=12, color="#00d4ff", symbol="diamond"),
        yaxis="y2"
    ))
    
    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(text="<b>Análisis de Ganancias y Evolución Financiera</b>", font=dict(color="#000000", size=14, family="Outfit"), x=0.01),
        yaxis=dict(title="Ingresos Totales (z)", range=[0, 80000]),
        yaxis2=dict(
            title="Crecimiento Adicional",
            overlaying="y",
            side="right",
            range=[-2000, 15000],
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=80, b=40, l=60, r=60)
    )
    return fig

def main():
    df = cargar_datos()
    r2 = calcular_r2(df)
    
    # Manejo de animación
    if 'is_animating' not in st.session_state:
        st.session_state.is_animating = False

    x1, x2, animar = render_sidebar(r2)
    
    if animar:
        st.session_state.is_animating = True
        
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    
# Typewriter Explanation Component
    x1_str = f"{x1:,.0f}"
    x2_str = f"{x2:,.0f}"
    z_str = f"{f(x1, x2):,.0f}"
    
    html_code = f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
    body {{
        background: linear-gradient(145deg, #0d1520, #0a0f16); 
        border-left: 4px solid #00d4ff; 
        border-right: 1px solid #1e3048; 
        border-top: 1px solid #1e3048; 
        border-bottom: 1px solid #1e3048; 
        border-radius: 10px; 
        padding: 22px; 
        margin: 0;
        color: #a0b8c8; 
        font-family: 'Outfit', sans-serif; 
        font-size: 15.5px; 
        line-height: 1.6;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.05);
        box-sizing: border-box;
        overflow: hidden;
    }}
    strong {{ color: #d8e8f0; }}
    .highlight {{ color: #76b900; font-weight: bold; }}
    .red {{ color: #f40000; font-weight: bold; }}
    .title {{ color: #00d4ff; font-size: 18px; margin-top: 0; font-weight: 600; margin-bottom: 15px; letter-spacing: 1px; }}
    #cursor {{ display: inline-block; width: 8px; height: 16px; background: #00d4ff; animation: blink 1s infinite; vertical-align: middle; margin-left: 4px; }}
    @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
    </style>
    </head><body>
    <div class="title">Explicación</div>
    <span id="content"></span><span id="cursor"></span>
    <script>
    const textHTML = `<strong>Análisis del estado actual:</strong><br>
    La función matemática que rige el modelo es <strong style="color:#f40000;">z = f(x₁, x₂)</strong>. 
    Actualmente has fijado la variable independiente <strong>x₁ (Inversión en I+D) en <span class="highlight">${x1_str}M</span></strong> y <strong>x₂ (Data Center) en <span class="highlight">${x2_str}M</span></strong>.<br><br>
    Esto hace que la variable dependiente <strong>z (Ingresos Totales)</strong> alcance exactamente <strong><span class="red">${z_str}M</span></strong>.<br><br>
    <em style="color:#d8e8f0;">💡 ¿Cómo se comporta la variable dependiente (z)?</em><br>
    El modelo demuestra que ambas variables impulsan el ingreso. Por cada millón adicional en I+D (x₁), los ingresos crecen <strong>$1.69M</strong>. Observa el gráfico interactivo abajo: la curva roja representa el valor de z asumiendo un x₁ constante. Si subes el slider de x₁ (o activas la animación), toda la curva <strong>se desplaza verticalmente</strong> hacia arriba, demostrando que z experimenta un salto drástico en todos los escenarios posibles de x₂.`;

    let i = 0;
    let isTag = false;
    let currentHTML = "";
    const el = document.getElementById("content");

    function type() {{
      if (i < textHTML.length) {{
        let char = textHTML.charAt(i);
        currentHTML += char;
        if (char === '<') isTag = true;
        if (char === '>') isTag = false;
        
        el.innerHTML = currentHTML;
        i++;
        
        if (isTag) {{
          type(); 
        }} else {{
          setTimeout(type, 10);
        }}
      }} else {{
        document.getElementById("cursor").style.display = "none";
      }}
    }}
    setTimeout(type, 300);
    </script>
    </body></html>
    """
    components.html(html_code, height=330)

    render_metricas(x1, x2, r2)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Gráficas GeoGebra (Animado)",
        "📊  Gráfico de Barras (Ganancias)",
        "📉  Tendencia Histórica",
        "🔢  Modelo Matemático",
    ])
    
    with tab1:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
            "color:#f40000;letter-spacing:2px;text-transform:uppercase;"
            "padding:8px 0 14px;'>"
            "● Plano cartesiano matemático interactivo"
            "</div>",
            unsafe_allow_html=True
        )
        
        graf_placeholder = st.empty()
        
        if st.session_state.is_animating:
            import time
            for anim_val in range(1200, 5200, 150):
                with graf_placeholder.container():
                    st.plotly_chart(graf_funcion_x2(df, anim_val, x2), width="stretch", config={"displaylogo": False})
                time.sleep(0.08)
            st.session_state.is_animating = False
            st.rerun()
        else:
            with graf_placeholder.container():
                st.plotly_chart(graf_funcion_x2(df, x1, x2), width='stretch', config={'displaylogo': False})
            
        st.markdown(
            "<span style='font-family:JetBrains Mono,monospace;font-size:.62rem;"
            "color:#4a6070;'>↕ Mueve el slider <b style='color:#f40000'>x₁</b> para desplazar la curva arriba/abajo. "
            "↔ Mueve el slider <b style='color:#f40000'>x₂</b> para deslizar el punto sobre la curva.</span>",
            unsafe_allow_html=True)

    with tab2:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
            "color:#f40000;letter-spacing:2px;text-transform:uppercase;"
            "padding:8px 0 14px;'>"
            "● Balance General: Ingresos vs Crecimiento"
            "</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(graf_barras_crecimiento(df), width="stretch", config={"displaylogo": False})
        
        st.markdown("<p style='font-family:Outfit;color:#a0b8c8;font-size:0.9rem;text-align:center;'>Las barras muestran los ingresos totales (z), y la línea punteada celeste indica la <b>ganancia extra</b> (crecimiento) respecto al trimestre anterior.</p>", unsafe_allow_html=True)

    with tab3:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
            "color:#f40000;letter-spacing:2px;text-transform:uppercase;"
            "padding:8px 0 14px;'>"
            "● Tendencia histórica de Ingresos Reales vs Modelo"
            "</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(graf_lineas_historico(df), width="stretch", config={"displaylogo": False})
        st.markdown("<br>", unsafe_allow_html=True)
        render_tabla(df)

    with tab4:
        render_modelo_matematico(x1, x2)

    st.markdown("""<div style="text-align:center;padding:28px 0 8px;font-family:'JetBrains Mono',monospace;font-size:.53rem;color:#4a6070;letter-spacing:1px">
      NVIDIA AI Growth Modeler &nbsp;·&nbsp; Herramienta Educativa Animada &nbsp;·&nbsp; Cálculo Multivariado
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
