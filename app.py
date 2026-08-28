import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import statsmodels.api as sm

st.set_page_config(page_title="NVIDIA AI Growth Modeler", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. GLOBAL DATA & STATSMODELS OLS
# ==========================================
df_global = pd.DataFrame({
    "Trimestre": [
        "FY25 Q1 (Abr 24)", "FY25 Q2 (Jul 24)", "FY25 Q3 (Oct 24)", "FY25 Q4 (Ene 25)",
        "FY26 Q1 (Abr 25)", "FY26 Q2 (Jul 25)", "FY26 Q3 (Oct 25)", "FY26 Q4 (Ene 26)"
    ],
    "x1": [2728, 3090, 3200, 3400, 3600, 3800, 4000, 4200],
    "x2": [22563, 26272, 28000, 31000, 34000, 37000, 40000, 44000],
    "z": [26044, 30040, 32500, 36000, 39500, 43000, 46500, 51000]
})

X_global = sm.add_constant(df_global[["x1", "x2"]])
ols_model = sm.OLS(df_global["z"], X_global).fit()
BETA0, BETA1, BETA2 = ols_model.params
R2_OLS = ols_model.rsquared

def f(x1, x2):
    return BETA0 + BETA1 * x1 + BETA2 * x2

# ==========================================
# 2. THEME & STYLES ENGINE
# ==========================================
def get_theme(is_dark):
    if is_dark:
        return {
            "name": "dark",
            "bg1": "#040608", "bg2": "#0d1520",
            "sidebar": "#060b12",
            "text": "#d8e8f0", "text_dim": "#7a8da0",
            "grid": "#1e3048", "zero": "#4a6070",
            "c_green": "#76b900", "c_red": "#ff4d6d", "c_blue": "#00d4ff",
            "card_bg": "#0a1118", "card_border": "#1e3048",
            "three_bg": "0x060b12", "three_grid": "0x1e3048"
        }
    else:
        return {
            "name": "light",
            "bg1": "#fcfdfd", "bg2": "#eef2f5",
            "sidebar": "#ffffff",
            "text": "#1a202c", "text_dim": "#4a5568",
            "grid": "#e2e8f0", "zero": "#cbd5e1",
            "c_green": "#2e7d32", "c_red": "#d32f2f", "c_blue": "#1976d2",
            "card_bg": "#ffffff", "card_border": "#cbd5e1",
            "three_bg": "0xffffff", "three_grid": "0xcccccc"
        }

def get_geo(T):
    return dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"], family="Outfit"),
        margin=dict(l=50, r=40, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor=T["grid"], zeroline=True, zerolinecolor=T["zero"], zerolinewidth=1.5,
                   tickfont=dict(color=T["text_dim"], size=11, family="JetBrains Mono")),
        yaxis=dict(showgrid=True, gridcolor=T["grid"], zeroline=True, zerolinecolor=T["zero"], zerolinewidth=1.5,
                   tickfont=dict(color=T["text_dim"], size=11, family="JetBrains Mono")),
    )

def inject_custom_css(T):
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&family=Oxanium:wght@400;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; color: {T["text"]}; }}
    
    /* Background Gradient Magic */
    .stApp {{
        background: linear-gradient(135deg, {T["bg1"]} 0%, {T["bg2"]} 100%) !important;
        background-attachment: fixed !important;
    }}
    
    [data-testid="stSidebar"] {{ background-color: {T["sidebar"]}; border-right: 1px solid {T["card_border"]}; box-shadow: 2px 0 10px rgba(0,0,0,0.05); }}
    .stSlider > div > div > div > div {{ background-color: {T["c_red"]}; }}
    .stSlider > div > div > div > div > div {{ background-color: {T["c_red"]}; border: 2px solid #fff; box-shadow: 0 0 10px {T["c_red"]}88; }}
    h1, h2, h3 {{ font-family: 'Oxanium', sans-serif; font-weight: 700; color: {T["text"]}; }}
    div[data-testid="stMetricValue"] {{ font-family: 'Oxanium', sans-serif; font-weight: 700; color: {T["text"]}; font-size: 1.8rem; }}
    div[data-testid="stMetricLabel"] {{ font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {T["text_dim"]}; letter-spacing: 1px; font-weight: 700; }}
    
    .stButton>button {{ background: {T["card_bg"]}; border: 1px solid {T["card_border"]}; color: {T["text"]}; font-family: 'JetBrains Mono'; border-radius: 6px; transition: all 0.3s ease; font-weight: 700; }}
    .stButton>button:hover {{ background: {T["c_blue"]}15; border-color: {T["c_blue"]}; color: {T["c_blue"]}; box-shadow: 0 0 12px {T["c_blue"]}44; }}
    
    .stTabs [data-baseweb="tab-list"] {{ background-color: transparent; border-bottom: 2px solid {T["card_border"]}; gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ color: {T["text_dim"]}; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 12px 16px; border: 1px solid transparent; border-radius: 8px 8px 0 0; transition: all 0.3s ease; font-weight: 600; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: {T["c_green"]}; border: 1px solid {T["c_green"]}; border-bottom: none; background: {T["c_green"]}10; text-shadow: 0 0 10px {T["c_green"]}33; }}
    
    [data-testid="metric-container"] {{ background: {T["card_bg"]}; border: 1px solid {T["card_border"]}; border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); }}
    </style>""", unsafe_allow_html=True)

def mathjax_block(titulo, color, lineas, T, altura=150):
    html = f"""<!DOCTYPE html><html><head>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
      body {{ margin:0; padding:16px; background: transparent; color: {T["text"]}; font-family: 'Outfit', sans-serif; font-size: 15px; overflow: hidden; }}
      .box {{ border-left: 3px solid {color}; padding-left: 14px; margin-bottom: 12px; }}
      .title {{ font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {color}; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; font-weight: 700; }}
      .mj-container {{ background: rgba(0,0,0,0.03); padding: 10px; border-radius: 6px; border: 1px solid {T["card_border"]}; margin-top: 6px; text-align: center; overflow-x: auto; overflow-y: hidden; }}
      .prose {{ color: {T["text_dim"]}; font-size: 14px; line-height: 1.5; }}
    </style></head><body>
    <div class="box"><div class="title">{titulo}</div>"""
    for l in lineas:
        if "prose" in l: html += f'<div class="prose">{l["prose"]}</div>'
        html += f'<div class="mj-container">\\({l["tex"]}\\)</div>'
    html += "</div></body></html>"
    components.html(html, height=altura)

# ==========================================
# 3. PLOTS
# ==========================================
def graf_funcion_x2(df, x1_val, x2_val, T):
    x_range = np.linspace(20000, 65000, 50)
    z_curve = f(x1_val, x_range)
    z_sim = f(x1_val, x2_val)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_range, y=z_curve, mode="lines",
        name=f"z = f({x1_val:,.0f}, x₂)",
        line=dict(color=T["c_red"], width=4),
        fill="tozeroy", fillcolor=f"{T['c_red']}15" # 15 opacity hex
    ))
    
    z_proj = f(x1_val, df["x2"].values)
    fig.add_trace(go.Scatter(
        x=df["x2"], y=z_proj, mode="markers",
        name="Trimestres Proyectados",
        marker=dict(symbol="circle-open", size=8, color=T["c_blue"], line_width=2),
        hovertemplate="x₂: %{x:,.0f}<br>z Proy: %{y:,.0f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[x2_val], y=[z_sim], mode="markers",
        name="Punto Actual",
        marker=dict(size=18, color=T["c_red"], symbol="diamond", line=dict(color="#fff", width=2.5)),
        hovertemplate=f"<b>Punto Simulado</b><br>x₁ = {x1_val:,.0f} M<br>x₂ = {x2_val:,.0f} M<br><b>ẑ = {z_sim:,.1f} M</b><extra></extra>",
    ))

    fig.update_layout(**get_geo(T))
    fig.update_layout(
        title=dict(
            text=f"<b>z = f(x₁, x₂)</b>  —  x₁ fijo = <span style='color:{T['c_red']}'>{x1_val:,.0f} M</span>",
            font=dict(color=T["text"], size=15, family="Outfit"), x=0.01,
        ),
        xaxis_title=dict(text="x₂ — Ingresos Data Center/IA [Mill. USD]", font=dict(color=T["text"], size=13, family="Outfit", weight="bold")),
        xaxis_range=[15000, 70000], xaxis_tickformat=",.0f",
        yaxis_title=dict(text="z — Ingresos Totales [Mill. USD]", font=dict(color=T["text"], size=13, family="Outfit", weight="bold")),
        yaxis_range=[20000, 75000], yaxis_tickformat=",.0f",
        annotations=[dict(
            x=x2_val, y=z_sim, text=f"  ẑ = {z_sim:,.0f} M",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color=T["c_red"], size=12, family="JetBrains Mono", weight="bold"),
            bgcolor=T["card_bg"], bordercolor=T["c_red"], borderwidth=1.5, borderpad=5,
        )],
    )
    return fig

def graf_lineas_historico(df, x1, x2, T):
    z_sim = f(x1, x2)
    z_r = list(df["z"].values) + [z_sim]
    t = list(df["Trimestre"].values) + ["🎯 SIMULADO"]
    z_m = list(f(df["x1"].values, df["x2"].values)) + [z_sim]
    x_vals = np.arange(1, len(t) + 1)
    
    fig = go.Figure()
    text_labels = [f"({i}, {y:,.0f})" for i, y in zip(x_vals[:-1], z_r[:-1])] + [f"🎯 {z_sim:,.0f}"]
    
    fig.add_trace(go.Scatter(
        x=x_vals[:-1], y=z_r[:-1], mode="lines+markers+text", name="Histórico Real",
        line=dict(color=T["c_red"], width=4.5),
        marker=dict(symbol="x", size=10, color=T["text"], line=dict(width=2.5, color=T["text"])),
        text=text_labels[:-1], textposition="top center", textfont=dict(color=T["text"], size=12, family="Outfit", weight="bold"),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>z Real: $%{y:,.0f}M<extra></extra>",
        customdata=t[:-1]
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals[-2:], y=z_r[-2:], mode="lines+markers+text", name="Proyección Simulación",
        line=dict(color=T["c_blue"], width=4, dash="dot"),
        marker=dict(symbol="diamond", size=14, color=T["c_blue"]),
        text=["", text_labels[-1]], textposition="top center", textfont=dict(color=T["c_blue"], size=14, family="Outfit", weight="bold"),
        hovertemplate="<b>%{customdata}</b><br>ẑ Simulado: $%{y:,.0f}M<extra></extra>",
        customdata=t[-2:]
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals[:-1], y=z_m[:-1], mode="lines+markers", name="Modelo ẑ",
        line=dict(color=T["c_green"], width=3, dash="dash"),
        marker=dict(symbol="circle", size=8, color=T["c_green"]),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>Modelo ẑ: $%{y:,.0f}M<extra></extra>",
        customdata=t[:-1]
    ))
    
    fig.update_layout(**get_geo(T))
    fig.update_layout(
        title=dict(text="<b>Tendencia Histórica NVIDIA + Escenario Interactivo</b>", font=dict(color=T["text"], size=15, family="Outfit"), x=.02),
        xaxis_title=dict(text="Intervalos / Simulación ⟶", font=dict(color=T["text"], size=13, family="Outfit", weight="bold")),
        yaxis_tickformat=",.0f",
        xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=t),
        yaxis_range=[20000, 95000],
        showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor=T["card_bg"], bordercolor=T["card_border"], borderwidth=1),
        clickmode="event+select"
    )
    return fig

def graf_barras_crecimiento(df, x1, x2, T):
    df_chart = df.copy()
    z_sim = f(x1, x2)
    nueva_fila = pd.DataFrame([{"Trimestre": "🎯 SIMULADO", "x1": x1, "x2": x2, "z": z_sim}])
    df_chart = pd.concat([df_chart, nueva_fila], ignore_index=True)
    df_chart['Delta'] = df_chart['z'].diff().fillna(0)
    
    fig = go.Figure()
    colors = [T["c_green"] if i < len(df_chart)-1 else T["c_blue"] for i in range(len(df_chart))]
    
    fig.add_trace(go.Bar(
        x=df_chart['Trimestre'], y=df_chart['z'], name="Ingresos Totales",
        marker=dict(color=colors, opacity=0.85, line=dict(color=colors, width=1.5)),
        text=[f"${v:,.0f}M" for v in df_chart['z']], textposition='outside', textfont=dict(color=T["text"], family="Outfit", weight="bold")
    ))
    
    fig.add_trace(go.Scatter(
        x=df_chart['Trimestre'], y=df_chart['Delta'], mode="lines+markers", name="Crecimiento Marginal",
        line=dict(color=T["c_red"], width=3, dash="dot"),
        marker=dict(size=12, color=T["c_red"], symbol="diamond"),
        yaxis="y2"
    ))
    
    fig.update_layout(**get_geo(T))
    fig.update_layout(
        title=dict(text="<b>Análisis de Ganancias Históricas + Escenario Interactivo</b>", font=dict(color=T["text"], size=15, family="Outfit"), x=0.01),
        yaxis=dict(title=dict(text="Ingresos Totales (z)", font=dict(weight="bold")), range=[0, 95000]),
        yaxis2=dict(title=dict(text="Crecimiento Adicional", font=dict(weight="bold")), overlaying="y", side="right", range=[-15000, 30000], showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor=T["card_bg"], bordercolor=T["card_border"], borderwidth=1),
        margin=dict(t=80, b=40, l=60, r=60)
    )
    return fig

# ==========================================
# 4. THREE.JS 3D RENDERER
# ==========================================
def render_threejs(x1, x2, z, b0, b1, b2, T):
    c_bg = T["three_bg"]
    c_grid = T["three_grid"]
    c_green = T["c_green"].replace('#', '0x')
    c_blue = T["c_blue"].replace('#', '0x')
    c_red = T["c_red"].replace('#', '0x')
    c_text = T["text"].replace('#', '0x')
    
    html = f"""
    <!DOCTYPE html><html><head>
    <style>body {{ margin: 0; background: transparent; overflow: hidden; border-radius: 12px; border: 1px solid {T["card_border"]}; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}</style>
    <script type="importmap">
      {{ "imports": {{
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }} }}
    </script>
    </head><body>
    <script type="module">
      import * as THREE from 'three';
      import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
      
      const scene = new THREE.Scene();
      scene.background = new THREE.Color({c_bg});
      scene.fog = new THREE.FogExp2({c_bg}, 0.015);
      
      const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(30, 20, 30);
      
      const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(renderer.domElement);
      
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.0;
      
      const gridHelper = new THREE.GridHelper(40, 20, {c_grid}, {c_grid});
      scene.add(gridHelper);
      
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight({c_green}, 1.2);
      dirLight.position.set(10, 20, 10);
      scene.add(dirLight);
      const pointLight = new THREE.PointLight({c_blue}, 1.5);
      pointLight.position.set(-10, 10, -10);
      scene.add(pointLight);

      const mapX1 = (val) => ((val - 1200) / (5000 - 1200)) * 30 - 15;
      const mapX2 = (val) => ((val - 20000) / (65000 - 20000)) * 30 - 15;
      const mapZ = (val) => ((val - 20000) / (80000 - 20000)) * 30 - 10;
      
      const geometry = new THREE.PlaneGeometry(30, 30, 15, 15);
      geometry.rotateX(-Math.PI / 2);
      const pos = geometry.attributes.position;
      
      for(let i=0; i<pos.count; i++) {{
          const gx = pos.getX(i);
          const gz = pos.getZ(i);
          const orig_x1 = ((gx + 15) / 30) * (5000 - 1200) + 1200;
          const orig_x2 = ((gz + 15) / 30) * (65000 - 20000) + 20000;
          const true_z = {b0} + {b1}*orig_x1 + {b2}*orig_x2;
          pos.setY(i, mapZ(true_z));
      }}
      geometry.computeVertexNormals();
      
      const material = new THREE.MeshPhongMaterial({{
          color: {c_green}, wireframe: false, transparent: true, opacity: 0.8,
          side: THREE.DoubleSide, shininess: 40
      }});
      const plane = new THREE.Mesh(geometry, material);
      scene.add(plane);
      
      const wireMat = new THREE.MeshBasicMaterial({{color: {c_text}, wireframe: true, transparent:true, opacity:0.15}});
      const wirePlane = new THREE.Mesh(geometry, wireMat);
      scene.add(wirePlane);
      
      const sX = mapX1({x1});
      const sY = mapZ({z});
      const sZ = mapX2({x2});
      
      const sphereGeo = new THREE.SphereGeometry(0.8, 32, 32);
      const sphereMat = new THREE.MeshBasicMaterial({{color: {c_red}}});
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.set(sX, sY, sZ);
      scene.add(sphere);
      
      const sphereLight = new THREE.PointLight({c_red}, 2, 20);
      sphereLight.position.set(sX, sY, sZ);
      scene.add(sphereLight);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(sX, sY, sZ),
          new THREE.Vector3(sX, 0, sZ)
      ]);
      const lineMat = new THREE.LineDashedMaterial({{color: {c_text}, dashSize: 0.5, gapSize: 0.5}});
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      scene.add(line);

      function animate() {{
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
      }}
      animate();
      
      window.addEventListener('resize', () => {{
          camera.aspect = window.innerWidth / window.innerHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(window.innerWidth, window.innerHeight);
      }});
    </script>
    </body></html>
    """
    components.html(html, height=500)

# ==========================================
# 5. UI COMPONENTS
# ==========================================
def render_header(T):
    grad_start = T["card_bg"]
    grad_end = T["bg2"]
    st.markdown(f"""<div style="background:linear-gradient(120deg,{grad_start} 0%,{grad_end} 100%);border-bottom:3px solid {T['c_green']};padding:18px 28px;margin:-1rem -1rem 1.4rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 8px 25px rgba(0,0,0,.08); border-radius: 0 0 16px 16px;">
      <div>
        <div style="font-family:'Oxanium',sans-serif;font-size:1.7rem;font-weight:900;color:{T['text']};letter-spacing:2px;">NVIDIA <span style="color:{T['c_red']}">AI GROWTH</span> MODELER</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{T['text_dim']};letter-spacing:1px;margin-top:4px;font-weight:bold;">UNJFSC · Estadística e Informática &nbsp;|&nbsp; Funciones de Varias Variables  z = f(x₁, x₂)</div>
      </div>
      <div style="display:flex;gap:8px">
        <div style="border:1px solid {T['c_green']};background:{T['c_green']}15;color:{T['c_green']};font-weight:bold;font-size:.55rem;padding:6px 12px;border-radius:12px;font-family:'JetBrains Mono'">MODELO OLS STATSMODELS</div>
        <div style="border:1px solid {T['c_blue']};background:{T['c_blue']}15;color:{T['c_blue']};font-weight:bold;font-size:.55rem;padding:6px 12px;border-radius:12px;font-family:'JetBrains Mono'">THREE.JS 3D</div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar(default_x1, default_x2, r2, T):
    with st.sidebar:
        st.markdown(f"""<div style="text-align:center;padding:10px 0 20px">
          <div style="font-family:'Oxanium',sans-serif;font-size:1.8rem;font-weight:900;color:{T['c_red']};letter-spacing:3px;">NVIDIA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{T['text_dim']};letter-spacing:2px;margin-top:4px;font-weight:bold;">AI GROWTH MODELER</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;text-transform:uppercase;padding-bottom:12px;border-bottom:1px solid {T['card_border']};margin-bottom:16px">● Parámetros de Simulación</div>
        """, unsafe_allow_html=True)
        st.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:{T['text_dim']};font-weight:bold;'>x₁ — <b style=\"color:{T['text']}\">Inversión en I+D</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x1 = st.slider("x1_sl", 1200, 5000, int(default_x1), 50, label_visibility="collapsed")
        st.markdown(f"<span style='font-family:JetBrains Mono,monospace;font-size:.65rem;color:{T['text_dim']};font-weight:bold;'>x₂ — <b style=\"color:{T['text']}\">Ingresos Data Center/IA</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x2 = st.slider("x2_sl", 20000, 65000, int(default_x2), 500, label_visibility="collapsed")
        
        st.markdown(f"<hr style='margin:18px 0;border-color:{T['card_border']}'>", unsafe_allow_html=True)
        st.markdown(f"<b style='color:{T['c_blue']};font-family:Outfit;font-size:0.95rem;'>▶ Animación Interactiva</b>", unsafe_allow_html=True)
        animar = st.button("Simular Escenario Automático ⏯️", use_container_width=True)
        
        r2c = T["c_green"] if r2 >= 0.97 else "#f57c00"
        st.markdown(f"""<div style="background:{T['card_bg']};border:1px solid {T['card_border']};border-radius:12px;padding:16px;text-align:center;margin-top:15px;box-shadow:0 4px 15px rgba(0,0,0,0.05);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{T['text_dim']};font-weight:bold;letter-spacing:1.5px;margin-bottom:6px">BONDAD DE AJUSTE · R² (OLS)</div>
          <div style="font-family:'Oxanium',sans-serif;font-size:2.2rem;font-weight:900;color:{r2c};">{r2:.6f}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<hr style='margin:16px 0;border-color:{T['card_border']}'>", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{T['text_dim']};line-height:2.2;letter-spacing:.5px;font-weight:bold;">
          📚 Cálculo Multivariado<br>🏛 UNJFSC · Est. e Informática<br>👤 Cristian Lucas<br>📅 NVIDIA FY25–FY26
        </div>""", unsafe_allow_html=True)
    return float(x1), float(x2), animar

def render_metricas(x1, x2, r2):
    z = f(x1, x2)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("📐  x₁ — Inversión I+D",  f"${x1:,.0f} M")
    with c2: st.metric("🖥️  x₂ — Data Center",     f"${x2:,.0f} M")
    with c3: st.metric("🟢  ẑ = f(x₁, x₂)",       f"${z:,.1f} M")
    with c4: st.metric("📊  R² del Modelo",          f"{r2:.6f}")

def render_modelo_matematico(x1, x2, T):
    z = f(x1, x2); t1 = BETA1*x1; t2 = BETA2*x2
    conf_int = ols_model.conf_int(alpha=0.05)
    
    st.markdown(f"""<div style="background:linear-gradient(90deg,{T['c_green']}10,{T['card_bg']});border:1px solid {T['card_border']};border-left:4px solid {T['c_green']};border-radius:0 12px 12px 0;padding:16px 22px;margin-bottom:18px;box-shadow:0 4px 15px rgba(0,0,0,0.03)">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px">▶ Marco Matemático — OLS con Statsmodels</div>
      <div style="font-family:'Outfit',sans-serif;font-size:.9rem;color:{T['text_dim']};line-height:1.75">El modelo representa una <strong style="color:{T['text']}">función de dos variables reales</strong> ajustada dinámicamente con <code>statsmodels.api.OLS</code>.</div>
    </div>""", unsafe_allow_html=True)

    col_math, col_info = st.columns([3, 2], gap="medium")

    with col_math:
        mathjax_block(
            "C — Modelo con Coeficientes OLS", T["c_green"],
            [{"label":"Ecuación calibrada","color":T["c_green"],
              "tex": rf"z = \underbrace{{{BETA0:.2f}}}_{{{{\beta_0}}}} + \underbrace{{{BETA1:.4f}}}_{{{{\beta_1}}}}\cdot x_1 + \underbrace{{{BETA2:.4f}}}_{{{{\beta_2}}}}\cdot x_2"}],
            T, altura=200)

        mathjax_block(
            f"D — Evaluación Numérica en x₁ = {x1:,.0f}  y  x₂ = {x2:,.0f}", T["c_red"],
            [{"label":"Desarrollo aritmético","color":T["c_red"],
              "tex": rf"= {BETA0:.2f} + {t1:,.2f} + {t2:,.2f}"},
             {"label":"Resultado final","color":T["c_red"],
              "tex": rf"\boxed{{\;\hat{{z}} = {z:,.2f}\;\text{{millones de USD}}\;}}"}],
            T, altura=220)

    with col_info:
        st.markdown(f"""<div style="background:{T['card_bg']};border:1px solid {T['card_border']};border-radius:12px;padding:16px 18px;margin-bottom:14px;box-shadow:0 4px 15px rgba(0,0,0,0.03)">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Statsmodels Summary</div>""", unsafe_allow_html=True)
        
        coef_table = pd.DataFrame({
            "Coeficiente": ["β₀ (Intercept)", "β₁ (I+D)", "β₂ (Data Center)"],
            "Valor OLS": [f"{BETA0:.4f}", f"{BETA1:.4f}", f"{BETA2:.4f}"],
            "IC Inferior": [f"{conf_int.iloc[0][0]:.4f}", f"{conf_int.iloc[1][0]:.4f}", f"{conf_int.iloc[2][0]:.4f}"],
            "IC Superior": [f"{conf_int.iloc[0][1]:.4f}", f"{conf_int.iloc[1][1]:.4f}", f"{conf_int.iloc[2][1]:.4f}"]
        })
        st.dataframe(coef_table, hide_index=True)
        st.markdown(f"<div style='font-size:0.85rem; color:{T['text_dim']}; font-weight:bold; margin-top:10px'>R-squared: <span style='color:{T['c_blue']}'>{R2_OLS:.6f}</span> &nbsp;|&nbsp; F-statistic: <span style='color:{T['c_blue']}'>{ols_model.fvalue:.1f}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_tabla(df, T):
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
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{T['text_dim']};font-weight:bold;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">● Datos Históricos — Real vs Modelo OLS</div>""", unsafe_allow_html=True)
    st.dataframe(tabla, width='stretch', hide_index=True)

# ==========================================
# 6. MAIN ENTRY
# ==========================================
def main():
    # Theme Toggle Sidebar
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    tema_selector = st.sidebar.radio("🎨 TEMA VISUAL DE LA APP", ["☀️ Claro (Proyector/Presentación)", "🌌 Oscuro (Estilo NVIDIA)"])
    is_dark = "Oscuro" in tema_selector
    T = get_theme(is_dark)

    if "is_animating" not in st.session_state:
        st.session_state.is_animating = False

    # Interacción Bidireccional Plotly On_Select -> Sliders via URL params
    if "chart_hist_select" in st.session_state and st.session_state.chart_hist_select.get("selection", {}).get("points"):
        points = st.session_state.chart_hist_select["selection"]["points"]
        if points:
            x_label = points[0]["x"]
            if isinstance(x_label, (int, float)):
                idx = int(x_label) - 1
                if 0 <= idx < len(df_global):
                    st.query_params["x1"] = str(df_global.iloc[idx]["x1"])
                    st.query_params["x2"] = str(df_global.iloc[idx]["x2"])

    # Leer URL Params
    q_x1 = float(st.query_params.get("x1", 3000.0))
    q_x2 = float(st.query_params.get("x2", 40000.0))

    inject_custom_css(T)
    render_header(T)
    
    x1, x2, animar = render_sidebar(q_x1, q_x2, R2_OLS, T)
    
    if animar:
        st.session_state.is_animating = True
        
    # Escribir URL Params
    st.query_params["x1"] = str(int(x1))
    st.query_params["x2"] = str(int(x2))

    # Typewriter Explanation Component
    x1_str = f"{x1:,.0f}"
    x2_str = f"{x2:,.0f}"
    z_str = f"{f(x1, x2):,.0f}"
    
    html_code = f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    body {{
        background: {T['card_bg']}; 
        border-left: 5px solid {T['c_blue']}; border-right: 1px solid {T['card_border']}; border-top: 1px solid {T['card_border']}; border-bottom: 1px solid {T['card_border']}; 
        border-radius: 12px; padding: 22px; margin: 0;
        color: {T['text']}; font-family: 'Outfit', sans-serif; font-size: 16px; line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); box-sizing: border-box; overflow: hidden;
    }}
    strong {{ color: {T['text']}; font-weight: 700; }} 
    .highlight {{ color: {T['c_green']}; font-weight: 700; background: {T['c_green']}15; padding: 2px 6px; border-radius: 4px; }} 
    .red {{ color: {T['c_red']}; font-weight: 700; font-size: 1.1em; }}
    .title {{ color: {T['c_blue']}; font-size: 19px; margin-top: 0; font-weight: 700; margin-bottom: 15px; letter-spacing: 1px; text-transform: uppercase; }}
    #cursor {{ display: inline-block; width: 8px; height: 16px; background: {T['c_blue']}; animation: blink 1s infinite; vertical-align: middle; margin-left: 4px; }}
    @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
    </style>
    </head><body>
    <div class="title">✨ Análisis Inteligente</div>
    <span id="content"></span><span id="cursor"></span>
    <script>
    const textHTML = `<strong>Modelo Estadístico OLS:</strong><br>
    La función matemática <strong style="color:{T['c_red']};">z = f(x₁, x₂)</strong> ha sido entrenada en tiempo real usando <code>statsmodels</code>. 
    Tu simulación actual establece la Inversión en I+D en <span class="highlight">${x1_str}M</span> y Data Center en <span class="highlight">${x2_str}M</span>.<br><br>
    Bajo este escenario, el modelo proyecta que la variable dependiente <strong>z (Ingresos Totales)</strong> alcance exactamente <span class="red">${z_str} Millones de USD</span>.<br><br>
    <em style="color:{T['text_dim']}; font-weight: 600;">💡 Interactividad Dinámica:</em><br>
    Rota la superficie 3D (pestaña 5), o <strong>haz clic en cualquier punto del gráfico histórico (pestaña 3)</strong> para viajar en el tiempo a ese trimestre. También puedes copiar la URL de esta página para compartir este análisis.`;

    let i = 0; let isTag = false; let currentHTML = ""; const el = document.getElementById("content");
    function type() {{
      if (i < textHTML.length) {{
        let char = textHTML.charAt(i); currentHTML += char;
        if (char === '<') isTag = true;
        if (char === '>') isTag = false;
        el.innerHTML = currentHTML; i++;
        if (isTag) {{ type(); }} else {{ setTimeout(type, 8); }}
      }} else {{ document.getElementById("cursor").style.display = "none"; }}
    }}
    setTimeout(type, 300);
    </script>
    </body></html>
    """
    components.html(html_code, height=330)

    render_metricas(x1, x2, R2_OLS)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈  Gráfica del Modelo (Animado)",
        "📊  Gráfico de Barras (Ganancias)",
        "📉  Tendencia Histórica (Clickeable)",
        "🔢  Modelo Matemático (OLS)",
        "🪐  Superficie 3D (Three.js)"
    ])
    
    with tab1:
        st.markdown(f"<div style='font-family:JetBrains Mono;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;padding:8px 0 14px;'>● Plano cartesiano matemático interactivo</div>", unsafe_allow_html=True)
        graf_placeholder = st.empty()
        
        if st.session_state.is_animating:
            import time
            for anim_val in range(1200, 5200, 150):
                with graf_placeholder.container():
                    st.plotly_chart(graf_funcion_x2(df_global, anim_val, x2, T), width="stretch", config={"displaylogo": False})
                time.sleep(0.08)
            st.session_state.is_animating = False
            st.rerun()
        else:
            with graf_placeholder.container():
                st.plotly_chart(graf_funcion_x2(df_global, x1, x2, T), width='stretch', config={'displaylogo': False})
            
        st.markdown(f"<span style='font-family:JetBrains Mono;font-size:.65rem;color:{T['text_dim']};font-weight:bold;'>↕ Mueve el slider <b style='color:{T['c_red']}'>x₁</b> para desplazar la curva arriba/abajo. ↔ Mueve el slider <b style='color:{T['c_red']}'>x₂</b> para deslizar el punto.</span>", unsafe_allow_html=True)

    with tab2:
        st.markdown(f"<div style='font-family:JetBrains Mono;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;padding:8px 0 14px;'>● Balance General: Ingresos vs Crecimiento</div>", unsafe_allow_html=True)
        st.plotly_chart(graf_barras_crecimiento(df_global, x1, x2, T), width="stretch", config={"displaylogo": False})
        
    with tab3:
        st.markdown(f"<div style='font-family:JetBrains Mono;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;padding:8px 0 14px;'>● Clic en cualquier punto de la línea para aplicar los valores de ese trimestre</div>", unsafe_allow_html=True)
        st.plotly_chart(graf_lineas_historico(df_global, x1, x2, T), width="stretch", config={"displaylogo": False}, on_select="rerun", selection_mode="points", key="chart_hist_select")
        render_tabla(df_global, T)

    with tab4:
        render_modelo_matematico(x1, x2, T)
        
    with tab5:
        st.markdown(f"<div style='font-family:JetBrains Mono;font-size:.65rem;color:{T['c_red']};font-weight:bold;letter-spacing:2px;padding:8px 0 14px;'>● Motor gráfico 3D nativo con Three.js (Usa el ratón para rotar en 360°)</div>", unsafe_allow_html=True)
        render_threejs(x1, x2, f(x1, x2), BETA0, BETA1, BETA2, T)

    st.markdown("---")
    st.markdown(f"""<div style="text-align:center;padding:12px 0;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{T['text_dim']};font-weight:bold;letter-spacing:1px">
      NVIDIA AI Growth Modeler &nbsp;·&nbsp; Herramienta Educativa Animada &nbsp;·&nbsp; Cálculo Multivariado
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
