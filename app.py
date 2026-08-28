import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import statsmodels.api as sm
import sqlite3

st.set_page_config(page_title="NVIDIA AI Growth Modeler", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. DATABASE LOGIC (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('simulaciones.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS simulaciones (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  x1 REAL, x2 REAL, z REAL)""")
    conn.commit()
    conn.close()

def save_simulacion(x1, x2, z):
    conn = sqlite3.connect('simulaciones.db')
    c = conn.cursor()
    c.execute("INSERT INTO simulaciones (x1, x2, z) VALUES (?, ?, ?)", (float(x1), float(x2), float(z)))
    conn.commit()
    conn.close()

def get_historial():
    conn = sqlite3.connect('simulaciones.db')
    df = pd.read_sql_query("SELECT id, datetime(timestamp, 'localtime') as Fecha, x1 as 'Inversión I+D', x2 as 'Data Center', z as 'Ingresos (z)' FROM simulaciones ORDER BY timestamp DESC LIMIT 15", conn)
    conn.close()
    return df

# ==========================================
# 2. GLOBAL DATA & STATSMODELS OLS
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
# 3. STYLES & UI CONFIG
# ==========================================
GEO = dict(
    plot_bgcolor="#0a0f16", paper_bgcolor="#0a0f16",
    font=dict(color="#a0b8c8", family="Outfit"),
    margin=dict(l=50, r=40, t=60, b=50),
    xaxis=dict(showgrid=True, gridcolor="#1a2b3c", zeroline=True, zerolinecolor="#4a6070", zerolinewidth=1.5,
               tickfont=dict(color="#a0b8c8", size=11, family="JetBrains Mono")),
    yaxis=dict(showgrid=True, gridcolor="#1a2b3c", zeroline=True, zerolinecolor="#4a6070", zerolinewidth=1.5,
               tickfont=dict(color="#a0b8c8", size=11, family="JetBrains Mono")),
)

def inject_custom_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600&family=Oxanium:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #060b12; color: #d8e8f0; }
    [data-testid="stSidebar"] { background-color: #081018; border-right: 1px solid #1e3048; }
    .stSlider > div > div > div > div { background-color: #f40000; }
    .stSlider > div > div > div > div > div { background-color: #ff4d6d; border: 2px solid #fff; box-shadow: 0 0 10px rgba(244,0,0,0.8); }
    h1, h2, h3 { font-family: 'Oxanium', sans-serif; font-weight: 700; color: #fff; }
    div[data-testid="stMetricValue"] { font-family: 'Oxanium', sans-serif; font-weight: 700; color: #fff; font-size: 1.8rem; }
    div[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0b8c8; letter-spacing: 1px; }
    .stButton>button { background: #1a2b3c; border: 1px solid #3a5060; color: #d8e8f0; font-family: 'JetBrains Mono'; border-radius: 6px; transition: all 0.3s ease; }
    .stButton>button:hover { background: #00d4ff; border-color: #00d4ff; color: #000; box-shadow: 0 0 15px rgba(0,212,255,0.4); }
    .stTabs [data-baseweb="tab-list"] { background-color: #060b12; border-bottom: 2px solid #1e3048; gap: 8px; }
    .stTabs [data-baseweb="tab"] { color: #6a8099; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 12px 16px; border: 1px solid transparent; border-radius: 8px 8px 0 0; transition: all 0.3s ease; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #76b900; border: 1px solid #76b900; border-bottom: none; background: rgba(118,185,0,0.05); text-shadow: 0 0 10px rgba(118,185,0,0.3); }
    </style>""", unsafe_allow_html=True)

def mathjax_block(titulo, color, lineas, altura=150):
    html = f"""<!DOCTYPE html><html><head>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
      body {{ margin:0; padding:16px; background: transparent; color: #a0b8c8; font-family: 'Outfit', sans-serif; font-size: 14px; overflow: hidden; }}
      .box {{ border-left: 3px solid {color}; padding-left: 14px; margin-bottom: 12px; }}
      .title {{ font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {color}; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; font-weight: 700; }}
      .mj-container {{ background: rgba(255,255,255,0.02); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); margin-top: 6px; text-align: center; overflow-x: auto; overflow-y: hidden; }}
      .prose {{ color: #7a8da0; font-size: 13px; line-height: 1.5; }}
    </style></head><body>
    <div class="box"><div class="title">{titulo}</div>"""
    for l in lineas:
        if "prose" in l: html += f'<div class="prose">{l["prose"]}</div>'
        html += f'<div class="mj-container">\\({l["tex"]}\\)</div>'
    html += "</div></body></html>"
    components.html(html, height=altura)

# ==========================================
# 4. PLOTS
# ==========================================
def graf_funcion_x2(df, x1_val, x2_val):
    x_range = np.linspace(20000, 65000, 50)
    z_curve = f(x1_val, x_range)
    z_sim = f(x1_val, x2_val)
    
    fig = go.Figure()
    
    # Superficie cortada (Plano 2D z vs x2)
    fig.add_trace(go.Scatter(
        x=x_range, y=z_curve, mode="lines",
        name=f"z = f({x1_val:,.0f}, x₂)",
        line=dict(color="#f40000", width=4),
        fill="tozeroy", fillcolor="rgba(244, 0, 0, 0.08)"
    ))
    
    # Datos proyectados
    z_proj = f(x1_val, df["x2"].values)
    fig.add_trace(go.Scatter(
        x=df["x2"], y=z_proj, mode="markers",
        name="Trimestres Proyectados",
        marker=dict(symbol="circle-open", size=8, color="#00d4ff", line_width=2),
        hovertemplate="x₂: %{x:,.0f}<br>z Proy: %{y:,.0f}<extra></extra>"
    ))
    
    # Punto Simulado
    fig.add_trace(go.Scatter(
        x=[x2_val], y=[z_sim], mode="markers",
        name="Punto Actual",
        marker=dict(size=18, color="#ff4d6d", symbol="diamond", line=dict(color="#fff", width=2.5)),
        hovertemplate=f"<b>Punto Simulado</b><br>x₁ = {x1_val:,.0f} M<br>x₂ = {x2_val:,.0f} M<br><b>ẑ = {z_sim:,.1f} M</b><extra></extra>",
    ))

    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(
            text=f"<b>z = f(x₁, x₂)</b>  —  x₁ fijo = <span style='color:#f40000'>{x1_val:,.0f} M</span>  |  mueve el slider → la curva sube o baja",
            font=dict(color="#000000", size=14, family="Outfit"), x=0.01,
        ),
        xaxis_title=dict(text="x₂ — Ingresos Data Center/IA [Mill. USD]", font=dict(color="#000000", size=12, family="Outfit")),
        xaxis_range=[15000, 70000], xaxis_tickformat=",.0f",
        yaxis_title=dict(text="z — Ingresos Totales [Mill. USD]", font=dict(color="#000000", size=12, family="Outfit")),
        yaxis_range=[20000, 75000], yaxis_tickformat=",.0f",
        annotations=[dict(
            x=x2_val, y=z_sim, text=f"  ẑ = {z_sim:,.0f} M",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color="#ff4d6d", size=11, family="JetBrains Mono"),
            bgcolor="rgba(10,15,24,.85)", bordercolor="#ff4d6d", borderwidth=1, borderpad=4,
        )],
    )
    return fig

def graf_lineas_historico(df, x1, x2):
    z_sim = f(x1, x2)
    z_r = list(df["z"].values) + [z_sim]
    t = list(df["Trimestre"].values) + ["🎯 SIMULADO"]
    z_m = list(f(df["x1"].values, df["x2"].values)) + [z_sim]
    x_vals = np.arange(1, len(t) + 1)
    
    fig = go.Figure()
    text_labels = [f"({i}, {y:,.0f})" for i, y in zip(x_vals[:-1], z_r[:-1])] + [f"🎯 {z_sim:,.0f}"]
    
    fig.add_trace(go.Scatter(
        x=x_vals[:-1], y=z_r[:-1], mode="lines+markers+text", name="Histórico",
        line=dict(color="#f40000", width=4),
        marker=dict(symbol="x", size=10, color="#000000", line=dict(width=2.5, color="#000000")),
        text=text_labels[:-1], textposition="top center", textfont=dict(color="#000000", size=11, family="Outfit"),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>z Real: $%{y:,.0f}M<extra></extra>",
        customdata=t[:-1]
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals[-2:], y=z_r[-2:], mode="lines+markers+text", name="Proyección Simulación",
        line=dict(color="#00d4ff", width=4, dash="dot"),
        marker=dict(symbol="diamond", size=14, color="#00d4ff"),
        text=["", text_labels[-1]], textposition="top center", textfont=dict(color="#00d4ff", size=13, family="Outfit", weight="bold"),
        hovertemplate="<b>%{customdata}</b><br>ẑ Simulado: $%{y:,.0f}M<extra></extra>",
        customdata=t[-2:]
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals[:-1], y=z_m[:-1], mode="lines+markers", name="Modelo ẑ",
        line=dict(color="#0055ff", width=2.5, dash="dash"),
        marker=dict(symbol="circle", size=7, color="#0055ff"),
        hovertemplate="<b>Trimestre: %{customdata}</b><br>Modelo ẑ: $%{y:,.0f}M<extra></extra>",
        customdata=t[:-1]
    ))
    
    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(text="<b>Tendencia Histórica NVIDIA + Escenario Interactivo</b>", font=dict(color="#000000", size=14, family="Outfit"), x=.02),
        xaxis_title=dict(text="Intervalos / Simulación ⟶", font=dict(color="#000000", size=13, family="Outfit")),
        yaxis_tickformat=",.0f",
        xaxis=dict(tickmode='array', tickvals=x_vals, ticktext=t),
        yaxis_range=[20000, 95000],
        showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        clickmode="event+select" # Enable click events
    )
    return fig

def graf_barras_crecimiento(df, x1, x2):
    df_chart = df.copy()
    z_sim = f(x1, x2)
    nueva_fila = pd.DataFrame([{"Trimestre": "🎯 SIMULADO", "x1": x1, "x2": x2, "z": z_sim}])
    df_chart = pd.concat([df_chart, nueva_fila], ignore_index=True)
    df_chart['Delta'] = df_chart['z'].diff().fillna(0)
    
    fig = go.Figure()
    colors = ["#76b900" if i < len(df_chart)-1 else "#00d4ff" for i in range(len(df_chart))]
    lines = ["#76b900" if i < len(df_chart)-1 else "#00d4ff" for i in range(len(df_chart))]
    
    fig.add_trace(go.Bar(
        x=df_chart['Trimestre'], y=df_chart['z'], name="Ingresos Totales",
        marker=dict(color=colors, line=dict(color=lines, width=1.5)),
        text=[f"${v:,.0f}M" for v in df_chart['z']], textposition='outside', textfont=dict(color="#d8e8f0", family="Outfit")
    ))
    
    fig.add_trace(go.Scatter(
        x=df_chart['Trimestre'], y=df_chart['Delta'], mode="lines+markers", name="Crecimiento Marginal",
        line=dict(color="#00d4ff", width=3, dash="dot"),
        marker=dict(size=12, color="#00d4ff", symbol="diamond"),
        yaxis="y2"
    ))
    
    fig.update_layout(**GEO)
    fig.update_layout(
        title=dict(text="<b>Análisis de Ganancias Históricas + Escenario Interactivo</b>", font=dict(color="#000000", size=14, family="Outfit"), x=0.01),
        yaxis=dict(title="Ingresos Totales (z)", range=[0, 95000]),
        yaxis2=dict(title="Crecimiento Adicional", overlaying="y", side="right", range=[-15000, 30000], showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=80, b=40, l=60, r=60)
    )
    return fig

# ==========================================
# 5. THREE.JS 3D RENDERER
# ==========================================
def render_threejs(x1, x2, z, b0, b1, b2):
    html = f"""
    <!DOCTYPE html><html><head>
    <style>body {{ margin: 0; background: #060b12; overflow: hidden; border-radius: 12px; }}</style>
    <!-- Import map for modern three.js -->
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
      scene.fog = new THREE.FogExp2(0x060b12, 0.015);
      const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(30, 20, 30);
      
      const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(renderer.domElement);
      
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.0;
      
      // Grid & Axes
      const gridHelper = new THREE.GridHelper(40, 20, 0x1e3048, 0x1e3048);
      scene.add(gridHelper);
      
      // Light
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
      scene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight(0x76b900, 1.5);
      dirLight.position.set(10, 20, 10);
      scene.add(dirLight);
      const pointLight = new THREE.PointLight(0x00d4ff, 2);
      pointLight.position.set(-10, 10, -10);
      scene.add(pointLight);

      // Function Surface: Z = b0 + b1*x1 + b2*x2
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
          color: 0x76b900, wireframe: false, transparent: true, opacity: 0.6,
          side: THREE.DoubleSide, shininess: 80
      }});
      const plane = new THREE.Mesh(geometry, material);
      scene.add(plane);
      
      const wireMat = new THREE.MeshBasicMaterial({{color: 0x00d4ff, wireframe: true, transparent:true, opacity:0.3}});
      const wirePlane = new THREE.Mesh(geometry, wireMat);
      scene.add(wirePlane);
      
      // Simulated Point
      const sX = mapX1({x1});
      const sY = mapZ({z});
      const sZ = mapX2({x2});
      
      const sphereGeo = new THREE.SphereGeometry(0.8, 32, 32);
      const sphereMat = new THREE.MeshBasicMaterial({{color: 0xff4d6d}});
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.set(sX, sY, sZ);
      scene.add(sphere);
      
      // Glow point light at sphere
      const sphereLight = new THREE.PointLight(0xff4d6d, 3, 20);
      sphereLight.position.set(sX, sY, sZ);
      scene.add(sphereLight);

      // Line from plane to grid
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(sX, sY, sZ),
          new THREE.Vector3(sX, 0, sZ)
      ]);
      const lineMat = new THREE.LineDashedMaterial({{color: 0xffffff, dashSize: 0.5, gapSize: 0.5}});
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
# 6. UI COMPONENTS
# ==========================================
def render_header():
    st.markdown("""<div style="background:linear-gradient(120deg,#040608 0%,#080f0a 60%,#060c04 100%);border-bottom:2px solid #76b900;padding:18px 28px;margin:-1rem -1rem 1.4rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 4px 32px rgba(118,185,0,.18)">
      <div>
        <div style="font-family:'Oxanium',sans-serif;font-size:1.55rem;font-weight:800;color:#fff;letter-spacing:3px;text-shadow:0 0 24px rgba(118,185,0,.5)">NVIDIA <span style="color:#f40000">AI GROWTH</span> MODELER</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.59rem;color:#3a5060;letter-spacing:1px;margin-top:4px">UNJFSC · Estadística e Informática &nbsp;|&nbsp; Funciones de Varias Variables  z = f(x₁, x₂)</div>
      </div>
      <div style="display:flex;gap:8px">
        <div style="border:1px solid #76b900;color:#76b900;font-size:.55rem;padding:4px 10px;border-radius:12px;font-family:'JetBrains Mono'">MODELO OLS STATSMODELS</div>
        <div style="border:1px solid #00d4ff;color:#00d4ff;font-size:.55rem;padding:4px 10px;border-radius:12px;font-family:'JetBrains Mono'">THREE.JS 3D</div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar(default_x1, default_x2, r2):
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:10px 0 20px">
          <div style="font-family:'Oxanium',sans-serif;font-size:1.4rem;font-weight:800;color:#f40000;letter-spacing:3px;text-shadow:0 0 18px rgba(118,185,0,.6)">NVIDIA</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.56rem;color:#3a5060;letter-spacing:2px;margin-top:4px">AI GROWTH MODELER</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:2px;text-transform:uppercase;padding-bottom:12px;border-bottom:1px solid #1e3048;margin-bottom:16px">● Parámetros de Simulación</div>
        """, unsafe_allow_html=True)
        st.markdown("<span style='font-family:JetBrains Mono,monospace;font-size:.63rem;color:#6a8099'>x₁ — <b style=\"color:#d8e8f0\">Inversión en I+D</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x1 = st.slider("x1_sl", 1200, 5000, int(default_x1), 50, label_visibility="collapsed")
        st.markdown("<span style='font-family:JetBrains Mono,monospace;font-size:.63rem;color:#6a8099'>x₂ — <b style=\"color:#d8e8f0\">Ingresos Data Center/IA</b> [Mill. USD]</span>", unsafe_allow_html=True)
        x2 = st.slider("x2_sl", 20000, 65000, int(default_x2), 500, label_visibility="collapsed")
        
        st.markdown("<hr style='margin:18px 0;border-color:#1e3048'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#00d4ff;font-family:Outfit;font-size:0.9rem;'>▶ Animación Interactiva</b>", unsafe_allow_html=True)
        animar = st.button("Simular Escenario Automático ⏯️", use_container_width=True)
        guardar = st.button("💾 Guardar Escenario en BD", use_container_width=True)
        
        r2c = "#76b900" if r2 >= 0.97 else "#f5a623"
        st.markdown(f"""<div style="background:#060b12;border:1px solid #1e3048;border-radius:9px;padding:14px;text-align:center;margin-top:15px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.54rem;color:#3a5060;letter-spacing:1.5px;margin-bottom:6px">BONDAD DE AJUSTE · R² (OLS)</div>
          <div style="font-family:'Oxanium',sans-serif;font-size:2rem;font-weight:800;color:{r2c};text-shadow:0 0 14px {r2c}66">{r2:.6f}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr style='margin:16px 0;border-color:#1e3048'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#2a4050;line-height:2.2;letter-spacing:.5px">
          📚 Cálculo Multivariado<br>🏛 UNJFSC · Est. e Informática<br>👤 Cristian Lucas<br>📅 NVIDIA FY25–FY26
        </div>""", unsafe_allow_html=True)
    return float(x1), float(x2), animar, guardar

def render_metricas(x1, x2, r2):
    z = f(x1, x2)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("📐  x₁ — Inversión I+D",  f"${x1:,.0f} M")
    with c2: st.metric("🖥️  x₂ — Data Center",     f"${x2:,.0f} M")
    with c3: st.metric("🟢  ẑ = f(x₁, x₂)",       f"${z:,.1f} M")
    with c4: st.metric("📊  R² del Modelo",          f"{r2:.6f}")

def render_modelo_matematico(x1, x2):
    z = f(x1, x2); t1 = BETA1*x1; t2 = BETA2*x2
    conf_int = ols_model.conf_int(alpha=0.05)
    
    st.markdown("""<div style="background:linear-gradient(90deg,#0c1a0a,#0d1822);border:1px solid #1e3048;border-left:4px solid #76b900;border-radius:0 12px 12px 0;padding:16px 22px;margin-bottom:18px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.61rem;color:#f40000;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px">▶ Marco Matemático — OLS con Statsmodels</div>
      <div style="font-family:'Outfit',sans-serif;font-size:.84rem;color:#7a8da0;line-height:1.75">El modelo representa una <strong style="color:#d8e8f0">función de dos variables reales</strong> ajustada dinámicamente con <code>statsmodels.api.OLS</code>.</div>
    </div>""", unsafe_allow_html=True)

    col_math, col_info = st.columns([3, 2], gap="medium")

    with col_math:
        mathjax_block(
            "C — Modelo con Coeficientes OLS", "#f5a623",
            [{"label":"Ecuación calibrada","color":"#f5a623",
              "tex": rf"z = \underbrace{{{BETA0:.2f}}}_{{{{\beta_0}}}} + \underbrace{{{BETA1:.4f}}}_{{{{\beta_1}}}}\cdot x_1 + \underbrace{{{BETA2:.4f}}}_{{{{\beta_2}}}}\cdot x_2"}],
            altura=200)

        mathjax_block(
            f"D — Evaluación Numérica en x₁ = {x1:,.0f}  y  x₂ = {x2:,.0f}", "#ff4d6d",
            [{"label":"Desarrollo aritmético","color":"#ff4d6d",
              "tex": rf"= {BETA0:.2f} + {t1:,.2f} + {t2:,.2f}"},
             {"label":"Resultado final","color":"#ff4d6d",
              "tex": rf"\boxed{{\;\hat{{z}} = {z:,.2f}\;\text{{millones de USD}}\;}}"}],
            altura=220)

    with col_info:
        st.markdown("""<div style="background:#0d1821;border:1px solid #1e3048;border-radius:10px;padding:16px 18px;margin-bottom:14px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:#f40000;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Statsmodels Summary</div>""", unsafe_allow_html=True)
        
        coef_table = pd.DataFrame({
            "Coeficiente": ["β₀ (Intercept)", "β₁ (I+D)", "β₂ (Data Center)"],
            "Valor OLS": [f"{BETA0:.4f}", f"{BETA1:.4f}", f"{BETA2:.4f}"],
            "IC Inferior": [f"{conf_int.iloc[0][0]:.4f}", f"{conf_int.iloc[1][0]:.4f}", f"{conf_int.iloc[2][0]:.4f}"],
            "IC Superior": [f"{conf_int.iloc[0][1]:.4f}", f"{conf_int.iloc[1][1]:.4f}", f"{conf_int.iloc[2][1]:.4f}"]
        })
        st.dataframe(coef_table, hide_index=True)
        st.markdown(f"<div style='font-size:0.8rem; color:#7a8da0'>R-squared: {R2_OLS:.6f} | F-statistic: {ols_model.fvalue:.1f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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

# ==========================================
# 7. MAIN ENTRY
# ==========================================
def main():
    init_db()
    
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

    inject_custom_css()
    render_header()
    
    x1, x2, animar, guardar = render_sidebar(q_x1, q_x2, R2_OLS)
    
    if animar:
        st.session_state.is_animating = True
        
    if guardar:
        save_simulacion(x1, x2, f(x1, x2))
        st.toast("Escenario insertado en la Base de Datos SQLite (simulaciones.db) ✅", icon="💾")
        
    # Escribir URL Params
    st.query_params["x1"] = str(int(x1))
    st.query_params["x2"] = str(int(x2))

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
        border-left: 4px solid #00d4ff; border-right: 1px solid #1e3048; border-top: 1px solid #1e3048; border-bottom: 1px solid #1e3048; 
        border-radius: 10px; padding: 22px; margin: 0;
        color: #a0b8c8; font-family: 'Outfit', sans-serif; font-size: 15.5px; line-height: 1.6;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.05); box-sizing: border-box; overflow: hidden;
    }}
    strong {{ color: #d8e8f0; }} .highlight {{ color: #76b900; font-weight: bold; }} .red {{ color: #f40000; font-weight: bold; }}
    .title {{ color: #00d4ff; font-size: 18px; margin-top: 0; font-weight: 600; margin-bottom: 15px; letter-spacing: 1px; }}
    #cursor {{ display: inline-block; width: 8px; height: 16px; background: #00d4ff; animation: blink 1s infinite; vertical-align: middle; margin-left: 4px; }}
    @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
    </style>
    </head><body>
    <div class="title">Explicación Inteligente</div>
    <span id="content"></span><span id="cursor"></span>
    <script>
    const textHTML = `<strong>Análisis Estadístico OLS:</strong><br>
    La función matemática que rige el modelo es <strong style="color:#f40000;">z = f(x₁, x₂)</strong> entrenada en tiempo real vía <code>statsmodels</code>. 
    Actualmente has fijado la Inversión en I+D en <strong><span class="highlight">${x1_str}M</span></strong> y Data Center en <strong><span class="highlight">${x2_str}M</span></strong>.<br><br>
    Esto predice que la variable dependiente <strong>z (Ingresos Totales)</strong> alcance <strong><span class="red">${z_str}M</span></strong>.<br><br>
    <em style="color:#d8e8f0;">💡 Interactividad Avanzada Activada:</em><br>
    Puedes rotar la superficie 3D (pestaña 5), o <strong>hacer clic en los puntos del gráfico histórico (pestaña 3)</strong> para viajar en el tiempo a ese trimestre. También puedes copiar la URL actual para compartir este escenario exacto con tus compañeros, gracias al mapeo bidireccional en la URL.`;

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
        "📈  Gráficas GeoGebra (Animado)",
        "📊  Gráfico de Barras (Ganancias)",
        "📉  Tendencia Histórica (Clickeable)",
        "🔢  Modelo Matemático (OLS)",
        "🪐  Superficie 3D (Three.js)"
    ])
    
    with tab1:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:.6rem;color:#f40000;letter-spacing:2px;padding:8px 0 14px;'>● Plano cartesiano matemático interactivo</div>", unsafe_allow_html=True)
        graf_placeholder = st.empty()
        
        if st.session_state.is_animating:
            import time
            for anim_val in range(1200, 5200, 150):
                with graf_placeholder.container():
                    st.plotly_chart(graf_funcion_x2(df_global, anim_val, x2), width="stretch", config={"displaylogo": False})
                time.sleep(0.08)
            st.session_state.is_animating = False
            st.rerun()
        else:
            with graf_placeholder.container():
                st.plotly_chart(graf_funcion_x2(df_global, x1, x2), width='stretch', config={'displaylogo': False})
            
        st.markdown("<span style='font-family:JetBrains Mono;font-size:.62rem;color:#4a6070;'>↕ Mueve el slider <b style='color:#f40000'>x₁</b> para desplazar la curva arriba/abajo. ↔ Mueve el slider <b style='color:#f40000'>x₂</b> para deslizar el punto.</span>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:.6rem;color:#f40000;letter-spacing:2px;padding:8px 0 14px;'>● Balance General: Ingresos vs Crecimiento</div>", unsafe_allow_html=True)
        st.plotly_chart(graf_barras_crecimiento(df_global, x1, x2), width="stretch", config={"displaylogo": False})
        
    with tab3:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:.6rem;color:#f40000;letter-spacing:2px;padding:8px 0 14px;'>● Clic en cualquier punto de la línea para aplicar los valores de ese trimestre</div>", unsafe_allow_html=True)
        st.plotly_chart(graf_lineas_historico(df_global, x1, x2), width="stretch", config={"displaylogo": False}, on_select="rerun", selection_mode="points", key="chart_hist_select")
        render_tabla(df_global)

    with tab4:
        render_modelo_matematico(x1, x2)
        
    with tab5:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:.6rem;color:#f40000;letter-spacing:2px;padding:8px 0 14px;'>● Motor gráfico 3D nativo con Three.js (Usa el ratón para rotar en 360°)</div>", unsafe_allow_html=True)
        render_threejs(x1, x2, f(x1, x2), BETA0, BETA1, BETA2)

    st.markdown("---")
    st.markdown("### 💾 Historial de Simulaciones (SQLite `simulaciones.db`)")
    st.markdown("<p style='font-family:Outfit;color:#7a8da0;font-size:0.85rem'>Nota: Este historial persiste en memoria gracias a la base de datos SQL local. En Streamlit Cloud gratuito, el archivo de la BD se reiniciará cuando la máquina entre en hibernación, a menos que conectes PostgreSQL/Supabase.</p>", unsafe_allow_html=True)
    df_hist = get_historial()
    if not df_hist.empty:
        st.dataframe(df_hist, hide_index=True, use_container_width=True)
    else:
        st.info("Aún no has guardado ningún escenario. Usa el botón en el panel izquierdo.")

    st.markdown("""<div style="text-align:center;padding:28px 0 8px;font-family:'JetBrains Mono',monospace;font-size:.53rem;color:#4a6070;letter-spacing:1px">
      NVIDIA AI Growth Modeler &nbsp;·&nbsp; Herramienta Educativa Animada &nbsp;·&nbsp; Cálculo Multivariado
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
