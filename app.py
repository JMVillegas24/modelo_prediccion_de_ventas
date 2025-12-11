"""
APLICACIÓN STREAMLIT - Sales Intelligence Platform v6.2
Frontend profesional con Sidebar y tema oscuro
VERSIÓN FINAL v6.2 - Categorías Corregidas
"""

import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime, timedelta
import os
import backend as be
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================

st.set_page_config(
    page_title="Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

be.aplicar_tema_oscuro()

# ==========================================
# GESTIÓN DE BD - v6.2
# ==========================================

DB_PATH = 'sales_app.db'

def init_db():
    """Inicializa la BD - VERSIÓN v6.2"""
    try:
        db_exists = os.path.exists(DB_PATH)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS users 
        (username TEXT PRIMARY KEY, password TEXT)''')
        
        if not db_exists:
            c.execute('''CREATE TABLE predictions
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT, 
            client_name TEXT,
            zone TEXT, 
            category_product TEXT, 
            prob REAL, 
            label TEXT, 
            recommendation TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        else:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
            if not c.fetchone():
                c.execute('''CREATE TABLE predictions
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                user_id TEXT, 
                client_name TEXT,
                zone TEXT, 
                category_product TEXT, 
                prob REAL, 
                label TEXT, 
                recommendation TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"❌ Error en init_db: {str(e)}")
        return False

def user_auth(username, password, mode='login'):
    """Autenticación de usuarios - v6.2"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        pwd_hash = hashlib.sha256(str.encode(password)).hexdigest()
        
        if mode == 'login':
            c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, pwd_hash))
            result = c.fetchall()
            conn.close()
            return result
        
        elif mode == 'signup':
            try:
                c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, pwd_hash))
                conn.commit()
                conn.close()
                return True
            except sqlite3.IntegrityError:
                conn.close()
                return False
    
    except Exception as e:
        print(f"❌ Error en user_auth: {str(e)}")
        try:
            conn.close()
        except:
            pass
        return False
    
    return False

def save_prediction_db(user, client, zone, category_product, prob, label, recommendation):
    """Guarda predicción - v6.2"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.isolation_level = None
        c = conn.cursor()
        
        prob_float = float(prob)
        
        c.execute('''INSERT INTO predictions 
        (user_id, client_name, zone, category_product, prob, label, recommendation) 
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (user, client, zone, category_product, prob_float, label, recommendation))
        
        conn.close()
        return True
    
    except Exception as e:
        print(f"❌ Error en save_prediction_db: {str(e)}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return False

def get_history(user_id=None, days=30):
    """Obtiene histórico de predicciones - v6.2"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        
        if user_id:
            query = f"""SELECT * FROM predictions 
            WHERE user_id = ? 
            AND timestamp > datetime('now', '-{days} days') 
            ORDER BY timestamp DESC"""
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:
            query = f"""SELECT * FROM predictions 
            WHERE timestamp > datetime('now', '-{days} days') 
            ORDER BY timestamp DESC"""
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        return df if not df.empty else pd.DataFrame()
    
    except Exception as e:
        print(f"❌ Error en get_history: {str(e)}")
        try:
            conn.close()
        except:
            pass
        return pd.DataFrame()

# ==========================================
# FUNCIONES DE ANÁLISIS
# ==========================================

def mostrar_dashboard_general(df):
    """Dashboard general con KPIs y gráficos principales - v6.2 CORREGIDO"""
    
    st.markdown("### 📊 Dashboard de Inteligencia de Ventas")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_cotizaciones = len(df)
    ventas_ganadas = len(df[df['¿Adjudicado?'] == 1])
    tasa_conversion = df['¿Adjudicado?'].mean()
    
    with col1:
        be.mostrar_kpi_grande("Total de Cotizaciones", f"{total_cotizaciones:,}", color='#4D96FF', icon='📋')
    
    with col2:
        be.mostrar_kpi_grande("Ventas Ganadas", f"{ventas_ganadas:,}", color='#6BCB77', icon='✅')
    
    with col3:
        be.mostrar_kpi_grande("Tasa de Conversión", f"{tasa_conversion:.1%}", color='#FFD93D', icon='📈')
    
    with col4:
        unique_clientes = df['Cliente'].nunique()
        be.mostrar_kpi_grande("Clientes Únicos", f"{unique_clientes:,}", color='#FF6B6B', icon='👥')
    
    st.divider()
    
    # Gráficos principales
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Tasa por categoría - CORREGIDO v6.2
        stats_categoria = df.groupby('Categoria_Producto')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_categoria.columns = ['Categoria', 'Total', 'Ganadas', 'Tasa']
        stats_categoria = stats_categoria.sort_values('Tasa', ascending=True)
        
        fig_cat = be.grafico_barras_horizontal(
            stats_categoria,
            'Tasa', 'Categoria',
            '📊 Tasa de Conversión por Categoría de Producto',
            '#FF6B6B'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_right:
        # Desempeño por zona
        stats_zona = df.groupby('Zona Geográfica')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_zona.columns = ['Zona', 'Total', 'Ganadas', 'Tasa']
        
        colores_zona = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#6BCB77']
        fig_zona = be.grafico_pie_categorias(stats_zona, 'Zona', '🌍 Distribución por Zona', colores_zona)
        st.plotly_chart(fig_zona, use_container_width=True)
    
    st.divider()
    
    # Top vendedores
    col_left, col_middle, col_right = st.columns(3)
    
    with col_left:
        st.markdown("### 🏆 Top 5 Vendedores")
        top_vendedores = df[df['¿Adjudicado?'] == 1]['Usuario_Interno'].value_counts().head(5)
        for i, (vendedor, cant) in enumerate(top_vendedores.items(), 1):
            st.markdown(f'<div style="margin: 8px 0;"><span style="color: #4D96FF; font-weight: bold;">{i}.</span> <span style="color: #FFF;">{vendedor}: <b style="color: #6BCB77;">{cant}</b></span></div>', unsafe_allow_html=True)
    
    with col_middle:
        st.markdown("### ⭐ Mejores Zonas")
        best_zonas = stats_zona.nlargest(5, 'Tasa')
        for i, row in best_zonas.iterrows():
            st.markdown(f'<div style="margin: 8px 0;"><span style="color: #FFD93D; font-weight: bold;">📍</span> <span style="color: #FFF;">{row["Zona"]}: <b style="color: #FF6B6B;">{row["Tasa"]:.0%}</b></span></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown("### 📦 Categorías Exitosas")
        # CORREGIDO v6.2 - Mostrar TODAS las categorías con más de 0% tasa
        best_cats = stats_categoria[stats_categoria['Tasa'] > 0].nlargest(5, 'Tasa')
        if best_cats.empty:
            st.markdown('<span style="color: #AAA;">Sin datos disponibles</span>', unsafe_allow_html=True)
        else:
            for i, row in best_cats.iterrows():
                st.markdown(f'<div style="margin: 8px 0;"><span style="color: #6BCB77; font-weight: bold;">✓</span> <span style="color: #FFF;">{row["Categoria"]}: <b style="color: #FFD93D;">{row["Tasa"]:.0%}</b></span></div>', unsafe_allow_html=True)

def mostrar_analisis_usuarios(df):
    """Análisis de desempeño por usuario - v6.2"""
    
    st.markdown("### 👤 Análisis de Vendedores")
    
    usuarios = sorted(df['Usuario_Interno'].unique())
    usuario_selected = st.selectbox("Selecciona un vendedor:", usuarios, key='usuario_select')
    
    df_usuario = df[df['Usuario_Interno'] == usuario_selected]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cotizaciones Totales", len(df_usuario))
    
    with col2:
        ventas_usuario = len(df_usuario[df_usuario['¿Adjudicado?'] == 1])
        st.metric("Ventas Ganadas", ventas_usuario)
    
    with col3:
        tasa_usuario = df_usuario['¿Adjudicado?'].mean() if len(df_usuario) > 0 else 0
        st.metric("Tasa de Cierre", f"{tasa_usuario:.1%}")

def mostrar_analisis_clientes(df):
    """Análisis de clientes y oportunidades de cross-sell - v6.2 CORREGIDO"""
    
    st.markdown("### 👥 Análisis de Clientes & Cross-Sell")
    
    clientes = sorted(df['Cliente'].unique())
    cliente_selected = st.selectbox("Selecciona un cliente:", clientes, key='cliente_select')
    
    df_cliente = df[df['Cliente'] == cliente_selected]
    
    # Info del cliente
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cotizaciones", len(df_cliente))
    
    with col2:
        ventas = len(df_cliente[df_cliente['¿Adjudicado?'] == 1])
        st.metric("Ventas Ganadas", ventas)
    
    with col3:
        tasa = df_cliente['¿Adjudicado?'].mean() if len(df_cliente) > 0 else 0
        st.metric("Tasa de Cierre", f"{tasa:.1%}")
    
    with col4:
        zonas_cliente = df_cliente['Zona Geográfica'].nunique()
        st.metric("Zonas", zonas_cliente)
    
    st.divider()
    
    # Cross-sell - CORREGIDO v6.2
    st.markdown("#### 🎯 Oportunidades de Cross-Sell")
    
    # Obtener todas las categorías donde el cliente ha ganado algo
    categorias_compradas = df_cliente[df_cliente['¿Adjudicado?'] == 1]['Categoria_Producto'].unique()
    
    # Obtener TODAS las categorías del dataset
    todas_categorias = sorted(df['Categoria_Producto'].unique())
    
    if len(categorias_compradas) > 0:
        st.markdown(f"**Comprado actualmente:** {', '.join(categorias_compradas)}")
    else:
        st.markdown("**Comprado actualmente:** Ninguna venta aún")
    
    # Encontrar categorías donde NO ha comprado
    categorias_oportunidad = [c for c in todas_categorias if c not in categorias_compradas]
    
    if categorias_oportunidad:
        st.markdown("**Oportunidades potenciales:**")
        for categoria in categorias_oportunidad:
            tasa_categoria = df[df['Categoria_Producto'] == categoria]['¿Adjudicado?'].mean()
            ventas_categoria = len(df[(df['Categoria_Producto'] == categoria) & (df['¿Adjudicado?'] == 1)])
            
            # Mostrar con más detalles
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"📦 **{categoria}**")
            with col2:
                st.metric("Tasa", f"{tasa_categoria:.0%}")
            with col3:
                st.metric("Ventas", ventas_categoria)
    else:
        st.markdown("**Oportunidades potenciales:** Este cliente ya ha comprado en todas las categorías disponibles ✅")

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

def main():
    init_db()
    
    # ========== LOGIN SYSTEM ==========
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="color: #FF6B6B; font-size: 48px;">📊</h1>
                <h2 style="color: #FFF;">Sales Intelligence</h2>
                <p style="color: #AAA;">Predicción de Ventas con IA</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            menu_login = st.radio("", ["Iniciar Sesión", "Registrarse"], horizontal=True)
            
            user = st.text_input("👤 Usuario")
            pwd = st.text_input("🔐 Contraseña", type="password")
            
            if st.button("Continuar", use_container_width=True, type="primary"):
                if menu_login == "Iniciar Sesión":
                    if user_auth(user, pwd, 'login'):
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user
                        st.rerun()
                    else:
                        st.error("❌ Credenciales inválidas")
                else:
                    if user_auth(user, pwd, 'signup'):
                        st.success("✅ Usuario creado. Por favor, inicia sesión.")
                    else:
                        st.error("❌ El usuario ya existe")
        
        return
    
    # ========== APP DASHBOARD ==========
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #FF6B6B; margin: 0;">📊 Sales AI</h2>
            <p style="color: #888; margin: 5px 0; font-size: 12px;">Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown(f"**👤 {st.session_state['user']}**")
        
        # MENÚ REORDENADO - ANÁLISIS AL FINAL
        menu = st.radio(
            "Menú Principal",
            ["⚙️ Entrenamiento", "📊 Dashboard", "👤 Vendedores", 
             "👥 Clientes", "🔮 Predicción", "📝 Historial", "📈 Análisis"],
            key='menu_principal'
        )
        
        st.divider()
        
        if st.button("🚪 Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # ========== CONTENIDO PRINCIPAL ==========
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #FFF; margin: 0; font-size: 32px;">📊 Sistema Predictivo de Ventas</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar dataset si no existe
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if st.session_state['df'] is None and menu != "⚙️ Entrenamiento":
        st.warning("⚠️ Primero debes entrenar el modelo. Ve a la sección '⚙️ Entrenamiento'")
    
    # MENÚ DE NAVEGACIÓN
    
    if menu == "⚙️ Entrenamiento":
        st.markdown("### ⚙️ Entrenar Modelo")
        
        file = st.file_uploader("📤 Sube tu dataset (Excel o CSV)", type=['xlsx', 'csv'])
        
        if file:
            with st.spinner("Cargando datos..."):
                df, err = be.load_data(file)
            
            if df is not None:
                st.success(f"✅ Dataset cargado: {len(df):,} registros")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Filas", f"{len(df):,}")
                
                with col2:
                    st.metric("Clientes", df['Cliente'].nunique())
                
                with col3:
                    st.metric("Zonas", df['Zona Geográfica'].nunique())
                
                with col4:
                    st.metric("Vendedores", df['Usuario_Interno'].nunique())
                
                st.divider()
                
                # Mostrar categorías disponibles
                st.markdown("#### 📦 Categorías de Producto")
                categorias = sorted(df['Categoria_Producto'].unique())
                cols = st.columns(len(categorias) if len(categorias) <= 4 else 4)
                for i, cat in enumerate(categorias):
                    with cols[i % 4]:
                        count = len(df[df['Categoria_Producto'] == cat])
                        st.metric(cat[:20], f"{count} registros")
                
                st.divider()
                
                if st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary"):
                    with st.spinner("Entrenando modelo y calculando métricas avanzadas..."):
                        model, metrics, artifacts = be.train_model_logic(df)
                        be.save_model_artifacts(model, artifacts)
                        st.session_state['df'] = df
                    
                    st.success("✅ ¡Modelo entrenado exitosamente!")
                    
                    # === SECCIÓN 1: EXPLICACIÓN MATEMÁTICA DE LA EXACTITUD ===
                    st.markdown("### 🔍 Desglose de Métricas del Modelo")
                    
                    # Recuperar valores de la matriz de confusión
                    cm = metrics['confusion_matrix']
                    tn, fp = cm[0][0], cm[0][1] # True Negative, False Positive
                    fn, tp = cm[1][0], cm[1][1] # False Negative, True Positive
                    total = metrics['n_samples']
                    
                    # 1. KPIs Generales
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    with kpi1:
                        st.metric("Exactitud (Accuracy)", f"{metrics['accuracy']:.2%}", help="Porcentaje total de predicciones correctas")
                    with kpi2:
                        precision = metrics['classification_report']['1']['precision']
                        st.metric("Precisión (Precision)", f"{precision:.2%}", help="De las que dijo que eran Ganadas, ¿cuántas lo fueron realmente?")
                    with kpi3:
                        recall = metrics['classification_report']['1']['recall']
                        st.metric("Sensibilidad (Recall)", f"{recall:.2%}", help="De todas las Ganadas reales, ¿cuántas detectó el modelo?")
                    with kpi4:
                        st.metric("Área bajo curva (AUC)", f"{metrics['roc_auc']:.2%}", help="Capacidad del modelo para distinguir clases")

                    st.divider()

                    # === SECCIÓN 2: VISUALIZACIÓN GRÁFICA DEL ERROR (CONFUSION MATRIX) ===
                    col_matrix, col_formula = st.columns([1, 1])
                    
                    with col_matrix:
                        st.markdown("#### 1. Matriz de Confusión")
                        # Crear Heatmap con Plotly
                        z = [[tn, fp], [fn, tp]]
                        x = ['Predicho: Perdida', 'Predicho: Ganada']
                        y = ['Real: Perdida', 'Real: Ganada']
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=z, x=x, y=y,
                            text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
                            texttemplate="%{text}", textfont={"size": 16},
                            colorscale='Viridis', showscale=False
                        ))
                        fig_cm.update_layout(
                            title="¿Dónde acierta y falla el modelo?",
                            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                            font=dict(color='white'), height=300,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col_formula:
                        st.markdown("#### 2. ¿Cómo se calcula la Accuracy?")
                        st.info("La exactitud es la suma de los aciertos dividida por el total.")
                        
                        # Explicación con LaTeX
                        st.latex(r'''
                            Accuracy = \frac{TruePositives + TrueNegatives}{Total}
                        ''')
                        
                        st.markdown(f"""
                        Aplicando a tus datos:
                        * **Aciertos Totales:** {tp} (Ganadas bien predichas) + {tn} (Perdidas bien predichas) = **{tp+tn}**
                        * **Total Datos:** {total}
                        * **Cálculo:** {tp+tn} / {total} = **{metrics['accuracy']:.2%}**
                        """)

                    st.divider()

                    # === SECCIÓN 3: IMPORTANCIA DE VARIABLES (EXPLICABILIDAD) ===
                    st.markdown("#### 3. Factores Determinantes (Top Drivers)")
                    st.markdown("¿Qué variables aumentan más la probabilidad de ganar?")
                    
                    # Convertir datos para gráfico
                    df_importance = pd.DataFrame(metrics['top_positive_features'])
                    
                    fig_imp = px.bar(
                        df_importance, x='Coef', y='Feature', orientation='h',
                        title="Variables que MÁS aportan al cierre exitoso",
                        color='Coef', color_continuous_scale='Greens'
                    )
                    fig_imp.update_layout(
                        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                        font=dict(color='white'), yaxis={'categoryorder':'total ascending'},
                        height=350
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.divider()
                    
                    st.markdown("#### 📊 Distribución por Categoría")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for i, cat in enumerate(artifacts['categorias']):
                        stats = next((s for s in artifacts['stats_categoria'] if s['Categoria'] == cat), None)
                        if stats:
                            with st.columns(4)[i % 4]:
                                st.metric(
                                    f"{cat[:20]}...",
                                    f"{stats['Tasa_Conversion']:.0%}",
                                    f"{int(stats['Ganadas'])} ganadas"
                                )
            else:
                st.error(f"❌ Error: {err}")
    
    elif menu == "📊 Dashboard":
        if st.session_state['df'] is not None:
            mostrar_dashboard_general(st.session_state['df'])
        else:
            st.info("Carga datos para ver el dashboard")
    
    elif menu == "👤 Vendedores":
        if st.session_state['df'] is not None:
            mostrar_analisis_usuarios(st.session_state['df'])
        else:
            st.info("Carga datos para ver análisis de vendedores")
    
    elif menu == "👥 Clientes":
        if st.session_state['df'] is not None:
            mostrar_analisis_clientes(st.session_state['df'])
        else:
            st.info("Carga datos para ver análisis de clientes")
    
    elif menu == "🔮 Predicción":
        if st.session_state['df'] is not None:
            st.markdown("### 🔮 Simulador de Predicción")
            
            model, artifacts = be.load_model_artifacts()
            
            if model is None:
                st.error("❌ Modelo no encontrado. Entrena primero.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    tipo_cliente = st.radio("Tipo de Cliente", ["Existente", "Nuevo"])
                    
                    if tipo_cliente == "Existente":
                        client = st.selectbox("Cliente", artifacts['unique_clients'])
                        is_new = False
                    else:
                        client = st.text_input("Nombre del Cliente")
                        is_new = True
                
                with col2:
                    zona = st.selectbox("Zona Geográfica", artifacts['unique_zones'])
                    categoria = st.selectbox("Categoría de Producto", sorted(artifacts['categorias']))
                
                usuario_sel = st.selectbox("Vendedor Asignado", sorted(artifacts['unique_usuarios']))
                
                if st.button("🚀 Predecir Adjudicación", use_container_width=True, type="primary"):
                    if client:
                        prob, label, recomendacion, color = be.make_prediction(
                            model, artifacts, client, zona, categoria, usuario_sel, is_new
                        )
                        
                        # Mostrar tarjeta primero
                        be.mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color)
                        
                        st.divider()
                        
                        # Guardar en BD - v6.2
                        saved = save_prediction_db(
                            st.session_state['user'], 
                            client, 
                            zona, 
                            categoria, 
                            prob, 
                            label, 
                            recomendacion
                        )
                        
                        # Mostrar resultado
                        if saved:
                            st.success("✅ Predicción guardada en el historial")
                        else:
                            st.error("❌ Error al guardar predicción. Contacta al soporte.")
        else:
            st.info("Carga datos para hacer predicciones")
    
    elif menu == "📝 Historial":
        st.markdown("### 📝 Historial de Predicciones")
        
        df_hist = get_history(user_id=st.session_state['user'], days=90)
        
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Sin predicciones en tu historial")
    
    elif menu == "📈 Análisis":
        if st.session_state['df'] is not None:
            st.markdown("### 📈 Análisis Avanzado")
            
            df_hist = get_history(days=30)
            
            if not df_hist.empty:
                st.markdown("#### Predicciones Recientes")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Predicciones", len(df_hist))
                
                with col2:
                    st.metric("Prob. Promedio", f"{df_hist['prob'].mean():.1%}")
                
                with col3:
                    alta_prob = len(df_hist[df_hist['prob'] > 0.65])
                    st.metric("Alta Probabilidad", alta_prob)
                
                st.divider()
                
                st.markdown("#### Histórico de Predicciones")
                st.dataframe(df_hist, use_container_width=True, height=400)
            else:
                st.info("Sin predicciones aún")
        else:
            st.info("Carga datos para ver análisis")

if __name__ == '__main__':
    main()
