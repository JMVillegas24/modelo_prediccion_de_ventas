"""
SALES INTELLIGENCE PLATFORM - VERSIÓN UNIFICADA (ALL-IN-ONE)
Este archivo contiene TODA la lógica del Backend y el Frontend combinada.
Conserva exactitud visual, gráficas, pestañas y estilos del diseño original.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import json
import os
import sqlite3
import hashlib
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN INICIAL (OBLIGATORIO AL INICIO)
# ==========================================

st.set_page_config(
    page_title="Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LÓGICA DEL BACKEND (CÁLCULOS, ESTILOS, ML)
# ==========================================

MODEL_PATH = 'model.pkl'
ARTIFACTS_PATH = 'artifacts.json'

# --- TEMA Y ESTILOS CSS ---
def aplicar_tema_oscuro():
    """Aplica tema oscuro y estilos personalizados a la aplicación Streamlit"""
    tema_css = """
    <style>
    :root {
        --color-primary: #FF6B6B;
        --color-secondary: #4ECDC4;
        --color-success: #6BCB77;
        --color-warning: #FFD93D;
    }
    
    body {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stMetric {
        background-color: #161B22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363D;
    }
    
    .stDataFrame {
        background-color: #0D1117;
    }
    </style>
    """
    st.markdown(tema_css, unsafe_allow_html=True)

# --- UTILIDADES ---

def formatear_porcentaje(valor):
    """Convierte decimal a porcentaje con 1 decimal"""
    try:
        if pd.isna(valor):
            return "0.0%"
        return f"{float(valor) * 100:.1f}%"
    except:
        return "0.0%"

def obtener_estadisticas_zona(df):
    """Genera estadísticas de zona para gráficos"""
    try:
        stats_zona = df.groupby('Zona Geográfica').agg({
            '¿Adjudicado?': ['sum', 'count']
        }).reset_index()
        
        stats_zona.columns = ['Zona', 'Ganadas', 'Total']
        stats_zona['Tasa_Conversion'] = (stats_zona['Ganadas'] / stats_zona['Total']).round(3)
        stats_zona['Tasa_Porcentaje'] = stats_zona['Tasa_Conversion'].apply(formatear_porcentaje)
        
        return stats_zona
    except Exception as e:
        return pd.DataFrame()

# --- MAPEO DE CATEGORÍAS ---

CATEGORIA_KEYWORDS = {
    'Válvulas y Actuadores': [
        'válvula', 'valvula', 'actuador', 'jamesbury', 'neles', 'metso', 'newco', 'walworth',
        'bola', 'compuerta', 'mariposa', 'angulares', 'reguladora', 'gv', 'btf', 'bv'
    ],
    'Instrumentos de Medición': [
        'indicador', 'medidor', 'medidores', 'transmisor', 'sensor', 'sonda',
        'magnetrol', 'jerguson', 'westlock', 'nivel', 'presión', 'temperatura', 'flujo',
        'rtd', 'manometro', 'termometro', 'rotametro', 'switch', 'interruptor', 'analizador',
        'iq70', 'iq90', 'iqs20', 'báscula', 'merrick', 'wedgmeter', 'varec', 'modulevel', 'teltru',
        'instrumentacion', 'instrumentación', 'instrumentos'
    ],
    'Controles Eléctricos': [
        'arrancador', 'variador', 'inversor', 'abb', 'siemens', 'plc',
        'controlador', 'módulo', 'modulo', 'fuente', 'electronico', 'electromagnético',
        'electrónico', 'motor', 'motorizado', 'ac500', 'sm1000',
        'elect', 'thermostat', 'chromalox', 'moxa', 'protector', 'transientes'
    ],
    'Equipos de Automatización': [
        'automatización', 'automatizar', 'sistema de control', 'autoclaves',
        'horno', 'unitronics', 'fieldlogger', 'novus', 'generador', 'hipoclorito', 'cromatógrafo',
        'merrick', 'clorador'
    ],
    'Accesorios y Repuestos': [
        'repuesto', 'repuestos', 'accesorio', 'accesorios', 'empaque', 'empaques',
        'sello', 'sellos', 'oring', 'brida', 'tubing', 'manifold', 'kit', 'cabezote', 'tornillo',
        'disco', 'ruptura', 'correa', 'banda', 'alambre', 'bellofram', 'tarjeta'
    ],
    'Servicios y Consultoría': [
        'servicio', 'servicios', 'mantenimiento', 'calibración', 'calibrar',
        'consultoría', 'montaje', 'instalación', 'reparación', 'asistencia técnica', 'toma muestras'
    ]
}

COLORES_CATEGORIAS = {
    'Válvulas y Actuadores': '#FF6B6B',
    'Instrumentos de Medición': '#4ECDC4',
    'Controles Eléctricos': '#FFD93D',
    'Equipos de Automatización': '#6BCB77',
    'Accesorios y Repuestos': '#A78BFA',
    'Servicios y Consultoría': '#FB7185',
    'Otros': '#9CA3AF'
}

def extraer_categoria(solicitud):
    """Extrae categoría de la solicitud basado en palabras clave"""
    if pd.isna(solicitud):
        return 'Sin Categoría'
    
    solicitud_lower = str(solicitud).lower().strip()
    
    for categoria, keywords in CATEGORIA_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in solicitud_lower:
                return categoria
    
    return 'Otros'

# --- LIMPIEZA DE DATOS ---

COLUMN_MAPPING = {
    'Cliente': ['Cliente', 'client', 'customer', 'CLIENTE'],
    'Zona Geográfica': ['Zona Geográfica', 'Zona', 'zona', 'Region', 'región', 'ZONA', 'region'],
    'Usuario_Interno': ['Usuario_Interno', 'Usuario interno', 'usuario interno', 'Vendedor', 'vendedor', 
                        'User', 'usuario', 'USUARIO', 'User_ID', 'user_id', 'Salesperson', 'salesperson',
                        'Usuario_interno', 'usuario_interno'],
    'Solicitud': ['Solicitud', 'Solicitud del Cliente', 'solicitud', 'Request', 'Requerimiento'],
    '¿Adjudicado?': ['¿Adjudicado?', 'Adjudicado', 'adjudicado', 'Adjudicada', 'won', 
                     'Won', 'ganada', 'Ganada', 'is_won', 'Sale_Status']
}

def normalizar_columnas(df):
    """Normaliza nombres de columnas"""
    df_normalized = df.copy()
    renaming_dict = {}
    
    for required_col, alternatives in COLUMN_MAPPING.items():
        encontrada = False
        if required_col in df_normalized.columns:
            encontrada = True
            continue
        if not encontrada:
            for alt_name in alternatives:
                if alt_name in df_normalized.columns:
                    renaming_dict[alt_name] = required_col
                    encontrada = True
                    break
    
    if renaming_dict:
        df_normalized = df_normalized.rename(columns=renaming_dict)
    
    return df_normalized

def limpiar_valores(df):
    """Limpia valores y extrae categoría"""
    df_limpio = df.copy()
    
    cols_to_strip = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Solicitud']
    for col in cols_to_strip:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].astype(str).str.strip()
    
    if 'Solicitud' in df_limpio.columns:
        df_limpio['Categoria_Producto'] = df_limpio['Solicitud'].apply(extraer_categoria)
        df_limpio = df_limpio.dropna(subset=['Cliente', 'Solicitud'])
    
    return df_limpio

def ordenar_lista_segura(lista):
    """Ordena lista de forma segura eliminando nulos"""
    try:
        lista_str = [str(x).strip() for x in lista if pd.notna(x) 
                     and str(x).strip() not in ['nan', 'None', 'NaN', 'Sin Categoría']]
        return sorted(list(set(lista_str)))
    except Exception as e:
        return [str(x).strip() for x in lista if pd.notna(x)]

def load_data(file):
    """Carga datos Excel/CSV"""
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return None, "❌ Formato no soportado. Usa Excel (.xlsx) o CSV (.csv)"
        
        try:
            df = normalizar_columnas(df)
        except ValueError as e:
            return None, str(e)
        
        try:
            df = limpiar_valores(df)
        except Exception as e:
            return None, f"❌ Error limpiando datos: {str(e)}"
        
        columnas_requeridas = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 
                              'Categoria_Producto', '¿Adjudicado?']
        
        for col in columnas_requeridas:
            if col not in df.columns:
                return None, f"❌ Falta columna: {col}"
        
        try:
            df['¿Adjudicado?'] = df['¿Adjudicado?'].astype(int)
        except:
            df['¿Adjudicado?'] = df['¿Adjudicado?'].map(
                {1: 1, 0: 0, 'Sí': 1, 'SI': 1, 'Yes': 1, 
                 'No': 0, 'NO': 0, True: 1, False: 0}
            ).fillna(0).astype(int)
        
        return df, None
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# --- ENTRENAMIENTO Y ML ---

def train_model_logic(df):
    """Entrena modelo ML con métricas detalladas"""
    try:
        features = ['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        target = '¿Adjudicado?'
        
        for col in features + [target]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada")
        
        df_model = df[features + [target]].copy()
        
        # Limpieza adicional para asegurar strings
        for col in features:
            df_model[col] = df_model[col].astype(str).str.strip()
            
        df_model = df_model.dropna()
        
        if len(df_model) == 0: raise ValueError("Dataset vacío")
        
        df_encoded = pd.get_dummies(df_model, columns=features, drop_first=True)
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        # ENTRENAMIENTO
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # --- CÁLCULOS DE MÉTRICAS ---
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # 1. Matriz de Confusión
        cm = confusion_matrix(y, y_pred) # [[TN, FP], [FN, TP]]
        
        # 2. Reporte de Clasificación (Precision, Recall, F1)
        report = classification_report(y, y_pred, output_dict=True)
        
        # 3. Curva ROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 4. Importancia de Variables (Coeficientes)
        coefs = pd.DataFrame({
            'Feature': X.columns,
            'Coef': model.coef_[0]
        }).sort_values(by='Coef', ascending=False)
        
        top_positive = coefs.head(10).to_dict('records')
        top_negative = coefs.tail(10).to_dict('records')

        # Stats por categoria
        stats_categoria = []
        categorias_ordenadas = ordenar_lista_segura(df['Categoria_Producto'].unique())
        for cat in categorias_ordenadas:
            df_cat = df[df['Categoria_Producto'] == cat]
            ganadas = len(df_cat[df_cat['¿Adjudicado?'] == 1])
            total = len(df_cat)
            tasa = (ganadas / total) if total > 0 else 0
            stats_categoria.append({'Categoria': cat, 'Total': total, 'Ganadas': ganadas, 'Tasa_Conversion': tasa})
        
        # ARTIFACTS
        artifacts = {
            'feature_names': list(X.columns),
            'unique_clients': ordenar_lista_segura(df['Cliente'].unique()),
            'unique_zones': ordenar_lista_segura(df['Zona Geográfica'].unique()),
            'categorias': categorias_ordenadas,
            'unique_usuarios': ordenar_lista_segura(df['Usuario_Interno'].unique()),
            'stats_categoria': stats_categoria,
            'model_accuracy': model.score(X, y)
        }
        
        # METRICS
        metrics = {
            'accuracy': model.score(X, y),
            'confusion_matrix': cm.tolist(), 
            'classification_report': report,
            'roc_auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'n_samples': len(df),
            'n_features': len(X.columns)
        }
        
        return model, metrics, artifacts
    
    except Exception as e:
        raise ValueError(f"❌ Error: {str(e)}")

# --- GUARDADO Y CARGA ---

def save_model_artifacts(model, artifacts):
    """Guarda modelo y artifacts"""
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ARTIFACTS_PATH, 'w') as f:
            json.dump(artifacts, f)
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def load_model_artifacts():
    """Carga modelo y artifacts"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACTS_PATH):
            return None, None
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ARTIFACTS_PATH, 'r') as f:
            artifacts = json.load(f)
        return model, artifacts
    except:
        return None, None

# --- PREDICCIÓN ---

def make_prediction(model, artifacts, client, zona, categoria, usuario, is_new=False):
    """Realiza predicción"""
    try:
        client = str(client).strip()
        zona = str(zona).strip()
        categoria = str(categoria).strip()
        usuario = str(usuario).strip()
        
        prediction_data = pd.DataFrame({
            'Cliente': [client],
            'Zona Geográfica': [zona],
            'Usuario_Interno': [usuario],
            'Categoria_Producto': [categoria]
        })
        
        prediction_encoded = pd.get_dummies(
            prediction_data,
            columns=['Cliente', 'Zona Geográfica', 'Usuario_Interno', 'Categoria_Producto']
        )
        
        for feature in artifacts['feature_names']:
            if feature not in prediction_encoded.columns:
                prediction_encoded[feature] = 0
        
        X_pred = prediction_encoded[artifacts['feature_names']]
        
        prob = model.predict_proba(X_pred)[0][1]
        label = "Probable" if prob > 0.5 else "Improbable"
        
        stats_cat = next(
            (s for s in artifacts['stats_categoria'] if s['Categoria'] == categoria),
            None
        )
        
        recomendacion = "Sin datos históricos"
        if stats_cat:
            categoria_tasa = stats_cat['Tasa_Conversion']
            categoria_ventas = stats_cat['Ganadas']
            categoria_tasa_pct = formatear_porcentaje(categoria_tasa)
            
            if prob > 0.7:
                recomendacion = f"🟢 {categoria}: {categoria_tasa_pct} ({categoria_ventas} ventas)"
            elif prob > 0.5:
                recomendacion = f"🟡 {categoria}: {categoria_tasa_pct} ({categoria_ventas} ventas)"
            else:
                recomendacion = f"🔴 {categoria}: {categoria_tasa_pct} ({categoria_ventas} ventas)"
        else:
            recomendacion = "⚠️ Categoría no encontrada"
        
        color = '#6BCB77' if prob > 0.7 else '#FFD93D' if prob > 0.5 else '#FF6B6B'
        
        return prob, label, recomendacion, color
    
    except Exception as e:
        return 0, "Error", f"❌ Error: {str(e)}", '#FF6B6B'

# --- VISUALIZACIONES Y COMPONENTES VISUALES ---

def grafico_barras_horizontal(df, x, y, titulo, color='#FF6B6B'):
    fig = px.bar(df, x=x, y=y, orientation='h', title=f'<b>{titulo}</b>',
                 color_discrete_sequence=[color], height=400, text=x)
    
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#111823',
        font=dict(color='#FFFFFF', size=12), showlegend=False,
        title_font_size=16, xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#31333D'),
        yaxis=dict(showgrid=False), margin=dict(l=100, r=50, t=80, b=50),
        hovermode='y unified'
    )
    fig.update_traces(textposition='outside', textfont_size=11)
    return fig

def grafico_pie_categorias(df, categoria_col, titulo, colores=None):
    if colores is None:
        colores = list(COLORES_CATEGORIAS.values())
    
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    if len(df) == 0:
        return px.pie(pd.DataFrame({'Categoría': ['Sin datos'], 'Valor': [1]}),
            names='Categoría', values='Valor', title=f'<b>{titulo}</b>',
            color_discrete_sequence=['#CCCCCC'])
    
    value_cols = [col for col in df.columns if col not in [categoria_col, 'Tasa_Conversion', 'Tasa_Porcentaje']]
    if len(value_cols) == 0:
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    else:
        if 'Ganadas' in value_cols: value_col = 'Ganadas'
        elif 'Total' in value_cols: value_col = 'Total'
        else: value_col = value_cols[0]
    
    fig = px.pie(df, names=categoria_col, values=value_col, title=f'<b>{titulo}</b>',
        color_discrete_sequence=colores, height=450)
    
    fig.update_layout(paper_bgcolor='#0E1117', font=dict(color='#FFFFFF', size=12),
        title_font_size=16, margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True, legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0)', bordercolor='#31333D', borderwidth=1))
    
    fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value}<extra></extra>', textposition='auto', textinfo='label+percent')
    return fig

def mostrar_kpi_grande(titulo, valor, color='#4D96FF', icon='📊'):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 25px; border-radius: 12px; border-left: 5px solid {color};
                text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <p style="color: #888; margin: 0; font-size: 13px; text-transform: uppercase;">
            {icon} {titulo}</p>
        <p style="color: {color}; margin: 12px 0 0 0; font-size: 36px; font-weight: bold;">
            {valor}</p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 25px; border-radius: 12px; border-left: 5px solid {color};
                margin: 20px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <h3 style="color: {color}; margin-top: 0;">🎯 Predicción</h3>
        <p style="color: #FFFFFF; font-size: 14px;"><strong>Categoría:</strong> {categoria}</p>
        <p style="color: #FFFFFF; font-size: 14px;">
            <strong>Probabilidad:</strong> <span style="color: {color}; font-size: 28px;">
                {prob:.1%}</span></p>
        <p style="color: #AAA; font-size: 13px;">{recomendacion}</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 3. LÓGICA DEL FRONTEND (APP Y BD)
# ==========================================

aplicar_tema_oscuro()

# --- GESTIÓN DE BD ---

DB_PATH = 'sales_app.db'

def init_db():
    """Inicializa la BD"""
    try:
        db_exists = os.path.exists(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
            (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, client_name TEXT,
            zone TEXT, category_product TEXT, prob REAL, label TEXT, 
            recommendation TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error en init_db: {str(e)}")
        return False

def user_auth(username, password, mode='login'):
    """Autenticación de usuarios"""
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
        return False
    return False

def save_prediction_db(user, client, zone, category_product, prob, label, recommendation):
    """Guarda predicción"""
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
        return False

def get_history(user_id=None, days=30):
    """Obtiene histórico"""
    try:
        conn = sqlite3.connect(DB_PATH)
        if user_id:
            query = f"SELECT * FROM predictions WHERE user_id = ? AND timestamp > datetime('now', '-{days} days') ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:
            query = f"SELECT * FROM predictions WHERE timestamp > datetime('now', '-{days} days') ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        conn.close()
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --- FUNCIONES DE ANÁLISIS ---

def mostrar_dashboard_general(df):
    """Dashboard general con KPIs y gráficos principales"""
    st.markdown("### 📊 Dashboard de Inteligencia de Ventas")
    
    col1, col2, col3, col4 = st.columns(4)
    total_cotizaciones = len(df)
    ventas_ganadas = len(df[df['¿Adjudicado?'] == 1])
    tasa_conversion = df['¿Adjudicado?'].mean()
    
    with col1: mostrar_kpi_grande("Total de Cotizaciones", f"{total_cotizaciones:,}", color='#4D96FF', icon='📋')
    with col2: mostrar_kpi_grande("Ventas Ganadas", f"{ventas_ganadas:,}", color='#6BCB77', icon='✅')
    with col3: mostrar_kpi_grande("Tasa de Conversión", f"{tasa_conversion:.1%}", color='#FFD93D', icon='📈')
    with col4:
        unique_clientes = df['Cliente'].nunique()
        mostrar_kpi_grande("Clientes Únicos", f"{unique_clientes:,}", color='#FF6B6B', icon='👥')
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    with col_left:
        stats_categoria = df.groupby('Categoria_Producto')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_categoria.columns = ['Categoria', 'Total', 'Ganadas', 'Tasa']
        stats_categoria = stats_categoria.sort_values('Tasa', ascending=True)
        fig_cat = grafico_barras_horizontal(stats_categoria, 'Tasa', 'Categoria', '📊 Tasa de Conversión por Categoría de Producto', '#FF6B6B')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_right:
        stats_zona = df.groupby('Zona Geográfica')['¿Adjudicado?'].agg(['count', 'sum', 'mean']).reset_index()
        stats_zona.columns = ['Zona', 'Total', 'Ganadas', 'Tasa']
        colores_zona = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#6BCB77']
        fig_zona = grafico_pie_categorias(stats_zona, 'Zona', '🌍 Distribución por Zona', colores_zona)
        st.plotly_chart(fig_zona, use_container_width=True)
    
    st.divider()
    
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
        best_cats = stats_categoria[stats_categoria['Tasa'] > 0].nlargest(5, 'Tasa')
        if best_cats.empty:
            st.markdown('<span style="color: #AAA;">Sin datos disponibles</span>', unsafe_allow_html=True)
        else:
            for i, row in best_cats.iterrows():
                st.markdown(f'<div style="margin: 8px 0;"><span style="color: #6BCB77; font-weight: bold;">✓</span> <span style="color: #FFF;">{row["Categoria"]}: <b style="color: #FFD93D;">{row["Tasa"]:.0%}</b></span></div>', unsafe_allow_html=True)

def mostrar_analisis_usuarios(df):
    st.markdown("### 👤 Análisis de Vendedores")
    usuarios = sorted(df['Usuario_Interno'].unique())
    usuario_selected = st.selectbox("Selecciona un vendedor:", usuarios, key='usuario_select')
    df_usuario = df[df['Usuario_Interno'] == usuario_selected]
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Cotizaciones Totales", len(df_usuario))
    with col2:
        ventas_usuario = len(df_usuario[df_usuario['¿Adjudicado?'] == 1])
        st.metric("Ventas Ganadas", ventas_usuario)
    with col3:
        tasa_usuario = df_usuario['¿Adjudicado?'].mean() if len(df_usuario) > 0 else 0
        st.metric("Tasa de Cierre", f"{tasa_usuario:.1%}")

def mostrar_analisis_clientes(df):
    st.markdown("### 👥 Análisis de Clientes & Cross-Sell")
    clientes = sorted(df['Cliente'].unique())
    cliente_selected = st.selectbox("Selecciona un cliente:", clientes, key='cliente_select')
    df_cliente = df[df['Cliente'] == cliente_selected]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Cotizaciones", len(df_cliente))
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
    st.markdown("#### 🎯 Oportunidades de Cross-Sell")
    categorias_compradas = df_cliente[df_cliente['¿Adjudicado?'] == 1]['Categoria_Producto'].unique()
    todas_categorias = sorted(df['Categoria_Producto'].unique())
    
    if len(categorias_compradas) > 0:
        st.markdown(f"**Comprado actualmente:** {', '.join(categorias_compradas)}")
    else:
        st.markdown("**Comprado actualmente:** Ninguna venta aún")
    
    categorias_oportunidad = [c for c in todas_categorias if c not in categorias_compradas]
    
    if categorias_oportunidad:
        st.markdown("**Oportunidades potenciales:**")
        for categoria in categorias_oportunidad:
            tasa_categoria = df[df['Categoria_Producto'] == categoria]['¿Adjudicado?'].mean()
            ventas_categoria = len(df[(df['Categoria_Producto'] == categoria) & (df['¿Adjudicado?'] == 1)])
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1: st.markdown(f"📦 **{categoria}**")
            with col2: st.metric("Tasa", f"{tasa_categoria:.0%}")
            with col3: st.metric("Ventas", ventas_categoria)
    else:
        st.markdown("**Oportunidades potenciales:** Este cliente ya ha comprado en todas las categorías disponibles ✅")

# ==========================================
# 4. INTERFAZ PRINCIPAL (MAIN LOOP)
# ==========================================

def main():
    init_db()
    
    # LOGIN SYSTEM
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
    
    # APP DASHBOARD (Sidebar)
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #FF6B6B; margin: 0;">📊 Sales AI</h2>
            <p style="color: #888; margin: 5px 0; font-size: 12px;">Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown(f"**👤 {st.session_state['user']}**")
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
    
    # CONTENIDO PRINCIPAL
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #FFF; margin: 0; font-size: 32px;">📊 Sistema Predictivo de Ventas</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if st.session_state['df'] is None and menu != "⚙️ Entrenamiento":
        st.warning("⚠️ Primero debes entrenar el modelo. Ve a la sección '⚙️ Entrenamiento'")
    
    # LÓGICA DE MENÚS
    if menu == "⚙️ Entrenamiento":
        st.markdown("### ⚙️ Entrenar Modelo")
        file = st.file_uploader("📤 Sube tu dataset (Excel o CSV)", type=['xlsx', 'csv'])
        
        if file:
            with st.spinner("Cargando datos..."):
                df, err = load_data(file)
            
            if df is not None:
                st.success(f"✅ Dataset cargado: {len(df):,} registros")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Filas", f"{len(df):,}")
                with col2: st.metric("Clientes", df['Cliente'].nunique())
                with col3: st.metric("Zonas", df['Zona Geográfica'].nunique())
                with col4: st.metric("Vendedores", df['Usuario_Interno'].nunique())
                st.divider()
                
                if st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary"):
                    with st.spinner("Entrenando modelo y calculando métricas avanzadas..."):
                        model, metrics, artifacts = train_model_logic(df)
                        save_model_artifacts(model, artifacts)
                        st.session_state['df'] = df
                    
                    st.success("✅ ¡Modelo entrenado exitosamente!")
                    
                    # MÉTRICAS DETALLADAS (Visual idéntico al original)
                    st.markdown("### 🔍 Desglose de Métricas del Modelo")
                    cm = metrics['confusion_matrix']
                    tn, fp = cm[0][0], cm[0][1]
                    fn, tp = cm[1][0], cm[1][1]
                    total = metrics['n_samples']
                    
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    with kpi1: st.metric("Exactitud (Accuracy)", f"{metrics['accuracy']:.2%}")
                    with kpi2: st.metric("Precisión", f"{metrics['classification_report']['1']['precision']:.2%}")
                    with kpi3: st.metric("Sensibilidad", f"{metrics['classification_report']['1']['recall']:.2%}")
                    with kpi4: st.metric("AUC", f"{metrics['roc_auc']:.2%}")

                    st.divider()
                    col_matrix, col_formula = st.columns([1, 1])
                    
                    with col_matrix:
                        st.markdown("#### 1. Matriz de Confusión")
                        z = [[tn, fp], [fn, tp]]
                        x = ['Predicho: Perdida', 'Predicho: Ganada']
                        y = ['Real: Perdida', 'Real: Ganada']
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=z, x=x, y=y,
                            text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
                            texttemplate="%{text}", textfont={"size": 16},
                            colorscale='Viridis', showscale=False
                        ))
                        fig_cm.update_layout(height=300, paper_bgcolor='#0E1117', font=dict(color='white'))
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col_formula:
                        st.markdown("#### 2. ¿Cómo se calcula la Accuracy?")
                        st.info("La exactitud es la suma de los aciertos dividida por el total.")
                        st.latex(r'''Accuracy = \frac{TruePositives + TrueNegatives}{Total}''')
                        st.markdown(f"**Cálculo:** ({tp} + {tn}) / {total} = **{metrics['accuracy']:.2%}**")

                    st.divider()
                    st.markdown("#### 3. Factores Determinantes")
                    df_importance = pd.DataFrame(metrics['top_positive_features'])
                    fig_imp = px.bar(df_importance, x='Coef', y='Feature', orientation='h',
                                     title="Variables que MÁS aportan al cierre exitoso",
                                     color='Coef', color_continuous_scale='Greens')
                    fig_imp.update_layout(paper_bgcolor='#0E1117', font=dict(color='white'),
                                          yaxis={'categoryorder':'total ascending'}, height=350)
                    st.plotly_chart(fig_imp, use_container_width=True)
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
            model, artifacts = load_model_artifacts()
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
                        prob, label, recomendacion, color = make_prediction(
                            model, artifacts, client, zona, categoria, usuario_sel, is_new
                        )
                        mostrar_tarjeta_prediccion(prob, categoria, recomendacion, color)
                        save_prediction_db(st.session_state['user'], client, zona, categoria, prob, label, recomendacion)
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
                with col1: st.metric("Total de Predicciones", len(df_hist))
                with col2: st.metric("Prob. Promedio", f"{df_hist['prob'].mean():.1%}")
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