# Sales Intelligence Platform — Documentación de Implementación

Resumen
-------
Proyecto "Sales Intelligence Platform": aplicación web unificada (backend + frontend) desarrollada con Streamlit para entrenamiento de un modelo predictivo (LogisticRegression) y visualización de métricas y dashboards comerciales.

Estructura del proyecto
-----------------------
- `app.py` : Código principal (UI Streamlit, lógica de limpieza, ML, persistencia y visualizaciones).
- `model.pkl` : Archivo creado al entrenar el modelo (serializado con `pickle`).
- `artifacts.json` : Metadatos y artefactos del entrenamiento (features, listas únicas, estadísticas).
- `sales_app.db` : Base de datos SQLite donde se guardan usuarios y predicciones.

Requisitos
----------
- Python 3.8+ (recomendado 3.10+)
- Dependencias listadas en `requirements.txt`.

Instalación y ejecución local
-----------------------------
1. Clonar o copiar el repositorio en tu equipo.
2. Crear y activar un entorno virtual (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

4. Ejecutar la aplicación Streamlit:

```powershell
streamlit run app.py
```

Flujo de uso de la app
----------------------
- Registro / Inicio de sesión: se almacenan usuarios en la base de datos SQLite (`sales_app.db`).
- Entrenamiento: sube un archivo Excel (.xlsx) o CSV con las columnas requeridas. Pulsa `Entrenar Modelo` para generar `model.pkl` y `artifacts.json`.
- Dashboard: visualizaciones de conversión por categoría, zona y vendedores.
- Predicción: usa el modelo entrenado para predecir probabilidad de adjudicación; guarda predicciones en la BD.
- Historial: visualiza predicciones guardadas por usuario.

Formato de datos esperado
-------------------------
El cargador es flexible con nombres de columna (mapea alternativas). Columnas necesarias finales:
- `Cliente`
- `Zona Geográfica`
- `Usuario_Interno`
- `Solicitud` (de la cual se extrae `Categoria_Producto`)
- `¿Adjudicado?` (0/1 o equivalentes 'Sí'/'No')

Proceso de implementación (paso a paso)
-------------------------------------
1. Configuración de la página Streamlit
   - `st.set_page_config()` para título, icono y layout.

2. Tema y estilos
   - Función `aplicar_tema_oscuro()` inyecta CSS para apariencia oscura y componentes estilo tarjeta.

3. Normalización y limpieza de datos
   - `normalizar_columnas(df)` mapea nombres alternativos a un esquema estándar.
   - `limpiar_valores(df)` limpia strings, extrae `Categoria_Producto` usando `extraer_categoria()`.
   - `CATEGORIA_KEYWORDS` define heurísticas por palabra clave para clasificar `Solicitud` en categorías.

4. Ingeniería y transformación de features
   - Variables categóricas (`Cliente`, `Zona Geográfica`, `Usuario_Interno`, `Categoria_Producto`) se convierten a dummies via `pd.get_dummies()`.

5. Entrenamiento del modelo
   - `train_model_logic(df)` entrena `LogisticRegression(max_iter=1000)` sobre las dummies.
   - Calcula métricas: matriz de confusión, reportes de clasificación, AUC/ROC, coeficientes.
   - Guarda `model.pkl` (pickle) y `artifacts.json` con metadatos útiles para predicción y UI.

6. Predicción
   - `make_prediction(model, artifacts, ...)` construye un DataFrame con las mismas columnas que el modelo espera, completa columnas faltantes con 0 y calcula `predict_proba`.

7. Persistencia y BD
   - `init_db()` crea tablas `users` y `predictions` en `sales_app.db`.
   - `user_auth()` maneja registro/login (almacena hash SHA-256 de contraseñas).
   - `save_prediction_db()` y `get_history()` guardan/recuperan registros de predicciones.

8. Visualizaciones
   - Se usan `plotly.express` y `plotly.graph_objects` para gráficos (barras, pie, heatmap de matriz de confusión).

9. Interfaz y experiencia
   - Menú lateral con secciones: Entrenamiento, Dashboard, Vendedores, Clientes, Predicción, Historial, Análisis.
   - Componentes estilizados y tarjetas de KPI con HTML/CSS embebido.

Consideraciones de seguridad y despliegue
--------------------------------------
- No guardar contraseñas en texto plano (ya se usa SHA-256). Para producción se recomienda usar hashing con salt (bcrypt/argon2).
- Los archivos `model.pkl`, `artifacts.json` y `sales_app.db` contienen datos sensibles; protegerlos en despliegues.

Despliegue recomendado
----------------------
- Desplegar en un servicio que soporte Streamlit (Streamlit Community Cloud, Heroku, Azure Web Apps). Asegurar variables de entorno para credenciales.

Mejoras futuras sugeridas
-------------------------
- Separar backend y frontend para escalabilidad (API + UI).
- Añadir pruebas unitarias e integración continua.
- Mejorar pipeline de entrenamiento (validación cruzada, hold-out, logging de experimentos con MLflow).
- Añadir control de versiones del dataset y model registry.

Archivos creados
---------------
- `requirements.txt` — lista de dependencias.
- `.gitignore` — entradas comunes para proyectos Python y artefactos generados.

Contacto
-------
Para preguntas o mejoras, abre un issue o crea un PR con cambios propuestos.

-- Fin de la documentación --
