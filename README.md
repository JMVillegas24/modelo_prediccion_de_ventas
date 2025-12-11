# Proyecto IA Exporatorio V3

Descripción
- Proyecto con archivos principales `app.py` y `backend.py`.

Instalación
1. Crear y activar un entorno virtual (recomendado)

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows (cmd):

```cmd
python -m venv venv
venv\Scripts\activate
```

2. Actualizar pip e instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Estructura del proyecto

- `app.py` - Punto de entrada principal de la aplicación.
- `backend.py` - Lógica del backend / servicios.
- `requirements.txt` - Dependencias del proyecto.
- `.gitignore` - Archivos y carpetas ignoradas por git.

Proceso de implementación (Windows)

1) Preparar entorno

- Clonar el repositorio y moverse a la carpeta del proyecto.

```powershell
git clone <url-del-repo>
cd "Proyecto IA Exporatorio V3\app"
```

- Crear y activar el entorno virtual (ver comandos arriba).

2) Instalar dependencias

```powershell
pip install -r requirements.txt
```

Si `requirements.txt` está vacío, instala manualmente las dependencias necesarias, por ejemplo:

```powershell
pip install flask
pip install numpy
```

3) Configurar variables de entorno (si aplica)

- Crea un archivo `.env` o configura variables de entorno en Windows. Ejemplo con PowerShell:

```powershell
$env:FLASK_ENV = "development"
$env:API_KEY = "tu_api_key"
```

4) Iniciar la aplicación

```powershell
python app.py
# o si arrancas el backend por separado
python backend.py
```

5) Pruebas rápidas

- Asegúrate de que la aplicación responde en el puerto esperado (por ejemplo, `http://localhost:5000`).
- Revisa logs y corrige dependencias faltantes.

Despliegue (opciones comunes)

- Heroku: usar `Procfile`, configurar variables de entorno y `git push heroku main`.
- VPS/Servidor: usar `systemd` o `pm2` (si es Node) y configurar un proxy inverso con Nginx.
- Contenedores: crear un `Dockerfile` y usar `docker build` / `docker run`.

Mantenimiento

- Actualizar dependencias con `pip install --upgrade <paquete>` y luego fijarlas con:

```powershell
pip freeze > requirements.txt
```

- Añadir instrucciones específicas del módulo o modelos de IA si los hay.

Contacto

- Añade información de contacto o responsables del proyecto aquí.

Notas finales

- Completa `requirements.txt` con las versiones exactas cuando el proyecto esté estable.
- Documenta rutas, endpoints y ejemplos de uso según se vayan incorporando.
