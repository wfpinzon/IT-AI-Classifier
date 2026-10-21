## IT-AI Classifier
Clasificación inteligente de incidentes de Soporte TI
Prototipo académico de aprendizaje automático que analiza la descripción de un ticket de soporte TI y sugiere una categoría para apoyar la clasificación inicial de incidentes.
Estado del proyecto: prototipo académico. Se evaluó con datos sintéticos; antes de utilizarlo en un entorno real requiere validación con tickets reales anonimizados, un piloto controlado y seguimiento continuo.

## Objetivo
Explorar cómo el procesamiento de lenguaje natural y el aprendizaje automático pueden ayudar a estandarizar la clasificación inicial de incidentes, reducir tareas repetitivas y apoyar al personal de una mesa de ayuda.

El sistema recomienda una categoría. La decisión final debe permanecer en manos del técnico, quien puede aceptar o corregir la sugerencia.
## ¿Cómo funciona?
1. El usuario ingresa la descripción de un incidente.
2. El texto se transforma en características numéricas mediante TF-IDF.
3. Un modelo de Regresión Logística estima la categoría más probable.
4. El prototipo muestra la categoría sugerida.
5. El técnico revisa el resultado y lo confirma o corrige.
   
## Tecnologías
- Python
- pandas y NumPy para manipulación de datos
- scikit-learn para TF-IDF, entrenamiento y evaluación
- Jupyter Notebook para experimentación
- joblib para guardar y cargar el modelo
- Visual Studio Code como entorno de desarrollo

## Categorías del conjunto de datos
El conjunto de datos sintético contiene 1.000 tickets, 12 variables y ocho categorías:
- Aplicaciones
- Acceso
- Hardware
- Impresoras
- Redes
- Seguridad
- Software
- Telefonía
  
Los datos son sintéticos y se crearon con fines académicos. No representan necesariamente la distribución, el lenguaje ni las condiciones de una mesa de ayuda real.

## Resultados de evaluación
En la evaluación realizada con un conjunto de prueba de 200 tickets sintéticos, el modelo obtuvo los siguientes resultados:
Métrica	Resultado
Exactitud (accuracy)	98,00 %
Precisión macro	98,21 %
Exhaustividad macro (recall)	97,92 %
F1 macro	97,90 %


Se observaron cuatro errores principales en los que tickets etiquetados como Hardware fueron clasificados como Impresoras. Este resultado destaca la necesidad de revisar los casos ambiguos y mantener supervisión humana.

La muestra adicional de 30 tickets se preparó para revisión humana e incluye deliberadamente los cuatro errores observados. Por tanto, su porcentaje de aciertos no debe interpretarse como una estimación independiente y representativa del rendimiento general. La validación humana debe registrarse por separado y no se considera completada hasta que un técnico la realice.

## Instalación y ejecución
1. Requisitos previos
- Python 3.10 o una versión compatible con las dependencias del proyecto.
- Git (opcional, si clonas el repositorio desde la terminal).
- Visual Studio Code o Jupyter Notebook para ejecutar el notebook.
2. Clonar el repositorio
Reemplaza TU-USUARIO por tu nombre de usuario de GitHub:
git clone https://github.com/TU-USUARIO/IT-AI-Classifier.git
cd IT-AI-Classifier
3. Crear y activar un entorno virtual
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Si PowerShell bloquea la activación, puedes abrir una terminal compatible o ejecutar Python desde el entorno virtual sin activarlo.
4. Instalar dependencias
Si el repositorio contiene requirements.txt:
python -m pip install -r requirements.txt

5. Ejecutar el prototipo
    pip install streamlit pandas scikit-learn matplotlib joblib
    python -m streamlit run app_it_ai_classifier.py
   
Nota: estos comandos presuponen que los archivos y rutas existen en el repositorio. Ajusta los comandos si la aplicación utiliza otro nombre o ubicación.

## Consideraciones éticas y limitaciones

- Datos sintéticos: las métricas obtenidas no garantizan el mismo rendimiento con tickets reales.
- Supervisión humana: el técnico debe confirmar o corregir la categoría propuesta.
- Privacidad: antes de un piloto, anonimiza los tickets y evita publicar información personal, credenciales, direcciones IP sensibles u otros datos confidenciales.
- Errores de clasificación: revisa especialmente las categorías con descripciones parecidas, como Hardware e Impresoras.
- Mantenimiento: monitorea el desempeño y vuelve a evaluar o entrenar el modelo si cambian los tipos de incidentes.
- Impacto económico: los cálculos de ahorro y retorno de inversión son escenarios estimados, no ahorros comprobados en producción.
- SLA: el conjunto de datos registra todos los tickets como cumplimiento de SLA, por lo que no permite medir ni demostrar una mejora real en incumplimientos.
- 
## Próximas mejoras
- Validar el modelo con un conjunto representativo de tickets reales anonimizados.
- Completar la revisión humana de la muestra de 30 tickets.
- Mejorar la distinción entre Hardware e Impresoras.
- Incorporar registro de correcciones y métricas de seguimiento.
- Evaluar confianza de las predicciones y derivar casos dudosos a revisión manual.
- Medir tiempos reales antes y después de un piloto controlado.
- Automatizar pruebas y documentar el proceso de entrenamiento.
  
Autor
Wilson Fernando Pinzón Guacaneme

perfil linkedin.co:  https://www.linkedin.com/in/wilson-fernando-pinzon-guacaneme/

Proyecto académico de Ingeniería de Sistemas

IT-AI Classifier — Inteligencia artificial aplicada al Soporte TI
