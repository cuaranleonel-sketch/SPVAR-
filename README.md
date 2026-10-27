# LuzEspacial – Spatial VAR Colombia

Aplicación Streamlit de un **Spatial VAR (SpVAR)** para departamentos de Colombia.

## Datos
- **ITAED** (DANE): actividad económica departamental trimestral
- **VIIRS** (NOAA): luces nocturnas

## Contenido
- Mapa satelital de luces nocturnas
- SpIRF (funciones de impulso-respuesta espaciales)
- Mapas de impacto económico del shock
- Mapas de crecimiento y radiancia de luces

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
