# Segmentación de clientes de una marca de moda mexicana

Caso práctico para la vacante AI & MarTech Developer Intern de Keyrus.

| Archivo | Contenido |
|---|---|
| `analysis.ipynb` | El análisis completo: calidad de datos, las tres preguntas y los arquetipos. Se puede leer sin ejecutarlo |
| `slides/executive_summary.pdf` | Resumen ejecutivo |
| `src/customer_segmentation/` | La lógica que usa el notebook: `privacy`, `cleaning`, `segmentation`, `charts` y `config` (umbrales y supuestos) |
| `tests/` | Pruebas de privacidad y de las reglas de limpieza, con datos sintéticos |

## Datos personales
Los datos incluyen email, teléfono y fecha de nacimiento, así que se trataron como reales:
- Solo `privacy.py` lee `customers.csv`. Email y teléfono se reemplazan por tokens HMAC-SHA256 con una clave secreta, la fecha de nacimiento por un rango de edad y las notas en texto libre se eliminan.
- El análisis trabaja únicamente con esa versión seudonimizada y el notebook solo muestra datos agregados (hay una prueba que lo verifica).
- Los datos y la clave no forman parte de esta entrega.

## Cómo ejecutarlo
Requiere [uv](https://docs.astral.sh/uv/).

```bash
# 1. Copiar los cinco CSV de dataset_caso.zip en data/raw/
# 2. Crear la clave de seudonimización
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # pegar el valor en .env
# 3. Instalar, seudonimizar una vez y probar
uv sync
uv run python -m customer_segmentation.privacy
uv run pytest
# 4. Abrir analysis.ipynb y ejecutar todas las celdas
```
