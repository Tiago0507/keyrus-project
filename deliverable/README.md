# Segmentación de clientes de una marca de moda mexicana

Caso práctico para la vacante AI & MarTech Developer Intern de Keyrus (Fase 1).

| Archivo | Contenido |
|---|---|
| `analysis.ipynb` | El análisis completo: calidad de datos, las tres preguntas y los arquetipos. Se entrega con sus resultados, así que se puede leer sin ejecutarlo |
| `slides/executive_summary.pdf` | Resumen ejecutivo |
| `src/customer_segmentation/` | La lógica que usa el notebook: `privacy`, `cleaning`, `segmentation`, `charts` y `config` (umbrales y supuestos) |
| `tests/` | Pruebas de privacidad y de las reglas de limpieza, con datos sintéticos |

## Resultados principales
| Pregunta | Criterio | Clientes | % de la venta (3 años) |
|---|---|---|---|
| ¿Quiénes se interesan por las nuevas colecciones? | Compraron en los primeros 30 días del lanzamiento en al menos la mitad de los lanzamientos que tuvieron disponibles | 539 | 52,7 % |
| ¿Quiénes compran de forma recurrente? | 3 o más días de compra, mediana de 60 días o menos entre compras y última compra hace 90 días o menos | 844 | 67,6 % |
| ¿Quiénes son los clientes de mayor valor? | 25.000 MXN o más de venta neta en los últimos 12 meses | 188 | 39,5 % |

Los clientes se agrupan en seis arquetipos: fieles a la moda, reponedores de básicos, cazadores de rebajas, compradores únicos de tienda, ocasionales online dormidos y registrados sin compra. Los dos primeros son el 35 % de los clientes y generan el 75 % de la venta.

Sin la limpieza, el mejor cliente habría sido un comprador por volumen, el más fiel una cuenta interna de la marca y la venta cargada habría sido un 18 % mayor que la venta neta real.

## Datos personales
Los datos incluyen email, teléfono y fecha de nacimiento, así que se trataron como reales:
- Solo `privacy.py` lee `customers.csv`. Email y teléfono se reemplazan por tokens HMAC-SHA256 con una clave secreta, la fecha de nacimiento por un rango de edad y las notas en texto libre se eliminan.
- El análisis trabaja únicamente con esa versión seudonimizada y el notebook solo muestra datos agregados (hay una prueba que lo verifica).
- Los datos y la clave no forman parte de esta entrega.

## Cómo ejecutarlo
Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11 o superior. Los comandos se ejecutan dentro de esta carpeta.

```bash
# 1. Copiar los CSV de dataset_caso.zip en data/raw/
# 2. Instalar dependencias y crear la clave de seudonimización
uv sync
cp .env.example .env
uv run python -c "import secrets; print(secrets.token_hex(32))"   # pegar el valor en .env
# 3. Seudonimizar una vez y correr las pruebas
uv run python -m customer_segmentation.privacy
uv run pytest
# 4. Ejecutar el notebook y guardar sus resultados
uv run jupyter-execute --inplace analysis.ipynb
```

También se puede abrir `analysis.ipynb` en VS Code o Jupyter con el entorno `.venv` que crea `uv sync`.
