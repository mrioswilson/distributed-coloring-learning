# Informe ejecutivo M6B

[Descargar el PDF para compartir](M6B_informe_ejecutivo.pdf).

Cinco páginas: resumen ejecutivo, diseño y evidencia, presupuestos y selección
de época, interpretación y límites, y referencias enlazadas. La primera página
se puede leer de forma independiente. El documento se basa en el estudio cerrado
el 17 de octubre de 2026 y fija sus referencias al commit `d734825`.

El [generador](../../experiments/build_m6b_executive_pdf.py) verifica los hashes
de los archivos publicados antes de usar sus datos. Para regenerarlo en un
entorno separado del entrenamiento:

```bash
python -m venv /tmp/m6b-pdf-build
/tmp/m6b-pdf-build/bin/python -m pip install reportlab==5.0.1
/tmp/m6b-pdf-build/bin/python experiments/build_m6b_executive_pdf.py
```

El generador usa las fuentes DejaVu instaladas en
`/usr/share/fonts/truetype/dejavu` y las incorpora al PDF. El documento compartido
incluye las fuentes necesarias para visualizarlo y no requiere ese entorno.
