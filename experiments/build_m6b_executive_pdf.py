"""Build the shareable M6B report from the frozen public results.

Requires ReportLab (built with 5.0.1). No training environment changes are needed.
Example: python experiments/build_m6b_executive_pdf.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results/m6study/2026-10-17-final"
SOURCE_COMMIT = "d734825051f2958b7a628b60ce4cd0c9ae3003f9"
BASE = "https://github.com/mrioswilson/distributed-coloring-learning/blob/" + SOURCE_COMMIT + "/"
PREFIX = "results/m6study/2026-10-17-final/"
WIDTH = A4[0] - 88
NAVY = colors.HexColor("#15364B")
TEAL = colors.HexColor("#007A82")
INK = colors.HexColor("#253C4B")
MUTED = colors.HexColor("#526574")
PALE = colors.HexColor("#EDF5F6")
LINE = colors.HexColor("#D9E3E8")
AMBER = colors.HexColor("#FFF4E5")
WHITE = colors.white

REFERENCES = {
    1: ("Informe científico final", "Interpretación completa, resultados negativos y límites.",
        [("ANALISIS.md", PREFIX + "ANALISIS.md")]),
    2: ("Resultados principales y variación entre semillas", "Éxito por grado, tamaño y presupuesto; medianas y bootstrap.",
        [("success-summary.csv", PREFIX + "success-summary.csv"),
         ("seed-summary.csv", PREFIX + "seed-summary.csv")]),
    3: ("Comparaciones emparejadas", "Métodos sobre los mismos grafos y tapes; comparación REGULAR / MIXED.",
        [("paired-by-seed.csv", PREFIX + "paired-by-seed.csv"),
         ("structural-paired.csv", PREFIX + "structural-paired.csv")]),
    4: ("Protocolo y decisiones anteriores al test", "Selección por validación, presupuestos y congelación de la prueba.",
        [("docs/M6VALIDATION.md", "docs/M6VALIDATION.md")]),
    5: ("Registro de coloraciones incorrectas", "Identifica réplica, época, grafo y resultado de cada fallo.",
        [("errors.csv", PREFIX + "errors.csv")]),
    6: ("Intervalos, rondas y tiempos por semilla", "Denominadores por familia/tamaño e intervalos de confianza.",
        [("scientific-success-by-seed.csv", PREFIX + "scientific-success-by-seed.csv")]),
    7: ("Diagnóstico de errores locales", "Estados, propuestas prohibidas y confirmaciones inseguras según k.",
        [("k-by-seed.csv", PREFIX + "k-by-seed.csv")]),
    8: ("Presupuesto y coste de entrenamiento", "Actualizaciones, datos expuestos y tiempo registrado por ejecución.",
        [("training.csv", PREFIX + "training.csv")]),
    9: ("Identidad y cierre del estudio", "Modelos incluidos, hashes, cohortes y cierre de los 64 grupos.",
        [("snapshot.json", PREFIX + "snapshot.json"),
         ("final-report-state.json", PREFIX + "final-report-state.json")]),
    10: ("Implementación del método", "Arquitectura, propuestas contrafactuales y etiquetas concordantes.",
         [("m4_models.py", "src/dcl/m4_models.py"),
          ("counterfactual.py", "src/dcl/counterfactual.py")]),
}


def register_fonts():
    root = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("DCL", "DejaVuSans.ttf"), ("DCL-Bold", "DejaVuSans-Bold.ttf"),
                           ("DCL-Mono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(root / filename)))
    pdfmetrics.registerFontFamily("DCL", normal="DCL", bold="DCL-Bold", italic="DCL", boldItalic="DCL-Bold")


def styles():
    common = dict(fontName="DCL", textColor=INK, spaceAfter=7, leading=14.2)
    return {
        "body": ParagraphStyle("body", fontSize=10.1, **common),
        "small": ParagraphStyle("small", fontSize=8.4, leading=11.6, fontName="DCL", textColor=MUTED, spaceAfter=6),
        "title": ParagraphStyle("title", fontName="DCL-Bold", fontSize=25, leading=30, textColor=NAVY, spaceAfter=12),
        "heading": ParagraphStyle("heading", fontName="DCL-Bold", fontSize=21, leading=26, textColor=NAVY, spaceAfter=12),
        "sub": ParagraphStyle("sub", fontName="DCL-Bold", fontSize=11.8, leading=16.3, textColor=TEAL, spaceBefore=8, spaceAfter=7),
        "lead": ParagraphStyle("lead", fontName="DCL", fontSize=11.2, leading=16.2, textColor=INK, spaceAfter=10),
        "kicker": ParagraphStyle("kicker", fontName="DCL-Bold", fontSize=8.5, leading=12, textColor=TEAL, spaceAfter=8),
        "table": ParagraphStyle("table", fontName="DCL", fontSize=9.2, leading=13, textColor=INK),
        "tablehead": ParagraphStyle("tablehead", fontName="DCL-Bold", fontSize=8.8, leading=12.5, textColor=WHITE),
        "metric": ParagraphStyle("metric", fontName="DCL-Bold", fontSize=18, leading=22, textColor=NAVY, alignment=TA_CENTER),
        "metriclabel": ParagraphStyle("metriclabel", fontName="DCL", fontSize=8.1, leading=11.5, textColor=MUTED, alignment=TA_CENTER),
        "ref": ParagraphStyle("ref", fontName="DCL", fontSize=9.2, leading=12.4, textColor=INK, spaceAfter=3),
    }


S = None


def p(text, style="body"):
    return Paragraph(text, S[style])


def cite(*numbers):
    return ' <font size="8" color="#007A82">' + " ".join(
        f'<link href="#ref{n}">[{n}]</link>' for n in numbers) + "</font>"


def table(rows, widths, *, font="table", paddings=7):
    cells = [[p(str(value), "tablehead" if i == 0 else font) for value in row]
             for i, row in enumerate(rows)]
    result = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("LINEBELOW", (0, -1), (-1, -1), .6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), paddings),
        ("RIGHTPADDING", (0, 0), (-1, -1), paddings),
        ("TOPPADDING", (0, 0), (-1, -1), paddings),
        ("BOTTOMPADDING", (0, 0), (-1, -1), paddings),
    ]))
    return result


def note(text, *, warm=False):
    result = Table([[p(text, "small")]], colWidths=[WIDTH])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER if warm else PALE),
        ("LINEBEFORE", (0, 0), (0, 0), 2.5, colors.HexColor("#BC7A26") if warm else TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def bullet(title, body):
    return p(f'<font color="#007A82">•</font> <b>{title}</b> {body}')


def number(value):
    return f"{int(value):,}".replace(",", ".")


def percentage(value):
    value = f"{100 * float(value):.2f}".rstrip("0").rstrip(".")
    return value.replace(".", ",") + "%"


class BudgetGrid(Flowable):
    def __init__(self, cells):
        super().__init__()
        self.cells = cells
        self.width, self.height = WIDTH, 163

    def draw(self):
        c = self.canv
        left, header, row_h = 40, 35, 29
        cell_w = (self.width - left) / 4
        labels = [("800 × 3", "épocas fijas"), ("200 × 3", "épocas fijas"),
                  ("2.400 × 1", "épocas fijas"), ("800 × 3", "30.720 actualiz.")]
        for j, pair in enumerate(labels):
            x = left + cell_w * (j + .5)
            c.setFillColor(NAVY); c.setFont("DCL-Bold", 9)
            c.drawCentredString(x, self.height - 12, pair[0])
            c.setFillColor(MUTED); c.setFont("DCL", 7.6)
            c.drawCentredString(x, self.height - 25, pair[1])
        for i, degree in enumerate((3, 5, 8, 10)):
            y = self.height - header - row_h * (i + 1)
            c.setFillColor(INK); c.setFont("DCL-Bold", 9.1)
            c.drawString(0, y + 10, f"Δ={degree}")
            for j, value in enumerate(self.cells[i]):
                v = float(value)
                t = v
                start, end = (.92, .97, .97), (.08, .24, .33)
                fill = colors.Color(*(a + (b - a) * t for a, b in zip(start, end)))
                c.setFillColor(fill)
                c.rect(left + cell_w * j, y, cell_w, row_h, fill=1, stroke=0)
                c.setFillColor(WHITE if v > .65 else NAVY); c.setFont("DCL-Bold", 10)
                c.drawCentredString(left + cell_w * (j + .5), y + 10, percentage(v))


def read_inputs():
    manifest = json.loads((DATA / "publication-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"The published source changed: {name}")
    metadata = json.loads((DATA / "snapshot.json").read_text())
    closure = json.loads((DATA / "final-report-state.json").read_text())
    assert metadata["status"] == closure["status"] == "complete"
    assert metadata["graph_evaluations_exported"] == 931700
    assert metadata["training_runs"] == 565 and metadata["complete_checkpoint_views_exported"] == 1210
    with (DATA / "seed-summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    def get(method, degree, graphs=800, tapes=3, updates=0, family="REGULAR"):
        selected = [r for r in rows if r["method"] == method and r["degree"] == str(degree)
                    and r["training_graphs"] == str(graphs) and r["training_tapes"] == str(tapes)
                    and r["updates_per_epoch"] == str(updates) and r["training_family"] == family
                    and r["checkpoint_policy"] == "epoch-120" and r["n"] == "40960"]
        assert len(selected) == 1
        return selected[0]

    main = {d: {method: get(method, d) for method in ("COUNTERFACTUAL", "OWN_ONLY")}
            for d in (3, 5, 8, 10)}
    for values in main.values():
        assert all(r["seeds"] == "20" and r["evaluations"] == "400" for r in values.values())
        assert values["COUNTERFACTUAL"]["successes"] == "400"
    principal = [r for r in rows if r["method"] == "COUNTERFACTUAL" and r["training_family"] == "REGULAR"
                 and r["training_graphs"] == "800" and r["training_tapes"] == "3"
                 and r["updates_per_epoch"] == "0" and r["checkpoint_policy"] == "epoch-120"]
    assert sum(int(r["successes"]) for r in principal) == sum(int(r["evaluations"]) for r in principal) == 61600
    assert sum(int(r["unsafe_commits"]) for r in principal) == 0
    grid = [[get("COUNTERFACTUAL", d, g, t, u)["mean_success_rate"]
             for g, t, u in ((800, 3, 0), (200, 3, 0), (2400, 1, 0), (800, 3, 256))]
            for d in (3, 5, 8, 10)]
    return main, grid


PAGE_TITLES = ["Resumen ejecutivo", "Diseño y evidencia", "Datos y selección de época",
               "Interpretación y límites", "Referencias y trazabilidad"]


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setTitle("M6B — Resumen ejecutivo e informe de resultados")
    canvas.setAuthor("Proyecto distributed-coloring-learning")
    canvas.setSubject("Estudio cerrado el 17 de octubre de 2026; resultados, límites y referencias verificables")
    canvas.setFillColor(TEAL); canvas.rect(0, A4[1] - 7, A4[0], 7, fill=1, stroke=0)
    canvas.setFont("DCL-Bold", 8.2); canvas.setFillColor(NAVY)
    canvas.drawString(44, A4[1] - 29, "M6B  /  APRENDIZAJE DE ALGORITMOS DISTRIBUIDOS")
    canvas.setStrokeColor(LINE); canvas.setLineWidth(.6)
    canvas.line(44, 39, A4[0] - 44, 39)
    canvas.setFont("DCL", 7.7); canvas.setFillColor(MUTED)
    canvas.drawString(44, 25, "Resultados cerrados: 17 oct 2026 · Fuentes: d734825")
    canvas.drawRightString(A4[0] - 44, 25, f"{doc.page} / 5")
    if doc.page <= len(PAGE_TITLES):
        key = f"page{doc.page}"
        canvas.bookmarkPage(key)
        canvas.addOutlineEntry(PAGE_TITLES[doc.page - 1], key, level=0)
    canvas.restoreState()


def build(destination):
    global S
    register_fonts(); S = styles()
    main, grid = read_inputs()
    story = []
    add = story.extend

    # 1. A complete executive brief that also works as a standalone first page.
    add([p("RESUMEN EJECUTIVO · ESTUDIO FINALIZADO", "kicker"),
         p("Una red pequeña,<br/>grafos mucho mayores", "title"),
         p("<b>Counterfactual</b> añade ejemplos modificados para enseñar a rechazar conflictos. "
           "Con este método, la configuración principal mantuvo la corrección al pasar de "
           "<b>40 a 40.960 nodos</b>: grafos 1.024 veces mayores." + cite(1, 2), "lead"),
         p("Problema: asignar colores de modo que dos nodos vecinos tengan colores distintos.", "small")])
    metrics = Table([[p("61.600/61.600", "metric"), p("80", "metric"), p("1.024×", "metric")],
                     [p("coloraciones correctas<br/>en el tratamiento principal", "metriclabel"),
                      p("réplicas principales<br/>20 por cada grado", "metriclabel"),
                      p("aumento del tamaño<br/>del grafo evaluado", "metriclabel")]],
                    colWidths=[WIDTH / 3] * 3)
    metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE),
                                ("TOPPADDING", (0, 0), (-1, 0), 10),
                                ("BOTTOMPADDING", (0, 1), (-1, 1), 10)]))
    add([metrics, Spacer(1, 7),
         p("La ventaja aumenta con el grado", "sub"),
         p("Prueba en 40.960 nodos, tras 120 ciclos de entrenamiento (época 120). "
           "Se comparan modelos emparejados." + cite(2, 3), "small")])
    rows = [["Vecinos máx.<br/>(Δ)", "Counterfactual<br/>con ejemplos modificados", "Own-only<br/>solo propuestas propias", "Éxito del<br/>control"]]
    for degree, values in main.items():
        cf, own = values["COUNTERFACTUAL"], values["OWN_ONLY"]
        rows.append([str(degree), f"<b>{cf['successes']}/400</b>", f"{own['successes']}/400",
                     percentage(own["mean_success_rate"])])
    add([table(rows, [65, 164, 167, WIDTH - 396], paddings=6), Spacer(1, 4),
         p("Cada fila reúne 20 modelos × 20 grafos compartidos: diez regulares y diez irregulares. "
           "Son 400 ejecuciones, no 400 grafos independientes.", "small"),
         bullet("Resultado principal.", "Cero errores globales y cero confirmaciones inseguras en las "
                "61.600 ejecuciones del tratamiento principal a época 120." + cite(1, 2)),
         bullet("La cantidad de datos importa.", "Con 200 grafos de entrenamiento y tres ejecuciones por grafo, "
                "el éxito en Δ=10 baja a 164/400 (41%)." + cite(2)),
         bullet("La selección de época importa.", "Una época elegida por validación puede fallar en el test, "
                "aunque la época 120 del mismo tratamiento funcione." + cite(4, 5)),
         note("<b>Alcance de la conclusión.</b> La evidencia respalda alta fiabilidad en las condiciones "
              "evaluadas. No demuestra corrección universal. El 100% corresponde al tratamiento principal "
              "y a una época fijada antes del test; otras configuraciones registran fallos.", warm=True),
         Spacer(1, 7),
         p("Estudio completo: 565 entrenamientos · 1.210 versiones de pesos · 931.700 evaluaciones." + cite(9), "small"),
         PageBreak()])

    # 2. Explain the intervention and the evidence without assuming repository knowledge.
    add([p("Cómo se obtuvo la evidencia", "heading"),
         p("Cada nodo aplica la misma red neuronal pequeña y se comunica solo con sus vecinos. "
           "Aprende tres decisiones: participar, proponer un color y confirmar o esperar. "
           "La ejecución avanza en rondas simultáneas; un color confirmado permanece fijo. "
           "Un algoritmo exacto actúa como profesor y genera los ejemplos de aprendizaje." + cite(10)),
         note("<b>Qué añade Counterfactual.</b> Enseña a evaluar las propuestas de la propia red y propuestas "
              "modificadas deliberadamente para producir conflictos. Las etiquetas indican qué decisión es "
              "correcta para la propuesta que realmente se está evaluando. Own-only omite los ejemplos modificados."),
         p("Diseño de la comparación principal", "sub")])
    add([table([
        ["Elemento", "Condición fijada"],
        ["Entrenamiento", "Desde cero; 800 grafos de 40 nodos, tres ejecuciones aleatorias por grafo, 120 épocas. "
         "Las primeras diez usan propuestas del profesor para entrenar la confirmación."],
        ["Grados y réplicas", "Δ=3, 5, 8 y 10, con Δ+1 colores y 20 semillas por grado. "
         "Se entrena un modelo distinto por grado y réplica."],
        ["Control emparejado", "Mismos datos, inicialización, orden y actualizaciones. A época 120, los módulos "
         "de participación, candidato y primera comunicación tienen pesos idénticos en los 80 pares."],
        ["Prueba independiente", "Seis tamaños: 40, 160, 640, 2.560, 10.240 y 40.960 nodos; familias REGULAR e IRREGULAR. "
         "Máximo 256 rondas; mismos grafos y números aleatorios para comparar modelos."],
    ], [105, WIDTH - 105], paddings=6),
         Spacer(1, 5),
         p("Durante la prueba la red ejecuta sus propias decisiones, sin correcciones del profesor. "
           "Las reglas exactas solo se usan para registrar diagnósticos." + cite(4, 10), "small"),
         p("La mejora está en la corrección; las rondas siguen próximas", "sub"),
         Image(str(DATA / "main-comparison.png"), width=WIDTH, height=WIDTH * .4),
         p("Figura 1. Modelos entrenados en 40 nodos y evaluados en 40.960. Las barras son desviaciones "
           "estándar entre 20 réplicas, no intervalos de confianza. Las rondas incluyen ejecuciones "
           "incorrectas; terminar rápido con una coloración inválida no mejora el resultado." + cite(2, 6), "small"),
         PageBreak()])

    # 3. Budget and checkpoint results are material qualifications, not footnotes.
    add([p("Datos y selección de época:<br/>dos decisiones que importan", "heading"),
         p("Los presupuestos siguientes se eligieron con validación antes de abrir el test. "
           "Cada celda resume 20 réplicas Counterfactual a época 120 y prueba en 40.960 nodos." + cite(2, 4)),
         BudgetGrid(grid),
         p("Figura 2. Grafos de entrenamiento × ejecuciones aleatorias por grafo (tapes). "
           "Las columnas de épocas fijas usan 120 épocas. La última fija 256 actualizaciones por época: "
           "30.720 en total.", "small"),
         note("<b>Hallazgo de presupuesto.</b> Reducir de 800 a 200 grafos con tres tapes conserva el 100% "
              "en Δ=3 y Δ=5, pero baja a 345/400 (86,25%) en Δ=8 y a 164/400 (41%) en Δ=10. "
              "Con épocas fijas también cambia el número de actualizaciones; el efecto no se atribuye solo "
              "a diversidad estructural."),
         p("Una buena validación no garantiza una buena época", "sub"),
         p("Se congelaron dos políticas: <b>best</b>, que toma la época elegida por validación, "
           "y <b>epoch-120</b>, la última época común. La validación de selección usa 16 grafos por candidato. "
           "En el tratamiento principal, best produjo estos fallos:" + cite(4, 5)),
         table([["Réplica", "Época elegida", "Fallos en el test"],
                ["Δ=3, semilla 17", "10, aún en la etapa inicial", "108/770, incluidos 20/20 en N=40.960"],
                ["Δ=8, semilla 18", "30", "1/770, en un grafo regular de 640 nodos"]],
               [112, 160, WIDTH - 272], paddings=6),
         Spacer(1, 5),
         p("Ambas políticas estaban fijadas antes de evaluar. Se conservan los fallos de best y se "
           "informa por separado el resultado de época 120; el test no se usa para reseleccionar modelos.", "small"),
         p("Grados variados y referencias históricas", "sub"),
         p("Entrenar con <b>MIXED</b> (mitad grafos regulares y mitad irregulares) obtiene también "
           "61.600/61.600 a época 120. Como REGULAR ya alcanza ese éxito, el test no distingue una mejora "
           "de MIXED en corrección. Su selección best sí registra fallos." + cite(1, 3)),
         p("En Δ=3 y con cinco semillas, ORACLE_MATCHED y TRADITIONAL con tres tapes obtienen 100/100 "
           "en el tamaño máximo a época 120. TRADITIONAL con un tape obtiene 54/100. "
           "Estas referencias muestran sensibilidad al presupuesto, con un alcance menor que la comparación principal." + cite(1, 2), "small"),
         PageBreak()])

    # 4. State the inferential limits alongside the observations that motivate follow-up work.
    add([p("Cómo interpretar los resultados", "heading"),
         note("<b>Unidad experimental.</b> En el tamaño máximo, cada modelo se prueba en 20 grafos: "
              "diez de cada familia. Esos mismos grafos se reutilizan en 20 réplicas. Las 400 ejecuciones "
              "por fila ayudan a comparar modelos, pero no amplían la cohorte a 400 grafos nuevos."),
         p("Incertidumbre y alcance", "sub"),
         bullet("Muestra de grafos limitada.", "Con 10 éxitos de 10 grafos de una familia, el límite "
                "inferior del intervalo exacto bilateral del 95% es aproximadamente 69,2%. "
                "Observar cero fallos no equivale a garantizar cero fallos futuros." + cite(6)),
         bullet("Réplicas y grafos compartidos.", "Los intervalos entre semillas condicionan en el conjunto "
                "de grafos observado. Las comparaciones emparejadas son exploratorias, sin corrección por "
                "multiplicidad. Las 931.700 evaluaciones no se tratan como ensayos independientes." + cite(2, 3)),
         bullet("Generalización acotada.", "Se verificó aumento de tamaño para modelos separados por grado, "
                "en las familias REGULAR e IRREGULAR del protocolo. La comunicación local se simula en "
                "un proceso; no se midió un despliegue físicamente distribuido." + cite(4)),
         p("Los errores locales no se explican solo por k=1", "sub"),
         p("<b>k</b> es la cantidad de colores aún disponibles. En Own-only, grado 3, las diez confirmaciones "
           "inseguras del tamaño máximo aparecen en k=1. En grado 10, 56.646 de 96.781 aparecen en k=11: "
           "también importan los conflictos entre propuestas simultáneas. Una decisión contraria a la regla "
           "del profesor puede dejar una coloración final válida si el vecino espera; se miden ambas cosas "
           "por separado." + cite(7)),
         p("Coste observado", "sub"),
         p("Los 565 entrenamientos suman <b>37,55 horas registradas</b>, incluida la generación inicial de "
           "datos. En la configuración principal, Counterfactual requiere aproximadamente un 19–51% más "
           "de tiempo que Own-only, según el grado, aunque iguala sus actualizaciones. Son costes de este "
           "entorno, no estimaciones de latencia en una red física." + cite(8)),
         p("Preguntas para una siguiente etapa", "sub"),
         bullet("Ampliar la prueba independiente.", "Añadir más grafos grandes para estimar mejor la fiabilidad."),
         bullet("Revisar la selección de época.", "Evaluar, con un protocolo nuevo, validaciones más amplias "
                "y reglas que distingan la etapa inicial de entrenamiento."),
         bullet("Separar mecanismos de error.", "Estudiar disponibilidad de colores y conflictos simultáneos "
                "en ablaciones nuevas; conservar intactos el protocolo y el test de este estudio."),
         PageBreak()])

    # 5. Stable, clickable references for checking every material claim.
    add([p("Referencias y trazabilidad", "heading"),
         p("Los números entre corchetes del informe llevan a esta guía. Cada enlace abre la versión "
           "publicada que respalda la afirmación; las fuentes quedan fijadas al commit del cierre.", "body")])
    for ident, (title, description, links) in REFERENCES.items():
        linked = " · ".join(f'<link href="{BASE + path}" color="#007A82"><u>{escape(label)}</u></link>'
                            for label, path in links)
        add([KeepTogether([
            p(f'<a name="ref{ident}"/><b>[{ident}] {title}</b>', "ref"),
            p(description, "small"), p(linked, "ref"), Spacer(1, 4),
        ])])
    add([Spacer(1, 6), note(
        '<b>Repositorio:</b> <link href="https://github.com/mrioswilson/distributed-coloring-learning" '
        'color="#007A82">github.com/mrioswilson/distributed-coloring-learning</link><br/>'
        f'<b>Versión de las fuentes:</b> <font name="DCL-Mono" size="7.4">{SOURCE_COMMIT}</font><br/>'
        '<b>Fecha de cierre:</b> 17 de octubre de 2026, hora de Chile. '
        'Pruebas terminadas a las 22:43; informes automáticos a las 22:55.'),
        Spacer(1, 6),
        p("Los datos originales por grafo y los checkpoints se conservan en el servidor. "
          "GitHub contiene código, tablas, figuras y hashes para revisar la evidencia y su procedencia.", "small")])

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=51, bottomMargin=49, pageCompression=1,
                            title="M6B — Resumen ejecutivo e informe de resultados",
                            author="Proyecto distributed-coloring-learning")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/informes/M6B_informe_ejecutivo.pdf")
    build(parser.parse_args().output.resolve())
