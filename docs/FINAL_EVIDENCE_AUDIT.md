# Auditoría final de evidencia — 26 de octubre de 2026

El cierre utiliza la rama local `research/scientific-closure-20261026`, creada
desde `71153993e50ec5833cee68e19636e96d802e087a`. No se modifican los pesos ni
las filas científicas históricas. No se publica ni se hace push. El manuscrito
autoritativo es `paper/ucgnn/main.tex`; la versión integrada anterior se
conserva en `paper/ucgnn/archive/pre-closure-71153993.{tex,pdf}`.

## Fuentes efectivamente disponibles

Los nombres de estudios son ramas y worktrees de una historia compartida,
no cohortes independientes por aparecer en directorios distintos.

| Estudio | Worktree conservado | Commit de referencia | Fuente principal |
|---|---|---|---|
| Pedro | `../pedro-source-audit` | `6836d855` | auditoría de fuente y reconstrucción histórica |
| Arquitecturas UCGNN | `../ucgnn-paper-study` | `b5427faf` | `results/ucgnn-paper-study/` |
| Integración siete profesores | `../ucgnn-scientific-integration` | `a54a18e3` | `results/integration/`, informe de correctitud |
| Complejidad muestral | `../ucgnn-sample-complexity` | `61f39760` | `results/sample-complexity/`, `data/sample-complexity-v1/` |
| Contrafactual | `../ucgnn-counterfactual-multiteacher` | `291d4b82` | `results/multiteacher-counterfactual/` |
| Generalización reciente | `../ucgnn-degree-generalization` | `71153993` | `results/multiteacher-degree-generalization/` |
| Paletas/distribuciones | worktrees `ucgnn-local-palette-study`, `ucgnn-global-distribution-study`, `ucgnn-robust-distribution-tests`, `ucgnn-size-degree-study` | inventario Git | resultados históricos con namespaces distintos |

Los grandes artefactos ignorados por Git se recuperaron localmente desde sus
worktrees, sin reconstruir pesos faltantes. Los hashes de 225 checkpoints
(195 originales más 30 controles CT) fueron comprobados. El código docente y
de inferencia permanece idéntico al conservado en las fuentes del estudio.
Los cuarenta pesos originales de Pedro que faltaban siguen faltando; no son
necesarios para reproducir las 21 certificaciones seleccionadas actuales.

## Reconciliación desde datos primarios

`scripts/final-evidence-audit.py` lee cada archivo crudo, verifica SHA-256,
reconstruye los grafos y sus fingerprints, calcula conflictos, colores fuera
de lista y vértices pendientes, y contrasta los indicadores guardados.
Verifica también las referencias exactas y el resumen antiguo por celda.

| Etapa | Ejecuciones estudiantes | Correctas | Incorrectas | Celdas | Celdas con fallo | Referencias exactas |
|---|---:|---:|---:|---:|---:|---:|
| Piloto | 3.570 | 3.526 | 44 | 2.070 | 44 | 930 |
| Confirmación | 810.000 | 808.413 | 1.587 | 810 | 15 | 150.000 |
| Exploración posterior de tamaño | 240 | 240 | 0 | 240 | 0 | 80 |

La confirmación contiene 1.559 colisiones y 28 cutoffs; el piloto dos
colisiones y 42 cutoffs de recursos con horizonte heterogéneo. Ni los pilotos
ni las referencias exactas se suman a 810.000.

| Cohorte confirmatoria | Correctas / ejecuciones | Fallos |
|---|---:|---:|
| Principal histórica | 299.972 / 300.000 | 28 |
| Principal contrafactual | 149.819 / 150.000 | 181 |
| Controles nuevos certificados | 239.521 / 240.000 | 479 |
| Controles CT a igual hito, sin certificación supuesta | 119.101 / 120.000 | 899 |

La cohorte principal completa aporta 449.791/450.000 y dos celdas fallidas;
los controles 358.622/360.000 y trece. En el control principal de grado tres
hay 90.000/90.000. Las observaciones agregadas son contabilidad, no la tasa de
un algoritmo promedio ni una distribución de políticas definida.

El archivo `results/final-closure/generalization-by-factor.csv` desglosa sin
mezclar etapas por profesor, arquitectura, checkpoint, semilla, tamaño,
grado, paleta, familia y cohorte. `generalization-cells.csv` conserva cada
checkpoint/condición con numerador, denominador, hash y ruta cruda. Sólo la
confirmación tiene intervalos CP, condicionales al diseño y la independencia;
los pilotos no reciben una interpretación inferencial por acumular filas.

## Independencia y alcance

810.000 ejecuciones reutilizan 4.000 grafos etiquetados, 20.000 semillas de
cinta y 25.000 pares caso–paleta. Se comparten grafos/cintas entre profesores,
arquitecturas, brazos y, en grado cuatro, paletas globales/locales. Un contador
de ejecuciones no es un contador de grafos ni de entrenamientos independientes.
Los cinco entrenamientos emparejados son las unidades para contrastar métodos.

La batería confirmatoria elige familias uniformemente por índice; los
denominadores realizados por familia difieren. El intervalo CP por checkpoint
estima el éxito bajo esa mezcla, no cada familia ni todo grafo de ese grado.
Su interpretación presupone independencia y neutralidad semántica del batch;
la neutralidad numérica para todos los kernels no está certificada.
1.000/1.000 da límite bilateral 95% 0,996317916; la cota histórica unilateral
0,997008750 es otra convención. Ninguna es simultánea sobre 810 celdas.

Confirmación: grado diez a tamaño 64, y tamaño 256 a grado cuatro; listas
locales sólo grado cuatro/tamaño 64. No se ha confirmado el producto cartesiano
de máximos. Grado 200 y tamaño 4.096 son exploratorios; este último sólo en
regulares y cuatro profesores paralelos. Seis profesores tienen extensión;
BE queda excluido. El cambio de grado también cambia K y, en CT, preferencias.

## Las 21 certificaciones

Doce PASS históricos de `results/sample-complexity/certification.json` y
nueve paquetes CF de `plans/successful-policies.json` dan 21/21. Cada uno
tiene cinco pesos, cinco semillas y 1.000/1.000 casos por peso. Son 105
checkpoints seleccionados, con protocolos distintos; no 21 entrenamientos.
La tabla de presupuesto real por combinación está en
`results/multiteacher-degree-generalization/native-21-consolidation.csv` y
en el apéndice del manuscrito. La tabla canónica de cierre precisa además
familias, batería y fuente exacta en `native-certifications.json`.

La verificación histórica completa con runtime confirma 735 entrenamientos,
12 PASS y nueve NOT CERTIFIED originales; no se reescribe este resultado.
La verificación CF confirma 195.000 ejecuciones finales, 194.979 correctas,
170 pesos distintos y cero inconsistencias de etiquetas. Esta contabilidad
incluye intentos fallidos y repeticiones sobre baterías nuevas; no se añade
al denominador de transferencia ni se interpreta como 195 modelos.

## Correcciones científicas incorporadas

1. Se distingue la cohorte principal de los controles en 810.000.
2. Se distingue primera familia fallida por orden de filas de otros testigos
   diagnósticos archivados. Los quince primeros fallos se reprodujeron de nuevo.
3. En GM CF el primer fallo es omisión de reparación tras rechazo correcto;
   los controles tienen escritura prohibida. Misma tasa no implica mecanismo.
4. BE CF tiene 6.776 etiquetas frente a 3.000; se retira atribución causal a
   igual presupuesto y se revela reutilización de tipos locales y development.
5. La revisión adversarial añade prevalencia, baselines triviales y confusiones
   de BE; el porcentaje bruto por sí solo es insuficiente.
6. Los certificados MAX históricos no se transfieren a los pesos actuales:
   los 45 núcleos canónicos seleccionados no pasan la condición suficiente.
7. El alfabeto CT nativo es bound 3 aun en grafos de menor grado; la extensión
   usa bound Δ. Las pruebas distinguen ambos parámetros.

## Comandos y límites verificables

Desde el repositorio con el entorno conservado:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/final-evidence-audit.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/final-failure-replays.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/final-counterfactual-audit.py
PYTHONPATH=src .venv/bin/python scripts/verify-sample-complexity.py --runtime
PYTHONPATH=src .venv/bin/python scripts/verify-multicf.py
PYTHONPATH=src .venv/bin/python scripts/final-native-audit.py
PYTHONPATH=src .venv/bin/python scripts/build-closure-assets.py
PYTHONPATH=src .venv/bin/python scripts/build-closure-paper.py
```

Los recibos de cierre, registro de afirmaciones, fuentes verificadas y
manifiesto final viven en `results/final-closure/`. Entorno: Python 3.11.2,
Torch 2.11.0+cu128 (inferencia CPU), NumPy 2.4.6, SciPy 1.17.1,
NetworkX 3.6.1; compilador Tectonic 0.17.0 con hash fijado. Las 480 regresiones
seleccionadas pasaron. El auditor terminal verifica indicadores históricos de
invariantes, pero no inventa trayectorias intermedias no guardadas. No se
afirma reproducción numérica bit a bit de entrenamientos en hardware arbitrario.

El registro `evidence-register.json` es la unión afirmación→prueba/protocolo→
datos→pesos→semillas→resultado→limitaciones→confianza. Las decisiones editoriales
y las respuestas a los revisores se conservan por separado.
