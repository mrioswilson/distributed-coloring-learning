# Cierre de intervención y entrega recuperable

Corte de estado: 2026-10-21T03:25:07.582668+00:00. La intervención del agente
se cierra por indicación del usuario para conservar créditos. La cola computacional
continúa en segundo plano y tiene un presupuesto acumulado máximo de 36 horas
activas de procesos GPU; no necesita al agente para avanzar.

Worktree: `/workspaces/AlgoLearning/ucgnn-paper-study`.
Rama local: `research/ucgnn-paper-study`. No se ha hecho push.
Protocolo congelado: `023deaf`, sin alteraciones. Addendum prospectivo: `8c69168`.
Estado del corte: **12/31 entrenamientos iniciales terminados**;
tarea activa: `train-raw_endpoints-s41002`; 0.980 horas GPU contabilizadas.
Estos números son una instantánea; `STATUS.md` y el manifiesto vivo prevalecen.

## Resultados cerrados y alcance

- Código modular de normalización aislada, MAX/SUM/MEAN, DeepSets y
  LocalPaletteTransformer; UCGNN original preservada. Los seis pilotos de
  semilla 41000 completaron 120 épocas y fueron promovidos por la regla fijada.
  Las cinco réplicas por tratamiento están programadas; no se presenta la
  batería como completada.
- Auditoría causal de semilla 41000: inicializaciones tensor por tensor iguales,
  datos, órdenes de minibatches y cintas contrafactuales iguales durante las
  120 épocas, y 17.880 actualizaciones en ambos controles.
- El Transformer aprendió el profesor a grado 3 (fidelidad de candidato de
  validación 0,9957) y cabe técnicamente en N=640, grado 200, K=201. **Su
  generalización de grado sigue pendiente de evaluación.**
- Nuevo diagnóstico CPU float32 de E1: 120 configuraciones de disponibilidad,
  K=17,63,251,997 y 2.049 puntos medios por estado. Cero candidatos prohibidos
  observados en esa malla. El predictor empírico congelado tiene error medio
  de rango 0,569, pero 10 configuraciones K=997 superan dos y la peor media
  es 4,761. El predictor analítico secundario condicionado por contexto tiene
  media 0,075; no sustituye retrospectivamente al predictor principal.
- Pruebas completas de una reducción finita de seguridad local para MAX y
  una cota de progreso **condicionada** a independencia, participación y
  anticoncentración. Las hipótesis de progreso uniformes no se han demostrado
  para ninguna red aprendida. No se afirma terminación universal.
- Certificados por intervalos de los cinco E1 históricos: regla local exacta
  de COMMIT bajo las hipótesis de arquitectura y aritmética documentadas.
  C1 histórico y el piloto own-only tienen 16 máscaras de tipos con aceptación
  insegura certificada cada uno. La reducción no se aplica a SUM/MEAN.
- Manuscrito inicial en inglés, bibliografía verificada, figuras/tablas
  generadas desde CSV/JSON y PDF compilado. TODOs señalan resultados pendientes.
  Especificación teacher-free RL terminada; no se lanzó entrenamiento RL.
- Profesor cíclico implementado y probado, con piloto desde cero programado
  como fase secundaria. No se afirma aún que ese piloto haya aprendido.

## Lo que continúa y lo que falta

La cola realiza primero las réplicas primarias, después la cohorte confirmatoria
independiente, las comparaciones históricas/especialistas, el piloto cíclico y
los diagnósticos por modelo, en secuencia y dentro del presupuesto. La cohorte
prevé 304 casos por modelo universal (288 identidades de grafo) con cobertura
por familia, grado y tamaño. No se habían observado resultados de esa nueva
cohorte al cerrar esta instantánea. Por tanto todavía no hay una atribución
causal experimental de la extrapolación a normalización, MAX o prefijos.
Los timeouts, OOM, fallos de coloración y casos no ejecutados se conservan
separados. Si el presupuesto limita la cobertura, se informa como incompleta.

Un segundo proceso CPU, ventana tmux `finalizer`, espera la finalización,
el agotamiento del presupuesto o una parada/error recuperable; entonces
regenera el informe, las tablas/figuras y el PDF. También detecta ausencia
prolongada del coordinador. La compilación y sus hashes quedan en
`data/ucgnn-paper-study/finalization.json` y `logs/finalize-latex.txt`.
El finalizador ya se probó con `--once`, produciendo informe y PDF verificables.
No crea commits automáticamente: los archivos posteriores quedarán como
resultados nuevos en el worktree para revisión.

La sesión tmux sobrevive a una desconexión del cliente. **No se promete
recuperación automática tras reiniciar el contenedor o el host**; en ese caso
hay que ejecutar `resume`. Nunca se detuvieron Ollama ni servicios ajenos.

## Comandos exactos

```bash
cd /workspaces/AlgoLearning/ucgnn-paper-study
bash experiments/ucgnn-paper-study status
cat STATUS.md
tmux attach -t ucgnn-paper-study
tail -n 30 data/ucgnn-paper-study/logs/coordinator.txt
tail -n 30 data/ucgnn-paper-study/logs/finalizer.txt
bash experiments/ucgnn-paper-study resume
```

Informe/PDF de la instantánea actual, sin entrenar ni acceder al oracle en inferencia:

```bash
cd /workspaces/AlgoLearning/ucgnn-paper-study
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.finalize --once
```

Diagnóstico de selección recuperable, con la calibración ya congelada:

```bash
cd /workspaces/AlgoLearning/ucgnn-paper-study
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.cpu_prediction_study
```

Regresión de nuevos modelos y diagnósticos en CPU:

```bash
cd /workspaces/AlgoLearning/ucgnn-paper-study
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m pytest tests/test_paperstudy.py tests/test_paperstudy_evaluation.py tests/test_paperstudy_mechanisms.py -q
```

La suite completa pasó 257 tests antes de añadir el predictor analítico. El
registro posterior de cierre está en `tests-closeout.txt`. Consultar
`VERIFICATION.md` para los fallos de lock de una ejecución anterior durante
entrenamiento y la corrección del perfil de recursos con autograd.

## Archivos principales

- `docs/UCGNN_PAPER_PROTOCOL.md`: hipótesis, semillas, cobertura y análisis.
- `docs/UCGNN_MATHEMATICAL_ANALYSIS.md`: definiciones, demostraciones y límites.
- `docs/UCGNN_TEACHER_FREE_RL.md`: investigación posterior sin profesor.
- `results/ucgnn-paper-study/REPORT.md`: informe regenerable.
- `results/ucgnn-paper-study/candidate-diagnostics.csv`: selección por estado.
- `results/ucgnn-paper-study/certificate-summary.csv` y `certificates/`: seguridad.
- `results/ucgnn-paper-study/frozen-prediction/`: calibraciones y registro previo.
- `results/ucgnn-paper-study/closeout-*.json`: manifiestos de esta instantánea.
- `data/ucgnn-paper-study/`: checkpoints, datos, cintas, NPZ y logs persistentes.
- `paper/ucgnn/main.tex` y `main.pdf`: manuscrito y compilación actual.

La siguiente decisión científica debe basarse en los contrastes emparejados y
los negativos de esta cola; no en nuevas búsquedas para rescatar una hipótesis.
