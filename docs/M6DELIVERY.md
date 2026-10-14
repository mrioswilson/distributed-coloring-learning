# Entrega y validación M6B

Fecha: 2026-10-14T12:26:11-03:00.

Rama: `feat/m6-systematic-study`. Commit del código validado: `cc359c3`.

93 tests pasan. Smoke CUDA de tres épocas para Δ=3,5,8,10. Reanudación exacta de Adam/pesos/RNG en CPU y CUDA; recuperación tras interrupción; inferencia sin shield; evaluación reanudable y gráficos. Los 21 artefactos históricos conservan sus hashes.

## Pilotos completos

| Δ | Parámetros | Épocas | Actualizaciones | Checkpoint seleccionado | Éxitos de validación | Commits inseguros |
|---|---|---|---|---|---|---|
| 3 | 8967 | 120 | 17760 | 100 | 16/16 | 0 |
| 10 | 11662 | 120 | 18240 | 70 | 16/16 | 0 |

La validación comprende REGULAR/IRREGULAR y N=40/160. Son dos semillas piloto; no demuestran reproducibilidad ni generalización a 40.960 nodos. El test final independiente sigue sin ejecutarse.

## Ejecución y ubicaciones

La Fase A contiene 80 experimentos. Los dos pilotos cuentan dentro de los 80; la cola salta ambos y continúa los restantes. B/C/D y las referencias están preparadas en planes separados.

- Sesión tmux: `m6study`.
- Raíz operativa: `data/m6study-release/`.
- Estado: `registry.sqlite`; logs: `logs/`.
- Checkpoints/configuración/datos/historial: `runs/<id>/`.
- CSV, PDF y PNG: `reports/`.
- Intento técnico fallido preservado: `data/m6study/` (véase REPORT.md).

## Comandos

Desde `/workspaces/AlgoLearning/distributed-coloring-learning`:

```bash
experiments/m6study status --root data/m6study-release
experiments/m6study stop --root data/m6study-release
tmux new-session -d -s m6study 'experiments/m6study run --plan configs/m6study/phase-a.json --root data/m6study-release --resume'
experiments/m6study aggregate --root data/m6study-release
```

La reanudación se lanza después de que la sesión detenida haya terminado. `stop` espera a la persistencia de la época en curso. Para cerrar SSH dejando la cola activa, salir de tmux con Ctrl-b d.

## Archivos creados o modificados

- `.gitignore`
- `REPORT.md`
- `configs/m6study/phase-a.json`
- `configs/m6study/phase-b-all-degrees.json`
- `configs/m6study/phase-b.json`
- `configs/m6study/phase-c-epochs.json`
- `configs/m6study/phase-c-updates.json`
- `configs/m6study/phase-d.json`
- `configs/m6study/references-original-budget.json`
- `configs/m6study/references.json`
- `docs/M6STUDY.md`
- `docs/m6-historical-artifacts.sha256`
- `experiments/m6study`
- `pyproject.toml`
- `requirements-m6study.lock`
- `src/dcl/counterfactual.py`
- `src/dcl/m4_data.py`
- `src/dcl/m4_models.py`
- `src/dcl/study/__init__.py`
- `src/dcl/study/cli.py`
- `src/dcl/study/config.py`
- `src/dcl/study/data.py`
- `src/dcl/study/evaluation.py`
- `src/dcl/study/graphs.py`
- `src/dcl/study/report.py`
- `src/dcl/study/storage.py`
- `src/dcl/study/train.py`
- `tests/test_study.py`
- `docs/M6DELIVERY.md` (esta entrega).
- `README.md` (enlaces al estudio).

Los datos, logs y checkpoints no están incluidos en los commits. No se hizo push remoto.
