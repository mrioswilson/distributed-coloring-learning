# Continuación de las fases de entrenamiento

La Fase A terminó sus 80 réplicas y el bloque inicial B terminó sus 20 controles
Own-only de Δ=3. Todos completaron 120 épocas y conservan los checkpoints previstos.
Los checkpoints seleccionados obtuvieron 16/16 éxitos y cero commits inseguros
en validación (N=40/160). Estos datos no son la prueba final independiente.

En los 20 pares Δ=3 se verificó igualdad exacta, a época 120, de datos,
inicialización, actualizaciones, exposiciones y pesos de PARTICIPATE/CANDIDATE/
message1. En todos difieren pesos de message2/COMMIT. El registro verificable está
en `data/m6study-release/reports/phase-b-initial-pairing-verification.json`.

## Cola encadenada

`experiments/continue_m6study.py` ejecuta los planes existentes en este orden:

| Etapa | Plan | Entrenamientos nuevos |
|---|---|---|
| Extensión B | `phase-b-all-degrees.json` | 60 Own-only, Δ=5/8/10 |
| Referencias emparejadas | `references.json` | 10, cinco semillas por método |
| Referencia de presupuesto original | `references-original-budget.json` | 5 Traditional con un tape |
| C, épocas fijas | `phase-c-epochs.json` | 40; cinco configuraciones ya están en A |
| C, actualizaciones fijas | `phase-c-updates.json` | 45 |
| D, entrenamiento MIXED | `phase-d.json` | 80 |

Son 240 entrenamientos nuevos; junto con los 100 terminados, 340 configuraciones
únicas. Se saltan los controles Δ=3 y las referencias de presupuesto que ya estén
completos. C conserva cinco semillas y Δ=3, con los ejes y combinaciones de 2.400
trayectorias ya definidos; no se amplía el producto de presupuestos a todos los
grados y veinte semillas. Las ampliaciones posteriores se decidirán mediante
validación cuando estén disponibles estos resultados.

Cada etapa usa el bloqueo GPU existente. El coordinador espera a que termine un
proceso antes de iniciar el siguiente, conserva hashes de los planes y se detiene
si un proceso falla. No inicia la prueba final: termina en
`awaiting_validation_review`, para revisar presupuestos y completar la selección
antes de congelar el protocolo E.

## Lanzamiento y operación

Desde la raíz del repositorio:

```bash
.venv/bin/python experiments/continue_m6study.py --dry-run
tmux new-session -d -s m6study '.venv/bin/python -u experiments/continue_m6study.py --root data/m6study-release'
experiments/m6study status --root data/m6study-release
cat data/m6study-release/continuation-state.json
experiments/m6study stop --root data/m6study-release
```

La cola continúa al cerrar SSH. `stop` persiste la época en curso y detiene tanto
el entrenador como el avance a la etapa siguiente. También se reenvía SIGTERM/
SIGINT al proceso de entrenamiento para permitir su cierre ordenado.

Después de que termine la sesión detenida:

```bash
tmux new-session -d -s m6study '.venv/bin/python -u experiments/continue_m6study.py --root data/m6study-release --resume'
```

La reanudación recorre los mismos planes y salta ejecuciones completas. Los logs
por entrenamiento/intento siguen en `data/m6study-release/logs/`. La etapa actual
se registra atómicamente en `continuation-state.json`. Un bloqueo adicional por
raíz evita lanzar dos coordinadores sobre el mismo estudio.

## Análisis pendiente antes del test final

La igualdad de resultados en la validación pequeña no establece una ventaja de
Counterfactual. Deben examinarse errores locales, k, dispersión y costes antes de
seleccionar ampliaciones. La comparación por épocas entre REGULAR y MIXED debe
conservar también los contadores reales de actualizaciones y rondas: la longitud
de las trayectorias cambia con la distribución.

Para atribuir diferencias específicamente a COMMIT conviene preregistrar la
evaluación del checkpoint común de época 120 junto al mejor checkpoint de
validación (`freeze --checkpoint-epochs 120`). En el checkpoint común P/C/message1
están emparejados; la selección de mejores épocas puede cambiar el presupuesto
efectivo. Un checkpoint seleccionado dentro del warmup todavía no ha recibido
ejemplos contrafactuales. Estas vistas deben fijarse antes de abrir el test final.
