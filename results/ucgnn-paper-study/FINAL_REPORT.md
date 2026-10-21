# Resultados finales de la batería UCGNN para el manuscrito

La cola finalizó sus **157 tareas** el 21 de octubre de 2026, con **6,172 horas
GPU acumuladas de un máximo de 36**. No queda entrenamiento activo de este
estudio. Se preservan el protocolo `023deaf`, los checkpoints históricos y
las cohortes anteriores. No se realizó push remoto.

Se completaron 30 redes principales (seis tratamientos por cinco semillas),
un piloto own-only y un piloto de profesor cíclico. Las 85 vistas de modelos
incluyen checkpoints de época 120, selección por validación y especialistas;
no son 85 redes nuevas independientes. Hay **18.363 evaluaciones de grafos**
y **3.520 ejecuciones de estados parciales**. Ninguna celda planificada quedó
sin ejecutar; no hubo OOM ni timeout de evaluación. Los límites de 256 rondas
se cuentan como fallos algorítmicos, no como problemas técnicos.

## Qué aprendieron y qué generalizó

Resultados descriptivos de época 120, todas las familias y tamaños. Cada red
principal usa los mismos 304 casos y 288 identidades de grafo:

| Tratamiento | Redes | Coloraciones correctas | Fallo principal |
|---|---:|---:|---|
| NORMALIZED | 5 | 1.520 / 1.520 | Ninguno observado |
| RAW causalmente aislado | 5 | 1.174 / 1.520 | 346 ejecuciones sin terminar |
| SUM | 5 | 324 / 1.520 | 1.196 ejecuciones sin terminar |
| MEAN | 5 | 681 / 1.520 | 839 coloraciones finales incorrectas |
| DeepSets sin prefijos | 5 | 1.520 / 1.520 | Ninguno observado |
| LocalPaletteTransformer | 5 | 1.520 / 1.520 | Ninguno observado |

Todos aprendieron bien el profesor en grado 3. NORMALIZED, DeepSets y
Transformer transfirieron experimentalmente a todos los grados previstos
hasta 200, paletas hasta 201 y tamaños hasta 640. No se afirma igualdad de
rendimiento poblacional: estos son resultados finitos sobre grafos compartidos.
Los intervalos por red/familia/grado/tamaño están en `cells.csv`; las tasas por
semilla y contrastes emparejados están en `overall-by-seed.csv` y
`paired-contrasts.csv`. No se usó un intervalo binomial sobre ejecuciones
correlacionadas de cinco redes. Con ocho grafos por estrato, incluso 8/8 tiene
un límite inferior Clopper--Pearson de aproximadamente 0,631.

## Qué mecanismos quedan apoyados o refutados

**Normalización:** los cinco pares pasan la auditoría completa de tensores
iniciales, ejemplos, órdenes, cintas contrafactuales y actualizaciones.
Cambiar únicamente los extremos perjudicó la transferencia bajo este
entrenamiento. RAW concentra su ley all-free K=201 en aproximadamente cuatro
colores y puede perder progreso sin confirmar conflictos. Esto no demuestra
imposibilidad de aprender otra parametrización sin normalización.

**Agregación:** las sustituciones SUM y MEAN perjudicaron la transferencia de
forma distinta. SUM rechaza propuestas seguras al multiplicar vecinos
ordinarios; MEAN diluye mensajes de conflicto. Las sondas de multiplicidad
reproducen ambos efectos en las cinco redes de cada tratamiento. Como se
cambiaron ambas comunicaciones, no se atribuye cada fallo a una única etapa.
MAX es útil en esta familia; su necesidad para cualquier algoritmo aprendido
sigue sin demostrarse.

**Prefijos:** su necesidad queda refutada en la batería. DeepSets y atención
local transfirieron sin recibir los prefijos originales. Conservan mensajes
compartidos nodo--color, MAX, disponibilidad aprendida y el canal COMMIT;
no son controles de la necesidad de esos otros ingredientes. La representación
nodo--color es común a los modelos universales: no se aisló causalmente.

**Contrafactuales:** el nuevo own-only y C1 histórico tienen cada uno 16
conjuntos de tipos con aceptación insegura certificada, aunque ambos pasan
304/304 grafos ordinarios y 320/320 sondas de progreso. Los rollouts exitosos
no expusieron esos errores. La comparación apoya el aprendizaje de una regla
COMMIT más completa fuera de las trayectorias ordinarias; no demuestra que
los contrafactuales sean necesarios para el éxito en cualquier distribución.

## Transformer y diversidad

Atención local aprendió con cuatro colores y completó toda la batería hasta
201. En all-free K=201 su soporte efectivo fue 57,3--112,9 colores, frente a
193,1--199,5 para NORMALIZED. Sus mayores masas individuales fueron
0,126--0,243, pero mantuvo una cola diversa. En la clique de grado 200 tardó
30 rondas de media, frente a 14,33 de NORMALIZED y 21,47 de DeepSets, sobre
cinco redes y tres cintas compartidas. Son medias descriptivas.

El coste de atención crece cuadráticamente en K. En el perfil N=400, K=201,
el Transformer requirió unos 3,280 GFLOPs densos aproximados, 345 MiB CUDA
asignados y 50 ms por ronda; consultar `resource-summary.csv` para exclusiones
y cifras exactas. El perfil N=640 también cupo. No se ha demostrado viabilidad
para paletas arbitrariamente grandes.

## Qué está demostrado y qué está medido

La demostración de reducción finita de MAX permite certificar la regla local
de COMMIT con 64 conjuntos de seis tipos legales. De 39 checkpoints:
27 cumplen el certificado de regla exacta en aritmética real y bajo el modelo
de error float32 explícito; dos own-only fallan; diez SUM/MEAN quedan fuera del
alcance del certificado. Las 2.496 estrellas reales coinciden en decisión con
la reducción, y los 1.856 logits float32 aplicables quedan encerrados por los
intervalos. Las perturbaciones muestreadas +/-1e-6 conservan los signos de los
27 certificados en float32 y float64. Los guardas de nodo fijado, espera y
ausencia de candidato pertenecen a la implementación, no al aprendizaje.

Una segunda proposición demuestra una limitación de MEAN en esta interfaz:
si un clasificador continuo sin entrada separada de grado acepta con margen
positivo un tipo ordinario, un conflicto único queda diluido al repetir ese
tipo y acaba aceptándose a algún grado finito. No demuestra que la política
alcance necesariamente ese estado. Para SUM se da una construcción explícita
de mensajes que sí representa el rechazo exacto, por lo que su fallo observado
es de aprendizaje bajo este entrenamiento, no una imposibilidad expresiva.

La fórmula de confirmación y sus cotas tienen demostraciones completas bajo
independencia privada, participación y rechazo exacto. Una extensión derivada
después de observar resultados permite un átomo grande si queda una masa
uniformemente positiva en una componente suficientemente difusa. Esto aclara
por qué el máximo de la distribución por sí solo no explica todos los
resultados. **No se han certificado las hipótesis uniformes de candidatos de
ninguna red; no se demuestra terminación universal.**

El experimento de cuantiles E1 conserva su predictor principal congelado y sus
negativos: error medio por estado 0,569, diez configuraciones K=997 por encima
de dos, peor media 4,761. La predicción analítica secundaria condicionada por
contexto alcanza 0,075 de media. Los 120 estados son configuraciones fijas,
algunas equivalentes, y las probabilidades son cuadraturas de 2.049 puntos.
Un ajuste posterior no sustituyó al predictor principal. Se corrigió la
agregación del informe para excluir las sondas no retenidas añadidas después;
la regresión correspondiente está incluida en los tests.

## Profesor alternativo y RL

El profesor cíclico conserva participación y seguridad y cambia la política
de candidato: su masa en un color disponible es el hueco cíclico anterior
dividido por K. Su definición, ley y seguridad se probaron antes del piloto.
La red nueva de semilla 41005 alcanzó fidelidad local 0,9903 y 304/304
coloraciones correctas. En sus rollouts, la fidelidad al profesor cíclico baja
de 0,9911 a grado 3 a 0,03975 a grado 200. Transfiere una política útil, no una
imitación exacta del profesor. Es un piloto de una sola semilla con trayectorias
distintas; sus columnas de fidelidad cíclica y uniforme permanecen separadas.

La nota `docs/UCGNN_TEACHER_FREE_RL.md` especifica RL desde cero, supervisión y
fine-tuning, entorno, acciones, recompensas, critic, PPO y métricas. No se
lanzó RL. Ese trabajo queda reservado para un protocolo posterior.

## Manuscrito, verificación y límites

`paper/ucgnn/main.tex` incorpora los resultados actuales, pruebas, tablas y
figuras reproducibles. `main.pdf` compila sin errores; la bibliografía de
17 referencias fue verificada en fuentes primarias. Es un borrador académico,
pendiente de revisión científica/editorial y de asignar autores y destino.
La suite completa pasó **259 tests** (`tests-final.txt`).

`completion-audit.json` contrasta cobertura e integridad contra protocolo,
manifiestos, checkpoints y cohortes. No sustituye revisión independiente ni
certifica universalmente el software. Los conjuntos aleatorios son pequeños;
los intervalos bootstrap empíricos pueden degenerar si todas las diferencias
observadas coinciden. No se publican p-valores ni una afirmación de superioridad
basada en seleccionar estratos.

Las trazas por ronda conservan tamaño del soporte, masa máxima y colisiones.
Los histogramas completos se retienen para las rejillas locales controladas,
no para cada ronda de cada grafo. Los NPZ de fallos y las semillas permanecen
en `data/ucgnn-paper-study` para reproducción y diagnóstico más fino.

Los siguientes experimentos útiles son separar las dos etapas de agregación,
ampliar cohortes independientes bajo un protocolo nuevo y certificar masa
difusa en todos los estados alcanzables. No hace falta rescatar los controles
negativos mediante una búsqueda retrospectiva de hiperparámetros.

## Consultar, verificar y reproducir

```bash
cd /workspaces/AlgoLearning/ucgnn-paper-study
bash experiments/ucgnn-paper-study status
env PYTHONPATH=src OMP_NUM_THREADS=1 /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.audit
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m dcl.paperstudy.finalize --once
env PYTHONPATH=src OMP_NUM_THREADS=1 /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -m pytest -q
```

La cola ya está completa: no queda una sesión tmux viva que consultar.
`bash experiments/ucgnn-paper-study resume` recupera tareas incompletas tras
una interrupción; con este manifiesto completo solo vuelve a cerrar el informe.
No hay recuperación automática tras reinicio del contenedor.

Para reproducir entrenamientos desde cero sin tocar este resultado, crear un
worktree nuevo desde el commit final, reutilizar el entorno Python existente
y ejecutar allí el coordinador directamente (el launcher abre la ruta del
estudio original). Los checkpoints históricos se leen de sus rutas originales:

```bash
git -C /workspaces/AlgoLearning/ucgnn-paper-study worktree add --detach /workspaces/AlgoLearning/ucgnn-paper-reproduction HEAD
cd /workspaces/AlgoLearning/ucgnn-paper-reproduction
mkdir -p data/ucgnn-paper-study
touch data/ucgnn-paper-study/EVALUATION_READY data/ucgnn-paper-study/MECHANISMS_READY
env PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python -u -m dcl.paperstudy.coordinator
```

Ejecutar de nuevo el mismo último comando reanuda ese worktree. Después se
regeneran análisis y manuscrito con `dcl.paperstudy.report` y un compilador
LaTeX; el Tectonic ya instalado está en la ruta absoluta
`/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic`. Tiempos y redondeos
pueden depender del backend; versiones, estados RNG y hashes están registrados.
