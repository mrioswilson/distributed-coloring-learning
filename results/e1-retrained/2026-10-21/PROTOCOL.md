# Reproducción E1 y dos comparaciones con Luby

El usuario autorizó explícitamente entrenar cinco réplicas nuevas el 21 de
octubre de 2026. Son una reproducción con la configuración original, con pesos
y hashes nuevos. No se atribuyen a los archivos históricos no disponibles.
La procedencia original de los fuentes está documentada en
`results/e1-behavior/2026-10-21/upstream-source.json`, commit
`1d8857b25e8b654e87351fce33813692c70a6250` de Martín.

## Entrenamiento fijado antes del test

- UniversalColorGNN NORMALIZED, 16035 parámetros; método COUNTERFACTUAL.
- Semillas 0–4, exactamente las configuraciones E1 canónicas del repositorio.
- 800 grafos REGULAR N40/Δ3/K4 y tres tapes por grafo, por réplica.
- 200 grafos independientes para validación local y 16 para validar ejecuciones,
  por réplica. Las configuraciones conservan todos los valores originales.
- 120 épocas, Adam, lr 0,001, batch 128, warmup 10. Después del warmup se mezclan
  COMMIT propio y contrafactual con pesos 1/2, manteniendo las etiquetas exactas.
- Se utilizan los fuentes originales sin editar. Datos, inicialización,
  orden de minibatches y sorteos contrafactuales usan los streams originales.
- CPU por falta de CUDA; un thread de PyTorch por entrenador, dos procesos.
  Un intento de adelantar la semilla4 en un tercer proceso provocó espera de
  memoria/I/O del sistema y se detuvo antes de completar su primera época.
  Esa semilla se entrenará posteriormente en la cola con el presupuesto completo.
  Se mide memoria y se conserva el guard de disco; únicamente el umbral de
  RAM libre del guard genérico se adapta a CPU: 0,35 GiB frente al default 2 GiB.
  El benchmark de batch 128 usó 0,425 GiB de RSS máximo y aproximadamente0,11s/update.
- Los cinco pesos de época 120 se congelan juntos antes de evaluar el test.
  No se eligen modelos, épocas ni semillas a partir de sus resultados.

`data/e1-retrained-20261021/training-plan.json` fija configuraciones, hashes
de todos los fuentes y entorno. Se conserva una copia íntegra de los fuentes.
Cada entrenamiento guarda configuración, manifiesto de datos y su hash,
historia, estado del optimizador/RNG y checkpoints. `frozen.json` autentica
los cinco pesos finales, su procedencia y presupuesto. La configuración
debe coincidir con E1; el índice histórico continúa teniendo otro significado.

## Dos comparaciones

Plan `configs/e1_behavior/retrained.json`, namespace 2026102103.
Se conserva el presupuesto prefijado para la E1 histórica: 80 grafos nuevos
REGULAR/IRREGULAR, tamaños 8/12/40/160/640 con 8/8/12/8/4 grafos por familia,
más PATH8/CYCLE8/K4. Se reconstruyeron y excluyeron los 5.000 fingerprints
originales de entrenamiento/validación de las cinco configuraciones. Además
se comprobó que no hay solapamiento con los 80 grafos de validación de ejecución.
No se excluyen isomorfismos: la comprobación se refiere a grafos etiquetados.

**Distribuciones finales:**256 tapes independientes en cada muestra A/B.
Luby A se compara con cada E1 B. Luby B es control nulo independiente de A
y referencia de acoplamiento con E1 B. Son 148.736 rollouts previstos.
Las coloraciones completas/parciales, fallos y timeouts se conservan; no se
condiciona el análisis a terminar correctamente. Se respeta la terminación
original E1: no parar en la primera colisión, sino al terminar o a 256 rondas.

Se usa MMD² insesgada con kernel característico de Hamming y estado terminal,
con 3.999 permutaciones balanceadas dentro de cada grafo. Hay 100 tests primarios
estratificados por familia/tamaño/semilla/representación, ajustados con Holm.
Se comparan tanto los nombres de colores como las particiones módulo una
permutación global de esos nombres. Los vértices mantienen sus etiquetas.

No rechazo no implica equivalencia. La TV empírica es descriptiva.
El desacuerdo final con la misma tape produce una cota superior de TV,
con intervalo binomial exacto y Bonferroni sobre 830 cotas, tolerancia 0,05.
Si la cota no es pequeña, no se deduce separación de distribuciones. Cualquier
certificación se limita al grafo y modelo fijos evaluados, no a toda una familia.

**Decisiones locales:**ambos algoritmos reciben el mismo estado completo y
los mismos sorteos. Se consideran por separado trayectorias Luby y trayectorias
propias E1. Se compara participación, candidato, acción WAIT/COMMIT(color)
y COMMIT con las mismas propuestas exactas de Luby. Diagnósticos de seguridad
no corrigen acciones. Se guardan ejemplos reproducibles y conteos por k colores
disponibles; nodos/rondas de una ejecución son observaciones correlacionadas.

## Ejecución y validación

Desde la raíz del repositorio:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python experiments/train_compare_e1_replicas.py --stage all --workers 2
.venv/bin/python results/e1-retrained/2026-10-21/build_report.py
```

El driver entrena, congela, compara y audita. Puede continuar desde el último
checkpoint completo sin alterar fuentes/configuración/entorno. No se debe
editar `src/dcl` mientras entrena, porque el entrenador exige esos hashes para
reanudar exactamente. Los logs por semilla están en el directorio de datos.

Las 88 pruebas dirigidas pasaron antes de ejecutar la campaña. La auditoría
posterior comprobará todas las coloraciones, referencias Luby escalares,
cinco ejecuciones E1 originales por grafo y todos los ejemplos locales guardados.
Los tests técnicos con pesos aleatorios/fixtures se mantienen en temporales;
no constituyen resultados de E1 entrenada. Revisión científica independiente
pendiente. Estado y resultados reales solo se afirman al completar los archivos
`execution.json`, `audit.json` y la interpretación final.
