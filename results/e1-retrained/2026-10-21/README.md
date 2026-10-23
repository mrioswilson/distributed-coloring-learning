# Cinco réplicas nuevas E1 frente a Luby

Las cinco réplicas nuevas E1 completaron 120 épocas con el modelo y entrenador
originales de Martín. Cada una obtuvo **20.480/20.480 coloraciones correctas**
en los 80 grafos aleatorios evaluados. Ninguno de los **100 tests MMD** detectó
diferencias con Luby tras el ajuste de Holm. Esto no demuestra equivalencia
general: se observaron pequeñas diferencias en las decisiones locales.

Esta cohorte es `newly-trained-replicas`, semillas 0–4, UniversalColorGNN
NORMALIZED/COUNTERFACTUAL, 16.035 parámetros. Sus cinco checkpoints tienen
hashes propios; no son los checkpoints históricos de E1_INDEX.json, que no
estaban disponibles para esta comparación. Se congeló la época 120 de todas
las réplicas antes de abrir el test, sin selección por sus resultados.

## Leer los resultados

- [Interpretación con números por réplica](INTERPRETACION.md).
- [Resultados clave](key-results.json), [informe](REPORT.md) y
  [protocolo fijado antes del test](PROTOCOL.md).
- [Auditoría de entrega](delivery-audit.json) y [auditoría de reproducciones](audit.json).
- [Tests distribucionales](distribution-tests.csv),
  [resultados por grafo](by-graph.csv), [decisiones locales agregadas](local-summary.csv)
  y [decisiones locales por grafo](local-by-graph.csv).

El test contiene 80 grafos nuevos REGULAR/IRREGULAR de 8/12/40/160/640 nodos,
más PATH8/CYCLE8/K4 como controles separados, con 256 tapes por muestra A/B.
Son 148.736 ejecuciones incluidas las referencias Luby. Los tests comparan
muestras independientes, con 3.999 permutaciones y ajuste conjunto de Holm.
Los conteos locales por nodo/ronda son correlacionados y descriptivos.

## Entender los cuatro gráficos

- [Entrenamiento](training.png) ([PDF](training.pdf)): pérdida y acuerdo de
  acción en validación durante las 120 épocas; no es una prueba de equivalencia.
- [Ejecuciones](rollouts.png) ([PDF](rollouts.pdf)): a la izquierda, porcentaje
  de coloraciones correctas; a la derecha, porcentaje de coloraciones distintas
  de Luby bajo los mismos sorteos. Una coloración distinta puede ser correcta.
  Ese desacuerdo acoplado no es la distancia entre distribuciones.
- [Distribuciones](distribution-tests.png) ([PDF](distribution-tests.pdf)):
  estimaciones MMD² por familia/tamaño y réplica, con nombres de colores y
  módulo renombramiento global. Rojo o azul no significan por sí solos una
  diferencia significativa; ninguna celda rechaza tras Holm.
- [Decisiones locales](local-decisions.png) ([PDF](local-decisions.pdf)):
  porcentajes de desacuerdo con el mismo estado y sorteos, en trayectorias
  Luby y propias. COMMIT con propuestas comunes coincide al 100 % en lo observado;
  no se ejecutaron confirmaciones inseguras. Los observadores no filtran acciones.

La TV empírica es descriptiva. Las cotas simultáneas de TV obtenidas mediante
el acoplamiento solo certifican cercanía para ciertos pares de grafo/modelo
fijos; una cota grande o un test sin rechazo no resuelve la equivalencia general.

## Procedencia e integridad

Los 17 archivos originales importados se conservaron byte a byte desde
[`1d8857b`](https://github.com/mrioswilson/distributed-coloring-learning/commit/1d8857b25e8b654e87351fce33813692c70a6250),
en la rama `feat/universal-color-gnn`. Sus hashes están en
[upstream-source.json](../../e1-behavior/2026-10-21/upstream-source.json).
El documento histórico [E1.md](../../../E1.md) se conserva sin cambios;
sus enlaces al resto del estudio histórico deben consultarse en
[la rama original](https://github.com/mrioswilson/distributed-coloring-learning/blob/feat/universal-color-gnn/E1.md).

[metadata/](metadata/) contiene copias idénticas del plan de entrenamiento,
la congelación, su fecha, la cohorte prefijada y las cinco historias completas.
El [manifiesto de publicación](git-delivery-manifest.json) registra cada archivo
entregado y sus hashes originales y versionados, incluso cuando coinciden.
Los CSV se almacenan con LF según .gitattributes, conservando sus valores.
Los hashes del [manifiesto original](artifact-manifest.json) corresponden a
los bytes originales del archivo local; cualquier normalización de saltos
de línea queda identificada por separado en el manifiesto de publicación.

El archivo completo `data/e1-retrained-20261021/` conserva datos, fuentes,
checkpoints y coloraciones por vértice fuera de Git. Se necesita ese archivo
para repetir la auditoría completa de modelos/datos o reanudar con la
procedencia exacta de esta ejecución; un clon por sí solo no incluye los pesos.

El pipeline original terminó con exit 0. La auditoría guardada comprobó
148.736 coloraciones, 332 ejecuciones escalares Luby, 415 E1 y 1.612 ejemplos
locales. Las 88 pruebas dirigidas pasaron antes de la campaña. Esta publicación
no repite entrenamiento, evaluación, pruebas ni generación de informes.
**La revisión científica independiente sigue pendiente.**
