# Relevo para Martín sobre la revisión del artículo

Martín: la revisión del artículo está publicada en
[`codex/paper-revision-20261030`](https://github.com/mrioswilson/distributed-coloring-learning/tree/codex/paper-revision-20261030).
El commit del artículo auditado es
[`e8241184bcf81505838b74c7acd2834f22676d78`](https://github.com/mrioswilson/distributed-coloring-learning/commit/e8241184bcf81505838b74c7acd2834f22676d78),
sobre tu draft `c3affd16`. Este relevo se añade en un commit posterior de la
misma rama. El siguiente trabajo es tu revisión científica independiente y
la decisión de integración y destino editorial.

El [PDF revisado](../paper/ucgnn/main.pdf) tiene **27 páginas** y SHA256
`196e2fccbd3401abebc42a995cf132b5c8a4a9406633172d97eb6623ddec5984`.
El fuente canónico es [main.tex](../paper/ucgnn/main.tex). El PDF y fuente
anteriores están en `paper/ucgnn/archive/pre-revision-c3affd16.*`.

## Qué cambió

El argumento se centra en **fidelidad de continuaciones, progreso y reparación**.
Se explicitan los oráculos, estados e intervenciones admisibles, lo que
aprende la red y lo que suministra el intérprete. La cobertura nativa de siete
maestros y tres arquitecturas queda como evidencia finita bajo protocolos
heterogéneos, sin ranking general ni certificado universal.

Se añadió un ensayo separado de **40 modelos**: BE y GM, E1 con contexto
nativo de 19.396 parámetros, diez semillas emparejadas por maestro y dos
brazos. Cada modelo accede a 6.600 objetivos; los valores P/C de
condicionamiento se contabilizan aparte. Son 1.500 actualizaciones fijas,
con congelación de todos los pesos antes del test. Los diagnósticos posteriores
al test están marcados y no cambian protocolo, split, pesos o resultados.

| Maestro | BA nativo | BA CF | Diferencia media CF−nativo e intervalo bootstrap | Éxitos nativo / CF |
|---|---:|---:|---|---:|
| BE | 84,64% | 90,41% | +5,77 puntos [−13,73; +25,26] | 9.204 / 7.405 |
| GM | 50,00% | 100,00% | +50 puntos [50; 50] | 10.000 / 10.000 |

Los éxitos son sobre 10.000 ejecuciones por brazo/maestro. El bootstrap usa
diez semillas emparejadas; el intervalo degenerado GM refleja diez diferencias
observadas idénticas. Se conservan las semillas adversas.

## Interpretación científica que hay que mantener

- En BE, una regla programada de elegibilidad/participación obtiene **99,71%
  BA**, por encima de CF. CF sí mejora la exactitud por tipo: **86,94% frente
  a 81,27%** de la regla. La BA por ocurrencias no basta para identificar el
  rechazo de opciones ocupadas; los estratos y confusiones están publicados.
- En GM, el control nativo no recibe confirmaciones negativas de ASSIGN/REPAIR.
  CF aprende esos rechazos, sin ventaja autónoma observada y sin comparación
  contra otra adquisición que también suministre negativos informativos.
- El holdout de tipos finos es válido, pero sus configuraciones lógicas ya
  vistas cubren 99,475–100% de las ocurrencias BE y 100% GM. No afirmar
  extrapolación a condiciones lógicas nuevas.
- CF BE 105001/105008 ajusta sus auxiliares casi perfectamente, pero falla en
  los contextos retenidos: BA de entrenamiento 99,510%/100%, final
  52,756%/52,094%. BE 105010 es distinto: una omisión de participación con
  propuesta válida explica una primera violación localizada. No se atribuyen
  los 796 fallos de esa semilla a un único witness ni se afirma rescate completo.
- En los dos witnesses históricos Johansson/GM, cambiar una propuesta permite
  completar el rollout copiado; sustituir COMMIT exacto una vez conserva el
  fallo. Son intervenciones seleccionadas, no una solución poblacional.
- Los promedios BE de 14,53/13,43 pasos incluyen ejecuciones fallidas. Todos
  los éxitos toman **15 pasos** en ambos brazos; no es un speedup entre éxitos.

## Figuras y navegación

Quedan tres figuras: arquitectura, contraste histórico de continuaciones y
un esquema de los dos witnesses reales. Se retiraron la gráfica de pares
prospectiva y las barras redundantes; se guardaron sus archivos. La nueva
Figura 3 muestra los vértices 28/54 de Johansson y 11/40 de GM, con propuesta,
veto y cambio de una propuesta. La tabla completa de seis intervenciones
está en el apéndice. No se recomienda añadir otras gráficas de los porcentajes
que ya se consultan en tablas.

Para revisar en orden:

1. [Matriz de afirmación, evidencia y límite](PAPER_REVISION_CLAIMS.md).
2. [Tercera auditoría y decisiones sobre figuras](PAPER_REVISION_AUDIT3.md).
3. [Informe de cambios y comprobaciones](PAPER_REVISION_REPORT.md).
4. [Protocolo fijado antes del test](PAPER_REVISION_PROTOCOL.md).
5. [Reproducción y recuperación de evidencia](PAPER_REVISION_REPRODUCE.md).

## Reproducción

Usar un clon nuevo de la rama evita mezclar este trabajo con cambios locales.
El runtime nuevo está dentro del repositorio en
`results/paper-revision-20261030/paper-revision-runtime.tar.gz`:
26.328.083 bytes, SHA256
`9a33d4bb8bd8b621fd35fd877e8c42f8afaa43f311c5ca8c4abda316551cec8a`.
Contiene 300 archivos y extrae bajo `data/paper-revision-20261030`.

Entorno comprobado: Python 3.12, PyTorch 2.5.1, una CPU/hilo y
`pypdfium2==5.14.0`. El build actual usa pdfLaTeX/BibTeX nativos; las
instrucciones de dependencias están en el README del paper y en la receta.
El recibo original de fuentes enumera un subconjunto: no es un lock completo
de dependencias transitivas ni del entorno.

Desde la raíz del clon, con las dependencias instaladas:

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
sha256sum results/paper-revision-20261030/paper-revision-runtime.tar.gz
tar -xzf results/paper-revision-20261030/paper-revision-runtime.tar.gz
python scripts/build-revision-assets.py
python scripts/build-revised-paper.py
python scripts/verify-paper-revision.py
```

El builder de assets usa los análisis entregados y los witnesses originales
versionados; no entrena ni simula. Para recalcular controles desde bancos,
consultar la receta completa: `audit-revision-continuations.py` y
`probe-revision-be-failures.py`. Las etapas de entrenamiento no son necesarias
para revisar esta entrega. El builder histórico de cierre rechaza este fuente
para impedir sobrescribir el PDF revisado.

## Qué se comprobó y qué sigue pendiente

El recibo de 499 regresiones corresponde a la revisión inicial. La segunda
auditoría consultó doce modelos congelados sobre 102.128 ejemplos y reprodujo
un lote original de 32 casos. La tercera recomputó etiquetas, claves y controles
de 80.000 consultas emparejadas y los bootstrap desde cuarenta matrices; no es
otra campaña neuronal independiente. La verificación de 40 modelos, pesos,
etiquetas, partición y emparejamiento pasa. El PDF fue inspeccionado y tiene
cero referencias pendientes y desbordes. Los recibos y hashes están en
`results/paper-revision-20261030`, incluido su manifiesto.

Queda pendiente tu revisión científica independiente de los contratos,
supuestos de las pruebas y fuerza de las conclusiones. No se identificó un
nuevo error matemático central en las revisiones propias, pero estas no
sustituyen esa revisión.

También falta resolver la disponibilidad de los **810.000 registros crudos
históricos** del cierre: no se regeneraron aquí y sus totales permanecen
atribuidos a los fuentes originales. Se recuperaron las identidades de los
225 checkpoints; eso no equivale a recuperar todos los crudos. Los cuarenta
pesos históricos de Pedro se verificaron localmente, sin nuevo replay de su
cohorte.

Para un seguimiento más fuerte, priorizar un control GM con negativos
informativos a igual presupuesto y una evaluación con condiciones lógicas
realmente nuevas. Requiere un protocolo y test nuevos; conservar el ensayo
actual tal como quedó consumido. Para el artículo, quedan por acordar autoría,
orden, venue y formato de envío.
