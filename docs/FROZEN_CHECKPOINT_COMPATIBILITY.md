# Compatibilidad de los checkpoints congelados

Inventario exacto: `results/multiteacher-degree-generalization/checkpoint-inventory.json`.
Contiene 195 archivos autenticados por SHA-256, copiados a un directorio nuevo:
60 históricos principales, 45 contrafactuales principales, 45 matched y 45
pasivos nuevos. Cinco semillas por combinación/procedimiento. Los controles
BE sólo se reproducen en el dominio nativo. No se reentrenan ni editan pesos.
El inventario conserva ruta original, hash, protocolo, parámetros, etiquetas,
actualizaciones y configuración completa del entrenamiento, cuando fue almacenada.

## Qué objetos se comparan

K4 se refiere a los colores permanentes. BE también presenta al scorer
trece o siete puntos auxiliares durante Linial; eso no es entrenamiento
de coloración permanente con K13/K7 ni transferencia a otro grado. Sus
evaluaciones y opciones auxiliares conservan tamaños especializados.

Las doce combinaciones históricas proceden de `sample-complexity`, no de los
pesos canónicos de otro estudio Luby ni de los pesos no recuperados de Pedro.
Son Luby/Johansson/JP/Greedy × tres arquitecturas, semillas 83101–83105.
La cohorte nueva principal es ColorTrials/BE/GM contrafactual, 94101–94105,
BE/CT a 4000 updates y GM a 500. Matched y pasivo se conservan con el hito
que realmente fue certificado; no se les atribuye el presupuesto del CF.
`cohort-packages.json` vincula cada uno a su evidencia nativa 5×1000.

## Auditoría de formas e información

| Capa/interfaz | Dependencia original | Resultado al ampliar |
|---|---|---|
| Estado externo Rooted | 42 características; color one-hot de cinco posiciones; preferencias de cuatro | No acepta K>4 directamente; debe estructurarse por (nodo,color) |
| Vecindarios Rooted | Dos vecindarios completos de anchura tres | El núcleo usa MAX, sin parámetros por vecino; es posible evaluación dispersa equivalente |
| MLP mensaje 1 | Dos predicados binarios; 2→32→32 | Independiente de K y grado |
| MLP estado 1 | Pendiente + 32 mensajes | Independiente de K y grado |
| Participación canónica | Draw y pendiente | Independiente de K y grado |
| Contexto nativo | Fase, permiso y relaciones ID/slot, mismo número de canales | Independiente de K para JP/CT/GM; media sobre la paleta propia explícita |
| E1 NORMALIZED | Prefijos de alpha y suma normalizados; scorer compartido | K variable sin pesos nuevos; alpha no se sustituye por disponibilidad exacta |
| DeepSets | phi compartido, posición c/(K−1), 1/K y media de paleta | K variable; 1/K y las posiciones cambian fuera del soporte entrenado |
| Transformer | Atención de dos cabezas dentro de la paleta; sin embedding de longitud fija | K variable; coste de atención cuadrático en K; comunicación de grafo sigue siendo MAX |
| MLP mensaje 2/COMMIT | Cuatro predicados por color y contexto nativo | Mismo peso a todo vecino; no se añade un detector externo |
| IDs JP/GM | Signos derivados de IDs distintos en [0,65535] | Se conserva el mismo orden coherente, no se recalibran IDs con n o Δ |
| BE | F13/F7, trece evaluaciones y cuatro opciones relativas | No existe adaptador de esta campaña para la nueva aritmética/fases; contraste no aplicable |

Clasificación: **2, adaptador puramente estructural**, para los tres núcleos
y los seis profesores extensibles, en paletas globales y locales. El
envoltorio original no es directamente compatible; el estudio identifica
explícitamente esta diferencia. **3, incompatibilidad de la interfaz BE
especializada para una nueva variante de campos/paletas**, sin afirmar
imposibilidad para cualquier red con esos pesos o para una variante KW distinta.
El LocalGNN directo y M4B históricos no pertenecen a estas tres cohortes.

## Adaptador y límites de equivalencia

`degree/adapter.py` aplica los MLP congelados a predicados crudos de colores,
draws, preferencias, fases, permisos y relaciones. Reorganiza las dimensiones
de almacenamiento, elimina el padding y evalúa sólo los receptores necesarios.
La primera reducción MAX se comprime al soporte de categorías de mensajes
no negativos. **No computa el color disponible ni corrige decisiones**:
todos los mensajes siguen pasando por los MLP originales. La segunda etapa
consulta el color elegido; CT conserva todos los colores porque elige después
del intercambio de preferencias. Los oráculos sólo viven en evaluación/tests.

Con listas locales se enmascara únicamente la paleta del receptor. Un vecino
con paleta mayor puede anunciar un color fuera de la lista del receptor;
ese vecino sigue representándose como fijado, con igualdad falsa para ese
color del receptor. Para CT, la disponibilidad aprendida de un color fuera de
la lista del emisor vale cero por el alfabeto declarado, no por un detector
de conflictos. Las preferencias se normalizan por el mismo Δ global usado
en su muestreo, análogo al divisor tres nativo. Esta parametrización cambia
el dominio de entrada y no el contenido informacional de los mensajes.

En los 195 checkpoints, los controles nuevos nativos dan 1170/1170 correctos.
En los 150 extensibles, 21630 decisiones sobre estados docentes coinciden
exactamente con el envoltorio original; ninguna diferencia de colores finales
en los controles apareados. El mayor error absoluto observado de logits es
1.526×10⁻⁵. Son comparaciones numéricas finitas, no equivalencia bit a bit
universal de kernels o un certificado de margen para entradas arbitrarias.
El detalle y las salidas están en `validation/native-controls.json`.

## Qué puede y qué no puede deducirse del agregador

Alpha es una variable latente con sigmoid; estos entrenamientos supervisan
P, propuesta, COMMIT y las otras decisiones por fase, **no calibran alpha
directamente con etiquetas de disponibilidad**. Su diagnóstico alpha>.5
se conserva para describir la representación, no como un error de decisión
necesario ni una prueba autónoma de la causa de un fallo. Los scorers también
reciben h1 y pueden compensar otras parametrizaciones de esa variable.

MAX conserva soporte, no multiplicidad: duplicar un mensaje no cambia su
agregado. Es una proposición exacta sobre esta operación y un contraejemplo
a su capacidad de contar vecinos idénticos. No demuestra incapacidad para
estos profesores: disponibilidad, veto de ofertas, prioridades y detección
son condiciones existenciales, para las que OR/MAX puede ser suficiente.
CT necesita máximos de preferencias **junto con** disponibilidad; el contexto
aprendido debe mantener esa dependencia. La suficiencia de la interfaz no
prueba que cada ancho fijo o estos pesos implementen todas las reglas.

En E1 los errores de alpha pueden acumular masa en más colores; los extremos
normalizados de intervalos se aproximan al crecer K. DeepSets recibe nuevos
valores 1/K y atención cambia de longitud. Son mecanismos a contrastar con
trazas y sondas, no atribuciones causales automáticas. Los mensajes no reciben
el grado como una regla correctora. Los límites de 42/19 características del
serializador histórico no se presentan como un teorema de incapacidad de sus
núcleos compartidos.

## Sensibilidad de la codificación CT

La extensión principal conserva la fracción pref/Δ, igual a pref/3 nativamente. Una sonda posterior preserva en cambio las unidades pref/3 también al crecer Δ. Ambas codificaciones conservan el orden de preferencias sin aportar decisiones docentes. Sus respuestas difieren; ninguna corrige todas las consultas medidas. Se conserva el adaptador principal congelado y se etiqueta la alternativa como sensibilidad posterior de la observación, sin presentarla como una extrapolación funcional nueva ni seleccionar el mejor embedding por grado. La compatibilidad estructural no equivale a invariancia numérica de los pesos.
