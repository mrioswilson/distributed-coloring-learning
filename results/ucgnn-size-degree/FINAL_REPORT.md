# Tamaño, grado y generalización distribucional: estudio factorial

Se completó la matriz principal de **16 celdas y 48 grafos regulares nuevos**,
con tres grafos por celda y los quince checkpoints históricos. Se realizaron
500 intentos nuevos por generador para el contraste y 125 adicionales e
independientes para testigos. El estrato secundario conserva ocho grafos
irregulares; el histórico replica el mismo grafo n512-d100 con nuevas semillas.
No hubo entrenamiento nuevo ni push.

## Diseño e interpretación

Cada observación es una coloración completa etiquetada. Las listas son siempre
locales, C_v={0,...,d(v)}. El estimando principal condiciona a completar
correctamente antes de 128 rondas; los cuatro estados operacionales se
conservan por separado. No se mezclaron checkpoints ni se modificó COMMIT.

El omnibus V2 es idéntico al validado: selección/evaluación separadas,
permutaciones conjuntas, B=19.999, empates inclusivos y p=(1+excedencias)/20.000.
Su resolución 0,00005 permite Holm para las 720 hipótesis principales. Los
120 contrastes irregulares y 15 históricos tienen familias separadas, al igual
que el MMD antiguo, los eventos y los testigos: no hay una garantía conjunta
del 5% sobre su unión. Los atajos conservadores se refinan cuando pueden
afectar el conjunto de rechazos de Holm; un p=1 abreviado no se interpreta
como una estimación precisa de la cola.

Los dos eventos preregistrados son **A={U≤Δ}** y **T={aparece el color Δ}**.
A significa que falta algún color de la paleta, no necesariamente el máximo.
Se estiman sus probabilidades y diferencias D=P_M−P_L. El modelo binomial
saturado conserva una probabilidad por grafo/generador, condicionado al número
de completaciones válidas. Los promedios de arquitectura dan pesos iguales a
los cinco checkpoints y tres grafos fijos; el profesor compartido entra una vez.

Se preregistraron 1.800 contrastes de celda, tamaño, grado e interacción.
CP simultáneo recibe presupuesto de fallo 2,5%, y Hoeffding ponderado otro
2,5%; la intersección proporciona cobertura simultánea al menos 95%. Son
intervalos de la cohorte fija, no de una población de infinitos grafos o
entrenamientos. Un evento en un grafo aporta una cota TV inferior, nunca una
cota superior. Comparar cotas inferiores entre tamaños no ordena sus TV reales.

## Contrastes globales sobre los mismos bancos nuevos

| Arquitectura | V2 | MMD antiguo | Sólo V2 | Sólo MMD |
|---|---:|---:|---:|---:|
| E1 NORMALIZED | 115/240 | 49/240 | 66 | 0 |
| DeepSets | 170/240 | 121/240 | 49 | 0 |
| LocalPaletteTransformer | 180/240 | 174/240 | 6 | 0 |

Cada arquitectura tiene 240 comparaciones principales. Se conservan ganancias
y pérdidas frente al MMD antiguo. No se interpreta esta tabla como distancia
ni se compara directamente con los 265 casos por arquitectura históricos.

## P1. ¿Aumenta la diferencia E1–Luby con n a grado fijo?

No se establece un aumento general de la TV. Los efectos observables
dependen del grado y del evento.

Para A, los doce contrastes adyacentes de tamaño dieron **2 incrementos y
0 decrementos** con intervalo simultáneo que excluye cero. Los efectos
completos E1 se muestran abajo; las figuras incluyen además T, cada grafo y
las cinco réplicas. Son cambios en una probabilidad observable, no simplemente
en la potencia del test. La evidencia no establece una ley monótona de TV.

| n | Δ=3 | Δ=10 | Δ=50 | Δ=100 |
|---:|---:|---:|---:|---:|
| 128 | +0.000 [-0.024, +0.024] | +0.000 [-0.024, +0.024] | +0.095 [+0.026, +0.163] | +0.000 [-0.024, +0.024] |
| 256 | +0.000 [-0.024, +0.024] | +0.000 [-0.024, +0.024] | +0.121 [+0.052, +0.190] | +0.157 [+0.089, +0.226] |
| 512 | +0.000 [-0.024, +0.024] | +0.000 [-0.024, +0.024] | +0.037 [-0.006, +0.079] | +0.639 [+0.570, +0.703] |
| 1024 | +0.000 [-0.024, +0.024] | +0.000 [-0.024, +0.024] | +0.004 [-0.023, +0.031] | +0.603 [+0.550, +0.629] |

El evento T aporta información distinta. A grado 50, su diferencia pasó de
-0.239 [-0.307, -0.170] en n=128 a
-0.004 [-0.031, +0.023] en n=1024: el sesgo observable
se atenúa en ese régimen. A grado 100, el cambio de A entre n256 y n512 es
+0.481 [+0.384, +0.579].
Estos comportamientos dependen del observable y no ordenan las TV completas.
De las bandas simultáneas de las dos celdas se deduce conservadoramente una
reducción de al menos 0.138 en |D_T| entre n128 y n1024
a grado 50, para esta cohorte fija. Es una implicación de las bandas ya
registradas; no es una comparación de TV entre espacios de distinta dimensión.

## P2. ¿Aumenta con el grado a tamaño fijo?

Se observan aumentos en algunos riesgos, sin identificar una ley monótona de
TV ni separar causalmente grado y paleta.

Para A, los doce contrastes adyacentes de grado dieron **3 incrementos y
0 decrementos** respaldados simultáneamente. Todos los signos y no
detecciones aparecen en `factorial-contrasts.csv` y en las figuras 2 y 4.
En regulares, cambiar Δ cambia también K=Δ+1: sus efectos causales separados
no están identificados. La respuesta debe leerse por evento y régimen.
En n128-d100, A tiene diferencia +0.000 [-0.024, +0.024], mientras
T tiene diferencia -0.610 [-0.679, -0.541]. Que falte
algún color bajo ambas leyes no implica que falten los mismos colores.

## P3. ¿Existe interacción entre n y Δ?

Sí, para contrastes de probabilidades en esta cohorte. Por ejemplo, la
interacción E1 de A entre n256/n512 y Δ50/Δ100 es
+0.566 [+0.428, +0.704].

- E1 NORMALIZED, A: 2/10 interacciones; cuatro esquinas +0.603 [+0.479, +0.699].
- E1 NORMALIZED, T: 1/10 interacciones; cuatro esquinas +0.007 [-0.131, +0.145].
- DeepSets, A: 1/10 interacciones; cuatro esquinas +0.001 [-0.094, +0.097].
- DeepSets, T: 0/10 interacciones; cuatro esquinas +0.107 [-0.031, +0.245].
- LocalPaletteTransformer, A: 3/10 interacciones; cuatro esquinas +0.201 [+0.093, +0.299].
- LocalPaletteTransformer, T: 0/10 interacciones; cuatro esquinas +0.095 [-0.043, +0.233].

Son diferencias de diferencias de probabilidades, definidas antes de los datos.
Un intervalo que excluye cero respalda esa interacción en esta cohorte y
escala; no demuestra una interacción causal invariante a reparametrizaciones.

## P4. ¿Se reproduce en las cinco E1?

En los tres grafos nuevos n512-d100, **15/15**
comparaciones rechazaron. La menor cota TV simultánea de A entre las quince
fue **0.023**. Sus probabilidades E1 variaron
entre 0.186 y 1.000.
La figura 7 conserva las diferencias entre checkpoints; cinco entrenamientos
no constituyen una muestra grande de todas las redes que podrían entrenarse.
Al utilizar ambos eventos preregistrados A y T, la menor cota inferior de TV
entre las quince parejas grafo/checkpoint es 0.078. Elegir
el mayor límite de los dos eventos conserva la cobertura simultánea de su
familia. La intensidad varía entre checkpoints, aunque la diferencia no se
limita a s41000.

| Checkpoint E1 | Media P_E1(A), 3 grafos | Rango entre grafos | Menor cota TV de A |
|---|---:|---:|---:|
| normalized-s41000 | 1.000 | 1.000–1.000 | 0.882 |
| normalized-s41001 | 0.975 | 0.968–0.980 | 0.843 |
| normalized-s41002 | 0.995 | 0.992–0.998 | 0.864 |
| normalized-s41003 | 0.196 | 0.190–0.204 | 0.030 |
| normalized-s41004 | 0.205 | 0.186–0.218 | 0.023 |

## P5. ¿Se reproduce en grafos nuevos?

| Grafo | P_L(A) | Rango P_E1(A) | Menor cota TV de A entre cinco E1 | V2 |
|---|---:|---:|---:|---:|
| main-regular-n512-d100-r0 | 0.028 | 0.190–1.000 | 0.047 | 5/5 |
| main-regular-n512-d100-r1 | 0.036 | 0.204–1.000 | 0.047 | 5/5 |
| main-regular-n512-d100-r2 | 0.042 | 0.186–1.000 | 0.023 | 5/5 |

Tres realizaciones independientes respaldan replicabilidad dentro de este
generador, sin demostrar universalidad. Los rangos leave-one-graph y
leave-one-checkpoint son análisis de sensibilidad, no intervalos de población.

## P6. ¿Qué ocurre en DeepSets y Transformer?

Ambos se evaluaron en las mismas 16 celdas, tres grafos y cinco checkpoints.
La tabla global y las figuras 2–5 permiten comparar probabilidades, efectos y
bandas en la misma escala. Las diferencias entre arquitecturas no se resumen
como una clasificación por número de rechazos. Los irregulares aparecen en
un estrato separado, con sus histogramas de grados y paletas; una realización
por celda limita su generalización.

Tres cortes en probabilidades comparables, con intervalos simultáneos:

| Arquitectura | D_A(256,50) | D_A(512,100) | D_T(128,100) |
|---|---:|---:|---:|
| E1 NORMALIZED | +0.121 [+0.052, +0.190] | +0.639 [+0.570, +0.703] | -0.610 [-0.679, -0.541] |
| DeepSets | +0.016 [-0.045, +0.075] | +0.038 [-0.031, +0.107] | -0.108 [-0.177, -0.039] |
| LocalPaletteTransformer | +0.231 [+0.162, +0.299] | +0.294 [+0.225, +0.363] | -0.100 [-0.169, -0.031] |

Además, en cinco de las ocho celdas irregulares las paletas locales fuerzan
A≡1 bajo cualquier algoritmo: U≤min_t[t+#{v:d(v)≥t}]≤Δ. Por ejemplo, en
n1024-d100 la cota determinista es U≤44. Esas diferencias nulas de A no
constituyen evidencia de semejanza global. Se anotan expresamente y se
conservan el omnibus, T y los demás perfiles; no se usa A para extrapolar una
tendencia de regulares a esas celdas.

### Estrato irregular separado

| Arquitectura | V2 | MMD antiguo |
|---|---:|---:|
| E1 NORMALIZED | 7/40 | 0/40 |
| DeepSets | 26/40 | 21/40 |
| LocalPaletteTransformer | 37/40 | 33/40 |

Holm se aplica a sus 120 hipótesis, separado de las 720 regulares. El evento
T sigue siendo observable aun cuando A esté forzado. Las diferencias medias
sobre los cinco checkpoints del mismo grafo son:

| n | Δ | P_L(T) | D_T E1 | D_T DeepSets | D_T Transformer |
|---:|---:|---:|---:|---:|---:|
| 128 | 3 | 0.656 | +0.051 [-0.132,+0.231] | +0.031 [-0.152,+0.213] | +0.043 [-0.140,+0.224] |
| 128 | 10 | 0.140 | -0.022 [-0.153,+0.110] | +0.116 [-0.040,+0.266] | +0.135 [-0.023,+0.287] |
| 128 | 50 | 0.022 | -0.018 [-0.065,+0.027] | +0.194 [+0.078,+0.299] | +0.187 [+0.073,+0.292] |
| 128 | 100 | 0.016 | -0.016 [-0.056,+0.022] | +0.173 [+0.066,+0.271] | +0.171 [+0.063,+0.269] |
| 1024 | 3 | 0.998 | +0.000 [-0.029,+0.029] | -0.000 [-0.030,+0.029] | -0.001 [-0.032,+0.029] |
| 1024 | 10 | 0.734 | -0.050 [-0.226,+0.128] | +0.212 [+0.075,+0.337] | +0.222 [+0.090,+0.342] |
| 1024 | 50 | 0.218 | -0.164 [-0.277,-0.043] | +0.667 [+0.508,+0.790] | +0.690 [+0.538,+0.807] |
| 1024 | 100 | 0.156 | -0.155 [-0.236,-0.068] | +0.704 [+0.550,+0.822] | +0.722 [+0.569,+0.837] |

Los intervalos proceden de las cotas CP elementales ya simultáneas: el profesor
compartido se resta una vez. No son intervalos de una población de grafos o
entrenamientos. La figura 11 muestra T y marca dónde las listas fuerzan A.
El grado medio, varianza, histogramas y fracción cercana al máximo están en
los metadatos archivados. Cambian también las paletas elegibles por vértice;
estos ocho casos no identifican una tendencia universal del grado máximo.

## P7. ¿Qué propiedades finales distinguen las leyes?

Los testigos adaptativos se eligieron sólo en las primeras 500 muestras y se
validaron en las 125 reservadas. En la cohorte principal:

- E1 NORMALIZED: 32 testigos; used_colors (18), class_entropy (14).
- DeepSets: 85 testigos; class_entropy (44), mean_color (39), largest_class_fraction (2).
- LocalPaletteTransformer: 162 testigos; class_entropy (128), mean_color (29), largest_class_fraction (4), used_colors (1).

Ejemplos con las mayores cotas inferiores dentro de cada arquitectura:

- **E1 NORMALIZED**, `main-regular-n1024-d100-r0 / normalized-s41000`: evento «used_colors ≤ 100». Holdout Luby=0/125=0.000; red=125/125=1.000; diferencia +1.000, intervalo simultáneo [+0.830,+1.000], TV≥0.830.
- **DeepSets**, `main-regular-n1024-d100-r1 / deepsets-s41004`: evento «mean_color ≤ 48.975586». Holdout Luby=4/125=0.032; red=121/125=0.968; diferencia +0.936, intervalo simultáneo [+0.697,+0.998], TV≥0.697.
- **LocalPaletteTransformer**, `main-regular-n1024-d10-r0 / transformer-s41000`: evento «class_entropy ≤ 2.3934074». Holdout Luby=0/125=0.000; red=125/125=1.000; diferencia +1.000, intervalo simultáneo [+0.830,+1.000], TV≥0.830.

`used_colors` es U; `mean_color` es el promedio de las etiquetas numéricas;
`class_entropy` es −Σp_c log(p_c), sin normalizar. Las definiciones completas,
incluidas las tuplas categóricas si se seleccionan, están en
`representative-heldout-witnesses.json`. Primero se eligió un evento por
contraste usando las 500 muestras principales; después se estimó en el
holdout. Elegir estos ejemplos entre los 855 testigos no invalida sus cotas
CP simultáneas sobre esa familia. No prueban que una sola propiedad explique
todas las diferencias de una arquitectura ni identifican una causa.

Los eventos por color se midieron todos prospectivamente con una familia CP
simultánea separada. El mayor testigo de color seleccionado en el objetivo
fue `main-regular-n512-d100-r0 / normalized-s41000`, «aparece el color 100»:
Luby=1.000, modelo=0.000, diferencia
-1.000, intervalo [-1.000,-0.944],
TV≥0.944. La selección del máximo no invalida esa cobertura
simultánea; no se usa un intervalo ingenuo del evento elegido retrospectivamente.

También se archivan U, color máximo, U/K, mayor clase, entropía, color medio y
reutilización entre no vecinos. Wasserstein sobre proyecciones normalizadas
tiene bandas DKW puntuales descriptivas. Una media estructural próxima puede
coexistir con una diferencia enorme en un evento de cola; esas proyecciones
no identifican siempre la ley completa.

### Explicación exploratoria con selección controlada

El test global no garantiza que el único testigo elegido lo explique. En
E1 n1024-d50, V2 rechazó las quince comparaciones; el selector original eligió
tuplas categóricas para algunos checkpoints y su holdout no las validó. Por
ejemplo, en r0/s41003 memorizó 500 tuplas distintas: el evento cubrió 250/250
filas Luby y 0/250 de la red durante selección, 1/250 y 0/250 en evaluación,
y 0/125 y 1/125 en reserva. El soporte grande produjo sobreajuste; la reserva
independiente lo reveló correctamente. Esto limita al explicador, sin cambiar
la calibración ni los valores-p del omnibus.

Después de observar ese problema se añadió un análisis **exploratorio** sobre
las siete funciones escalares congeladas antes del muestreo. Usa los 625
intentos originales por banco y bandas DKW uniformes sobre todos los umbrales.
La unión de 57 grafos × 16 generadores × 7 funciones = 6.384 CDF da cobertura
simultánea al menos 95%; el profesor se cuenta una vez. Cada banda tiene radio
0,09980 cuando N=625. Restarlas permite elegir función, umbral y caso tras
observar los datos, sin usar un intervalo ingenuo del evento seleccionado.
La garantía corresponde a **otra familia**, separada de los contrastes y
testigos originales; no convierte este análisis en confirmación independiente
ni aumenta retrospectivamente los rechazos originales.

| Arquitectura | Cotas TV positivas / 240 | Entre rechazos globales V2 |
|---|---:|---:|
| E1 NORMALIZED | 119/240 | 109/115 |
| DeepSets | 167/240 | 167/170 |
| LocalPaletteTransformer | 180/240 | 180/180 |

En E1 n1024-d50 las quince cotas son positivas, con mínimo
0.039, aunque A y T apenas separan los
bancos en esa celda. En el mismo grafo r0, la figura 13 muestra:

- `normalized-s41000`, evento `entropy_normalized ≤ 0.996043424564`: Luby=20/625=0.0320, IC [0.0000,0.1318]; E1=602/625=0.9632, IC [0.8634,1.0000]. Diferencia +0.9312, IC simultáneo [+0.7316,+1.0000], TV≥0.7316.
- `normalized-s41003`, evento `mean_color_normalized ≤ 0.49783203125`: Luby=230/625=0.3680, IC [0.2682,0.4678]; E1=394/625=0.6304, IC [0.5306,0.7302]. Diferencia +0.2624, IC simultáneo [+0.0628,+0.4620], TV≥0.0628.
- `normalized-s41004`, evento `mean_color_normalized ≤ 0.4980078125`: Luby=237/625=0.3792, IC [0.2794,0.4790]; E1=409/625=0.6544, IC [0.5546,0.7542]. Diferencia +0.2752, IC simultáneo [+0.0756,+0.4748], TV≥0.0756.

Los umbrales impresos se redondean; los eventos exactos usan los valores de
precisión completa de `exploratory-cdf-witnesses.json`.

En estos grafos regulares, `entropy_normalized` es la entropía de clases
dividida por log(K), y `mean_color_normalized` el color medio dividido por Δ.
En irregulares, esta última función promedia c_v/max(1,d(v)). Las probabilidades,
intervalos y eventos de los 855 casos y sus siete funciones se archivan,
incluidas las cotas cero. No son una clasificación causal de propiedades ni
cotas superiores de TV: estas proyecciones no son inyectivas.


## P8. ¿El caso n512-d100 es excepcional?

La reproducción histórica recuperó exactamente 2/250 frente a 250/250 para A
en s41000, manteniendo su cota TV≥0,894. En la replicación independiente del
mismo grafo, las cinco E1 dieron probabilidades entre
0.210 y 1.000,
frente a 0.030 de Luby. Los tres grafos nuevos se
presentan arriba, y la matriz completa evita reducir la explicación a ese caso.
No se mezcló ninguna observación histórica con las nuevas.
El sesgo no aparece por primera vez en n512: a grado100, T ya tiene
diferencia -0.610 [-0.679, -0.541] en n128, aunque
A apenas separa esos bancos. La saturación de un evento puede ocultar una
diferencia que otro evento sí muestra; el caso destacado por A no identifica
por sí solo un umbral causal de tamaño.

Los quince contrastes del grafo histórico con datos nuevos, bajo su familia
Holm separada, dieron:

| Arquitectura | V2 | MMD antiguo |
|---|---:|---:|
| E1 NORMALIZED | 5/5 | 5/5 |
| DeepSets | 5/5 | 5/5 |
| LocalPaletteTransformer | 5/5 | 5/5 |

## P9. ¿Cómo se relacionan candidatos y leyes completas?

La cuadrícula fina de 4.096 uniformes, sobre estados iniciales comunes, dio:

| Checkpoint E1 | q_grid(top), K=51 | q_grid(top), K=101 |
|---|---:|---:|
| normalized-s41000 | 0.000000 | 0.000000 |
| normalized-s41001 | 0.000000 | 0.000000 |
| normalized-s41002 | 0.000000 | 0.000000 |
| normalized-s41003 | 0.009277 | 0.000732 |
| normalized-s41004 | 0.005859 | 0.000000 |

Las referencias uniformes son 1/51≈0,019608 y 1/101≈0,009901. Un cero de
cuadrícula no demuestra masa operacional exactamente cero. Se conservaron
cuadrículas de 1.024, sus discrepancias y estados de rondas 2,4,8, con el
mismo análisis para las tres arquitecturas. En el estado inicial no aislado,
la reducción MAX de tipos de mensaje hace que, a K fijo, la función de
candidatos no dependa de n ni del grafo regular. Las comprobaciones numéricas
de esa invariancia fueron True.

Un contraejemplo al uso aislado del estado inicial es `normalized-s41004`:
q_grid(top)=0.000000 con K=101, pero el color máximo aparece con
probabilidad media 0.813 en sus coloraciones finales de los tres
grafos nuevos n512-d100. La disponibilidad cambia durante las rondas; una
masa inicial de cuadrícula nula no prueba ausencia en toda trayectoria.

Una referencia de primera ronda, sin condicionar al éxito futuro, con
participación p, propuestas independientes
q_c y COMMIT exacto da P(v confirma c)=p q_c(1−p q_c)^Δ. Los vecinos deben
evitar participar y proponer el mismo color. El profesor ideal tiene p=1/2 y
q_c=1/(Δ+1). Es una relación no lineal entre propuestas y confirmaciones;
las confirmaciones de vértices vecinos no son observaciones independientes.
La fórmula no presupone que el COMMIT aprendido coincida en todos los estados
con la regla exacta, ni se utilizó para modificar ninguna acción.

Las trayectorias registran propuestas, participación, disponibilidad de
referencia, COMMIT y clases finales sin modificar decisiones. Se observaron
0 COMMIT inseguros en esos diagnósticos. La cadena
temporal es descriptiva: correlación y contabilidad de autorizaciones no
identifican por sí solas el efecto causal de intervenir en una propuesta.

### Intervención exploratoria separada

Se reemplazó únicamente el canal de candidatos por la propuesta exacta local
del profesor, sobre el estado de la política intervenida I. Se conservaron
sus pesos, participación y COMMIT originales, sin escudo. El plan se versionó
antes de obtener datos de esta intervención; su tamaño se eligió sólo por
coste: 1.000 parejas independientes por celda. Estado: `complete`,
60/60 celdas, tres grafos nuevos n512-d100 y el histórico,
con las quince redes.

| Arquitectura | Celdas | TV(I,L)≤0,05 respaldada | TV(original,I)>0 respaldada | Rango cotas superiores TV(I,L) |
|---|---:|---:|---:|---:|
| E1 NORMALIZED | 20/20 | 0 | 20 | 0.073–1.000 |
| DeepSets | 20/20 | 0 | 4 | 0.091–1.000 |
| LocalPaletteTransformer | 20/20 | 0 | 16 | 0.085–1.000 |

Cada pareja I–Luby comparte números aleatorios; parejas distintas son
independientes. No se aplicó un test de dos muestras independientes a esas
parejas. La desigualdad de acoplamiento y cotas binomiales simultáneas dan
las cotas superiores de TV condicional. Cotas de eventos originales y nuevos
dan los efectos inferiores TV(original,I), con cobertura conjunta al menos
95%. Se observaron 0 COMMIT inseguros, sin corregirlos.

Ninguna celda logró una cota superior ≤0,05: no se estableció recuperación
aproximada de la ley de Luby. Una cota ≤0,05 habría respaldado equivalencia
aproximada sólo de I con Luby.
Un límite superior grande puede reflejar un acoplamiento poco ajustado; no
demuestra discrepancia residual. El efecto controlado del canal de propuestas
es distinto de identificar mediación completa o separar grado de paleta.
Estos resultados no cambian ninguna de las 720 hipótesis originales.

### Suplemento exploratorio con los testigos originales

El addendum `0d41d79d62a1acab76e9658a88e40183e72ee490` se registró después de inspeccionar
ejemplos originales y antes de cualquier muestra intervenida. Conserva el
único evento elegido por el procedimiento original en cada una de las 60
celdas; no lo cambia usando la intervención. Reestima sus probabilidades en
el holdout original y combina la cota TV(original,Luby) con la cota de
acoplamiento TV(I,Luby), mediante la desigualdad triangular. Evalúa además
el mismo evento bajo I y obtiene una cota directa TV(original,I). Ambas
cotas pertenecen a esta misma familia simultánea, por lo que se toma su máximo.

| Arquitectura | Celdas | Efecto TV(original,I)>0 | Mayor cota inferior del efecto |
|---|---:|---:|---:|
| E1 NORMALIZED | 20/20 | 12 | 0.875 |
| DeepSets | 20/20 | 11 | 0.628 |
| LocalPaletteTransformer | 20/20 | 20 | 0.917 |

Las 360 probabilidades (seis por celda) tienen cobertura simultánea al menos
95% mediante CP y unión de eventos. Ésta es una familia distinta de la tabla
anterior: no se toma el máximo entre sus cotas ni se afirma cobertura conjunta
95% sobre ambas familias. La figura 12 utiliza este suplemento. No se añadieron
simulaciones ni se modificaron los 720 contrastes principales. Las celdas con
cota cero se conservan y no prueban ausencia de efecto.


## P10. ¿Qué podemos afirmar sobre pérdida de equivalencia?

Las cotas positivas de eventos respaldan diferencias entre leyes completas
condicionales en los grafos y checkpoints indicados. Un no rechazo no prueba
igualdad. La equivalencia TV≤0,05 en los 35 casos E1 pequeños anteriores
conserva su alcance previo y no se extrapola a n≥128. No obtenemos una cota
superior útil de TV a partir de MMD pequeña o medias similares.

En los casos de cliques, la uniformidad condicional puede proceder de la
equivarianza de vértices y de que toda coloración propia usa cada color una
vez. Ese teorema puede coexistir con sesgos de candidatos. En regulares
generales hay varios perfiles de clases de color y esa transitividad no
está disponible. Ni la igualdad de un clique ni los otros casos pequeños
estadísticamente próximos establecen generalización universal.

Las seis formas binomiales exploratorias se compararon con interceptos de
checkpoint y tres folds que dejan fuera grafos completos. Para A, la menor
pérdida predictiva fue:

- E1 NORMALIZED: `interaction`, pérdida por ejecución 0.12263; se archivan las seis formas y los tres folds.
- DeepSets: `interaction`, pérdida por ejecución 0.09094; se archivan las seis formas y los tres folds.
- LocalPaletteTransformer: `interaction`, pérdida por ejecución 0.10933; se archivan las seis formas y los tres folds.

| Forma predictiva para A | E1 | DeepSets | Transformer |
|---|---:|---:|---:|
| size | 0.48869 | 0.33092 | 0.42547 |
| degree | 0.22415 | 0.28409 | 0.27689 |
| ratio | 0.20248 | 0.09703 | 0.12509 |
| density | 0.19993 | 0.09668 | 0.12287 |
| additive | 0.12568 | 0.09099 | 0.11106 |
| interaction | 0.12263 | 0.09094 | 0.10933 |

Se presentan todas las formas, sin convertir la selección predictiva en una
hipótesis causal confirmada ni en una escala universal de discrepancia.
La razón n/K y la densidad están ligadas por el propio diseño; una pequeña
diferencia de predicción no identifica una variable causal única. Los
coeficientes, convergencia y pérdidas de cada fold están archivados.

## Auditoría y recursos

Hubo 570,000 intentos confirmatorios incluyendo
holdouts, con estados [570000, 0, 0, 0] en orden [válida, inválida,
cutoff, error]. Los controles de traslado pasaron 29 pruebas, reprodujeron V2
y registraron 12/200, 10/200, 10/200 rechazos nulos y 0/20 en el puntual,
con intervalos exactos archivados. Son controles de leyes empíricas iguales;
la validación extensa anterior sigue siendo la referencia de calibración.
Estrella m=500 al umbral 0,05/720: V2 30/30, antiguo 0/30; límite CP puntual
inferior de potencia 0,884. Difuso: ambos 30/30. No se garantiza esa potencia
para toda alternativa: persisten las limitaciones de orden alto y masa rara.

La tabla `event-precision-and-power-guarantees.csv` aporta garantías
conservadoras adicionales para los contrastes Bernoulli preregistrados. Con
500 completaciones válidas por banco, efectos absolutos de aproximadamente
0,094 (celda de arquitectura), 0,133 (cambio adyacente) o 0,189 (interacción)
son suficientes para potencia al menos 80% por el componente Hoeffding,
incluyendo su presupuesto simultáneo. No son garantías de potencia del
omnibus, de TV arbitraria ni de obtener siempre 500 éxitos antes del cutoff.
Para diferencias menores las cotas pueden ser poco informativas.

Se verificaron 650,656 semillas nuevas distintas, sin
solapamiento con 5,406,784 históricas, y
2,720 auxiliares. Se auditaron hashes, paletas,
validez, estados, independencia de particiones, un banco completo Luby, dos
replays V2 y Holm. Coste: **10.185/12 horas GPU-proceso**;
la estadística usa CPU. El commit de protocolo es
`5633b547682b129bb0f0baa81f6ad9fe1012eb25`, anterior a los nuevos bancos.

## What We Can Prove, What We Observe, and What Remains Open

**Demostrable:** simetría de colores del profesor ideal en regulares con
paletas comunes; invariancia inicial de MAX a K fijo; validez condicional del
test por intercambio y cobertura simultánea bajo ejecuciones independientes;
un evento con probabilidades distintas implica leyes distintas. Las identidades
ideales no son certificados bit a bit de toda la aritmética numérica.

**Observado:** efectos, interacciones, replicaciones, variación entre grafos y
checkpoints y perfiles de propuestas/COMMIT de esta cohorte. Los intervalos
delimitan cuáles diferencias respaldan las muestras.

**Abierto:** ordenar las TV completas, separar grado de paleta, establecer
mediación completa y extender estas conclusiones a todos los grafos o a todos
los entrenamientos. Los no rechazos grandes no establecen equivalencia.

Las nueve figuras requeridas, los suplementos y el PDF de 16 páginas con
todas las distribuciones de U se regeneran de los datos archivados. Se
conservan los resultados y secciones anteriores del manuscrito.
