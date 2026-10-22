# Leyes del banco independiente de controles

Esta descripción documenta los generadores ya congelados en `controls.py`
y `controls_v2.py`. No introduce alternativas, metas ni decisiones nuevas.
Todos los controles implementados son coloraciones propias con paletas
locales de un mismo grafo para ambas procedencias. No son simulaciones de
las leyes de Luby o de las redes, salvo los controles reales contra sí mismos.

## Nulos

- `point_null`: masa puntual en (0,1,0), sobre el camino de tres vértices.
- `uniform_null`: uniforme en las seis coloraciones locales válidas de ese
  camino, enumeradas explícitamente en el código.
- `discrete_null`: sobre esas mismas seis soluciones, probabilidades
  (.10,.20,.05,.40,.15,.10), idénticas para las dos muestras.
- `star_null`: estrella de D=200 hojas; hojas iid Bernoulli(.5) y centro
  independiente uniforme en {2,...,D}, con la misma ley en ambas muestras.
- `clique_null`: uniforme explícita en las 24 permutaciones de colores de
  K4. La igualdad se conoce por construcción; no se deduce de un no rechazo.
- Controles reales: dos bloques de ejecuciones independientes de Luby y de
  cada checkpoint NORMALIZED, DeepSets y Transformer con semilla 41000 en
  path3. Las dos rondas utilizan namespaces nuevos. Cada ley nula tiene
  400 repeticiones; cada repetición recibe 250 ejecuciones por muestra.

## Alternativas

Salvo `localized`, el centro de la estrella es 2 bajo P y Q. Bajo P las
D hojas son iid Bernoulli(.5). Las modificaciones de Q figuran abajo;
los vértices no mencionados conservan esa ley y son independientes de los
modificados. Los identificadores de hojas comienzan en 1, el centro es 0.
D=200 en la rejilla principal, de modo que el grafo tiene 201 vértices.

| Nombre | Modificación bajo Q | TV exacta |
|---|---|---|
| localized | Centro uniforme en {2,...,101}; bajo P es uniforme en {2,...,200}. Hojas sin cambios. | 99/199 |
| localized_4 | Hojas 1,...,4 independientes Bernoulli(.8). | (1/2) sum(j=0,...,4) binom(4,j) abs(.8^j .2^(4-j) - 1/16) |
| marginal | Hoja 1 Bernoulli(.85). | .35 |
| large_support | La misma modificación de marginal, con nuevas semillas. | .35 |
| correlation | Hoja 2 igual a hoja 1. Todas las marginales coinciden. | .5 |
| correlation_remote | Igualdad entre hojas 137 y 181, mediante una permutación fija de columnas de correlation. | .5 |
| higher_order | Hoja 3 = hoja 1 XOR hoja 2. Coinciden todas las marginales y pares. | .5 |
| higher_order_remote | La misma relación XOR trasladada a hojas 97,155,197. | .5 |
| parity_global | La última hoja es el XOR de todas las anteriores: paridad total par. Coinciden todas las marginales de subconjuntos propios. | .5 |
| structural | Todas las hojas iguales a un único Bernoulli(.5). Coinciden las marginales, cambian las clases de color y las dependencias. | 1 - 2^(1-D) |
| diffuse | Todas las hojas iid Bernoulli(.55). | (1/2) sum(j=0,...,D) abs(Binomial(D,.55)[j] - Binomial(D,.5)[j]) |
| rare | El centro pasa a 3 con probabilidad epsilon=.001. | epsilon |

`large_support` es una réplica independiente de una ley, no una clase
adicional de alternativas. Se usa para comprobar directamente la ausencia
de repeticiones de vectores completos. Su soporte tiene 2^200 elementos;
el test categórico de tablas completas puede no detectar nada cuando cada
celda observada contiene un solo vector, mientras un testigo de la hoja 1
puede separar las leyes. La auditoría conserva los recuentos de unicidad.

Las TV de correlación y paridad se obtienen porque Q es P condicionada a
un evento con probabilidad 1/2. Para structural, el evento de hojas todas
iguales tiene probabilidad 2^(1-D) bajo P. Para diffuse, el número de unos
es suficiente para la razón de probabilidades; agrupar por ese número
conserva exactamente la TV. No se aproxima mediante una distancia normal.

La sensibilidad añade D=20,1000 para localized, probabilidades .60,.70 para
marginal, .51,.53 para diffuse y epsilon=.01 para rare, en m=250,1000. En
localized se reemplaza el subconjunto de 100 por D/2 colores; la TV es
1-(D/2)/(D-1). Cambian conjuntamente n y el grado del centro en esa prueba;
no se pretende aislar un efecto causal del grado.

Cada alternativa principal se repite treinta veces en m=100,250,500,1000,2000.
Los métodos reciben exactamente los mismos vectores en cada repetición,
contando la mitad de selección dentro de m. Hay muestras nuevas para
desarrollo, validación 1, desarrollo 2, validación 2 y el chequeo de colas.
La versión V1 también se aplica a la validación 2 para comparaciones pareadas.

El triple remoto demuestra una limitación de regiones del instrumento:
el evento XOR conocido sí es fácil de contrastar. Los controles raros tienen
además la cota fundamental de potencia
1-(1-alpha)(1-epsilon)^m. Ni TV por sí sola ni esa cota superior garantizan
una potencia determinada para un test de dos muestras con leyes desconocidas.
