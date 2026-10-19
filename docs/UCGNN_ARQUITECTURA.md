# Universal Color GNN (UCGNN): arquitectura y decisiones delicadas

Documento de diseño, escrito **antes** de entrenar. Las decisiones experimentales
(presupuestos, cohortes, criterios) están en [UCGNN_PROTOCOLO.md](UCGNN_PROTOCOLO.md).

## 1. Qué conserva y qué cambia respecto de M6B

M6B usa la arquitectura M4B: una red por grado, con la cabeza CANDIDATE de
`K+1` salidas y entradas one-hot de tamaño `K+3`. Sus parámetros dependen de Δ
(8.967 / 9.737 / 10.892 / 11.662 para Δ=3/5/8/10).

UCGNN conserva:

- el profesor (oracle sin cambios), sus dos números aleatorios por nodo y ronda
  (`r_p`, `r_c`) y el consumo idéntico del tape;
- el modelo de ejecución: ronda síncrona, dos comunicaciones entre vecinos,
  colores permanentes y tres decisiones PARTICIPATE → CANDIDATE → COMMIT;
- la supervisión contrafactual `L_P + L_C + 0,5 L_own + 0,5 L_cf`, el warmup de
  diez épocas con propuestas del profesor y las probabilidades 0,17 / 0,04;
- la inferencia autónoma, sin shield, sin oracle y sin máscaras de seguridad.

UCGNN cambia la representación: cada nodo mantiene **una representación por
color** `h(v,c)`, calculada con las mismas funciones para todos los colores.
El constructor del modelo **no recibe K**: los mismos tensores de parámetros se
aplican a 4, 11 o 17 colores. `K` sólo determina cuántas "ranuras de color" se
evalúan.

## 2. Observaciones del modelo

Por nodo `v` el modelo recibe exactamente:

| Entrada | Significado | Origen |
|---|---|---|
| `color(v)` | `-1` no coloreado o color fijado `0..K-1` | estado del protocolo |
| `r_p(v)`, `r_c(v)` | dos uniformes del tape de la ronda | tape |
| `K` | tamaño de la paleta declarada | definición del problema |
| aristas | vecinos inmediatos | topología local |

Codificaciones deterministas usadas (todas funciones de la observación local):

- `u(v) = [color(v) = -1]`;
- `own(v,c) = [color(v) = c]` (one-hot del color propio sobre las ranuras);
- tras la etapa 1: `p(v)` (participación dura) y `offer(v,c) = [c*(v) = c]`.

El modelo **no** recibe: conjunto de colores disponibles, `k = |A(v)|`, rango
del color, propuestas del profesor, etiquetas de COMMIT ni indicadores de
conflicto. Estos objetos sólo existen en el generador de etiquetas y en el
observador de diagnóstico, que se ejecuta después de decidir la acción.

## 3. Ronda: dos comunicaciones

```text
estado ──φ1──► m1(u,c) ──max vecinos──► M1(v,c) ──ψ1──► h1(v,c)
                                                      │
                     α(v,c)=σ(a(h1))  ───►  k̂, prefijo, [lo,hi)
                                                      │
          r_c(v) ─────────────────────────► s(v,c)=f(h1, α, lo, hi, r_c)
                                                      │ argmax sobre la paleta
r_p(v) ──► PARTICIPATE p(v)                 CANDIDATE c*(v)
                    │                                 │
                    └──── f2(u,c)=[u, own, p, offer] ─┘
                                 ──φ2──► m2(u,c) ──max vecinos──► M2(v,c)
                                 ──ψ2──► z(v,c) ──recoger c=c*(v)──► COMMIT
```

**Etapa 1.** Cada nodo envía, para cada color `c`, `m1(u,c) = φ1([u(u), own(u,c)])`.
El receptor agrega con máximo por color: `M1(v,c) = max_{u∈N(v)} m1(u,c)`.
`h1(v,c) = ψ1([u(v), M1(v,c)])`.

**Selección universal (sección 5).** A partir de `h1` se obtiene una disponibilidad
aprendida `α(v,c)`, un prefijo aprendido y la puntuación `s(v,c)`.

**PARTICIPATE.** `P(v) = g_P([r_p(v), u(v)])`.

**Etapa 2.** Con las propuestas duras, `m2(u,c) = φ2([u(u), own(u,c), p(u), offer(u,c)])`,
`M2(v,c) = max_{u∈N(v)} m2(u,c)`, `z(v,c) = ψ2([f2(v,c), M2(v,c)])`. COMMIT lee la
ranura de su propio candidato: `COMMIT(v) = g_C([z(v,c*(v)), p(v), u(v), Σ_c offer(v,c)])`.

Todas las funciones (`φ1, ψ1, a, f, g_P, φ2, ψ2, g_C`) son MLPs compartidos. Las
agregaciones son máximos sobre vecinos (escala independiente del grado) y una
suma/prefijo sobre colores de la disponibilidad aprendida.

## 4. Por qué el máximo y no la suma

Una suma sobre vecinos produce valores cuya escala crece con el grado. Entrenado
con Δ=3, un modelo con suma vería en Δ=10 magnitudes nunca observadas. El máximo
por ranura de color depende sólo de qué tipos de vecino existen para ese color,
no de cuántos. La entrada de cada ranura tiene así el mismo soporte para cualquier
Δ: "algún vecino fijado con c", "algún vecino participa proponiendo c", etc.
Ésta es la hipótesis arquitectónica central de la transferencia.

## 5. Selección universal del candidato

El profesor elige `A(v)[floor(r_c·k)]`. Equivalentemente: los `k` colores
disponibles, en orden numérico, dividen `[0,1)` en `k` intervalos iguales y se
elige el color cuyo intervalo contiene `r_c` (muestreo por CDF inversa):

```text
c elegido  ⇔  c disponible  y  P_<(c)/k ≤ r_c < (P_<(c)+1)/k,
P_<(c) = número de colores disponibles menores que c.
```

UCGNN implementa la misma estructura sobre predicciones aprendidas:

```text
α(v,c)  = σ(a(h1(v,c)))                         disponibilidad aprendida
k̂(v)    = Σ_c α(v,c)                            alternativas aprendidas
S_<(v,c) = Σ_{c'<c} α(v,c')                     prefijo exclusivo aprendido
lo = S_< / max(k̂, 0,5),  hi = (S_< + α) / max(k̂, 0,5)
s(v,c)  = f([h1(v,c), α(v,c), lo, hi, r_c(v)])
```

Decisiones delicadas:

1. **El prefijo es sobre `α`, no sobre la disponibilidad exacta.** Nada en la
   entrada indica qué color está ocupado. Si `α` se equivoca, el prefijo se
   desplaza y la selección falla. `α` no tiene pérdida propia: se aprende sólo a
   través de `L_C` (no se añade una pérdida auxiliar de disponibilidad, para no
   alterar `L` prescrita).
2. **Coordenadas normalizadas.** La comparación `r_c·k ≥ P_<` requeriría que un
   MLP extrapole un producto a `k` no observados. En coordenadas normalizadas la
   condición es lineal (`r_c - lo ≥ 0`, `hi - r_c > 0`) y el rango de `lo, hi, r_c`
   es `[0,1]` para cualquier `K`. Lo que cambia con Δ es el ancho de los
   intervalos (`1/k` hasta `1/11`); en Δ=3 sólo se observan anchos `1, 1/2, 1/3, 1/4`.
   Es el principal riesgo de extrapolación. `f` debe *aprender* la comparación.
3. **Es un sesgo inductivo fuerte.** La normalización y el prefijo son operaciones
   fijas, como la normalización de una atención. No resuelven la decisión: la red
   debe aprender qué colores ocupan los vecinos, que `α` es la cantidad a acumular
   y cómo comparar `r_c` con el intervalo. Una ablación preregistrada (`RAW_PREFIX`)
   entrega a `f` el prefijo y `k̂` sin normalizar, para medir el papel de la
   normalización en la transferencia.
4. **Guarda numérica `max(k̂, 0,5)`.** Evita divisiones por cero al principio del
   entrenamiento. En estados reales `k ≥ 1`.
5. **Sin opción "sin candidato" para nodos no coloreados.** Con grado ≤ Δ y Δ+1
   colores, `A(v)` nunca es vacía y el profesor nunca usa esa clase. Los nodos
   coloreados no tienen candidato por protocolo. La única máscara es la de ranuras
   fuera de la paleta declarada (relleno al mezclar paletas en un lote).

## 6. Dependencias del orden de colores

- Única operación dependiente del orden: el prefijo exclusivo `S_<` en orden
  numérico ascendente de colores (la regla del profesor está definida así).
- No hay codificaciones posicionales absolutas (`c/K`, embeddings de color). Un
  color nunca observado en entrenamiento (p. ej. el color 7 con Δ=3) se procesa
  con las mismas funciones que el color 0.
- La etapa 2 y COMMIT no usan el orden: son **equivariantes** ante cualquier
  permutación global de colores (comprobado por test). La regla de seguridad no
  puede depender de "qué color" se propone.
- `α` y `h1` también son equivariantes por ranura; sólo `lo/hi` y, por tanto,
  `s`, dependen del orden.

## 7. COMMIT universal

La decisión lee una sola ranura, la del candidato propio. En esa ranura `M2`
resume, con funciones compartidas: algún vecino tiene fijado `c*`; algún vecino no
coloreado participa proponiendo `c*` (la conjunción de `p(u)` y `offer(u,c*)` del
mismo vecino la aprende `φ2` antes del máximo, como en M2B/M4B); el nodo participa.
No hay una salida por color: la misma función decide para `c*=0` o `c*=10`.
El gradiente de COMMIT no atraviesa las propuestas duras, igual que en M6B.

## 8. Recursos: qué depende y qué no depende de Δ

| Recurso | UCGNN | M4B/M6B especialista |
|---|---|---|
| Parámetros | constante (ver `count_parameters`) | crece con `K` |
| Rondas de comunicación por iteración | 2 | 2 |
| Mensaje implementado por arista y etapa | `K × 32` reales | 32 reales |
| Información mínima equivalente del mensaje | etapa 1: `⌈log2(K+1)⌉` bits; etapa 2: `2⌈log2(K+1)⌉+1` bits | igual |
| Memoria de activaciones | `O((N + E)·K·d)` (aristas por fragmentos: `O(N·K·d + B·K·d)`) | `O((N+E)·d)` |
| Cálculo local por nodo | `O(K·(deg+1)·d + K·d²)` | `O(deg·d + d² + K·d)` |

El mensaje aprendido es una función determinista del estado del emisor (color,
participación, candidato), por lo que puede reconstruirse en el receptor a partir
de `O(log K)` bits. Lo informamos por separado: la implementación transmite
tensores `K×32` y **no** es independiente del grado en tamaño de mensaje, memoria
ni cálculo. Sólo los parámetros y el número de rondas por iteración lo son.

## 9. Riesgos anticipados (antes de ver resultados)

- Anchos de intervalo no observados (`1/k`, `k>4`) en la selección.
- Acumulación de errores pequeños de `α` en paletas grandes (`k̂` y prefijo).
- Mayor densidad de fronteras de decisión con `K` grande: más estados cerca de un
  umbral y, por tanto, más discrepancias con el profesor; no necesariamente
  inseguras.
- COMMIT nunca vio vecindarios con más de tres vecinos coloreados; la agregación
  por máximo debería hacerlo irrelevante, pero es una hipótesis a medir.
