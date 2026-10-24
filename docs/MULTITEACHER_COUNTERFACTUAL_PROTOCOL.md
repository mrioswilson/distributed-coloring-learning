# Protocolo multiteacher contrafactual

Protocolo `multiteacher-counterfactual-v1`; configuración completa en
`configs/multiteacher-counterfactual.json`. Rama de trabajo independiente:
`research/counterfactual-multiteacher-learning`, base `61f39760`. La congelación
es local, mediante hashes; no se afirma prerregistro externo. La distribución
y los controles se especifican antes de observar las nuevas confirmaciones.

## Objetos y causalidad

Sea S una instantánea nativa admisible, a una intervención tipada en las
decisiones disponibles en la fase, y xi la aleatoriedad privada necesaria.
Se define

\[
\mathcal O_T(S,a,\xi)=D_T^\phi(R_T^\phi(S,a,\xi)),
\]

donde R reconstruye causalmente las cantidades descendientes de a y D
consulta la decisión de la fase phi. La fase, sus permisos y los datos
algebraicos derivados no pueden modificarse independientemente. Se distinguen
dos consultas: **decisión nativa**, que usa la regla original y un estado
repetible; y **continuación de propuestas**, que pregunta si una propuesta
tipada confirma bajo los mensajes efectivamente suministrados. La segunda
extiende defensivamente un componente y no convierte una oferta no nativa en
una trayectoria exacta del profesor. Su fidelidad se mide por separado.

Cada adaptador declara dominio, variables intervenibles, dependencias,
componentes, restricciones y casos difíciles en `multicf/oracles.py`.
`states.py` sólo produce estados mediante prefijos de ejecución verificables.
Cada testigo conserva grafo, cinta, paso, hash de instantánea, acciones de
prefijo y ofertas intervenidas. `replay` rechaza cambios en la historia,
aritmética obsoleta, permisos o hashes. Los mensajes se vuelven a codificar
desde S; nunca se edita una igualdad o disponibilidad derivada de forma aislada.

## Adaptadores específicos

| Profesor | Estado admisible e intervención | Dependencias que se recalculan | Restricción importante |
|---|---|---|---|
| Luby | Coloración parcial propia; draws frescos; P/C alternativos | Paletas, rangos, ofertas y colisiones | COMMIT usa todas las ofertas intervenidas finales |
| Johansson | Coloración parcial propia; draw de candidato; ofertas alternativas | Igual a Luby | Participación nativa es 1; silencio es consulta de continuación fuera de esa ley nativa |
| ColorTrials | Preferencias discretas privadas; propuestas consultadas; prefijos seguros | Disponibilidad propia y vecina, mensajes por color, comparaciones estrictas | La segunda etapa comunica preferencias y disponibilidad aprendida; no usa ofertas monocolor como sustituto |
| JP | IDs distintos fijos por historia; nuevo ID exige nuevo replay | Órdenes en todas las aristas, elegibilidad y paletas | Un vértice no máximo no confirma por una participación forzada |
| BE | Colores auxiliares propios acotados por fase; punto Linial u opción KW legal | Polinomios, valores, igualdades, offsets, nuevo bound | Linial y KW tienen oráculos distintos; colores bajos de KW se conservan |
| GM | Prefijo de asignaciones libres bajo cuatro bloques, detección y reparación | Slots, permisos, colores reales, marcas y cola | No se etiqueta una historia especulativa arbitraria sólo porque el simulador la procese |
| Greedy | Prefijo propio en el orden suministrado | Orden, paleta y primer color | Sólo escribe la raíz programada |

Los prefijos alternativos eligen un punto Linial no bloqueado u otro color
libre/ganador donde la continuación está definida; cada transición se valida.
Una nueva asignación de IDs se obtiene inicializando otra historia completa.
No se permutan IDs a mitad de una historia dejando slots u órdenes obsoletos.
Los profesores históricos permanecen idénticos.

## Tratamientos y atribución

La selección inicial eficiente prioriza BE y GM con B nativo=3000 y tres
tratamientos, cinco semillas nuevas 94101–94105 y tres arquitecturas. Cada
entrenamiento conserva hitos 500, 1500 y 4000, con batches de 64 raíces,
Adam .001 y fases muestreadas cíclicamente. El mismo banco, inicialización y
secuencia de posiciones se usan en los tres brazos de cada celda/semilla.

- `passive`: referencia exacta de la pérdida pasiva histórica; COMMIT de la
  salida aprendida recibe la etiqueta docente almacenada. Su eventual
  discordancia se diagnostica, no se oculta.
- `matched`: COMMIT se evalúa con propuestas docentes y su etiqueta docente.
  Corrige concordancia sin una nueva adquisición docente.
- `counterfactual`: cien actualizaciones de calentamiento concordante;
  después .5 COMMIT docente + .5 COMMIT de un banco finito intervenido.
  P y C conservan sus objetivos nativos. Los cuatro candidatos se recorren en
  cada raíz; en profesores de ofertas se varían también mensajes vecinos y
  silencios. Son consultas nuevas, adquiridas y almacenadas una vez.
- `owncf`: mecanismo adicional inspirado en M6B/Pedro, con .5 COMMIT de las
  ofertas propias y .5 del mismo banco contrafactual. Todas las nuevas
  consultas propias distintas se cobran y se conservan en el checkpoint.

Los hitos dentro de un brazo permiten estudiar optimización con idénticas
etiquetas adquiridas para los bancos fijos. `owncf` puede adquirir más etiquetas
al seguir optimizando y no permite esa atribución aislada. `matched` vs
`counterfactual` mantiene el banco de estados y la arquitectura, pero cambia
el presupuesto y la distribución de etiquetas de confirmación; no prueba
eficiencia muestral a igual total de consultas. `passive` vs `matched` estudia
concordancia y entradas de COMMIT, no exclusivamente contrafactuales.

La cobertura prospectiva incluye aislados, grafos sin aristas y mezclas con
componentes propias. El banco inicial usa prefijos docentes; el generador de
prefijos alternativos está implementado y probado como tratamiento separado.
Cambiar cobertura o prefijos requiere identidad de dataset distinta. Las
ampliaciones para ColorTrials o tratamientos posteriores se registran antes
de entrenar y se seleccionan sólo con development. No se declara agotada una
capacidad por un fallo de esta primera curva.

## Regímenes de información e inferencia

Luby/Johansson/ColorTrials permanecen anónimos. JP y GM reciben relaciones de
orden coherentes. BE recibe fases y aritmética local declarada; GM/Greedy su
planificador nativo. Ninguna función nueva de etiquetas se importa desde los
modelos o la predicción autónoma. Se conservan los tres núcleos principales,
la disponibilidad aprendida, candidatos y COMMIT neuronales. No se añaden
veto de colisiones, reparación, máscaras de disponibilidad exacta ni consultas
docentes al camino de inferencia.

## Datos, consultas y recursos

Training usa tamaños 8/12/16/24. Development usa 60 ejecuciones N28/N36,
separadas por namespace de training. Final usa exactamente 1000 ejecuciones
N32/N48/N64 por checkpoint, diez familias prospectivas de 100 casos cada una,
incluidos grafos sin aristas y mezclas desconectadas. Los IDs/cintas son nuevos
por caso y semilla. El control de regresión singleton 83105 es adicional.
Las familias deterministas pueden repetirse como clases de isomorfismo; los
tamaños separan la identidad entre training y evaluación principal.

Cada raíz de color adquiere P/C/COMMIT; Linial adquiere un punto y DETECT una
marca. El banco agrega cuatro etiquetas COMMIT por raíz de color, cobradas
conservadoramente incluso si dos consultas coinciden. Se guardan cantidades
por componente, bytes, pasos de simulador, estados realmente usados,
presentaciones, visitas por etiqueta, actualizaciones y tiempos CPU/wall.
Los bancos se comparten entre arquitecturas y brazos, distinguiendo coste
de adquisición física del acceso supervisado por modelo. Un label almacenado
y repetido no es una nueva consulta. Las consultas propias se cachean por
estado/raíz y por el conjunto completo de mensajes efectivos.

El piloto aislado de 21 modelos utilizó 24 actualizaciones y B nativo=600.
Las estimaciones lineales a 4000 updates fueron aproximadamente 25–53 s por
modelo; el máximo RSS por worker fue 592 MB. Se aplica margen ×2 para decidir
la siguiente ampliación. Se usan tres workers CPU de un thread, prioridad
reducida y checkpoints cada 250 updates. La GPU no se usa en esos trainings.
El replay histórico M6B sí utiliza CUDA cuando está disponible y se contabiliza
separadamente. Los tiempos observados prevalecen sobre estas estimaciones.

## Selección, congelación y confirmación

La enmienda prospectiva `configs/multiteacher-counterfactual-screen.json`,
registrada antes de ejecutar el nuevo screen y antes de toda confirmación,
amplía development de 60 a 300 casos por semilla. Conserva el prefijo de 60
ya observado, los pesos y los bancos. Se exige desarrollo perfecto común de
las cinco semillas sobre esos 300 casos, dentro de un tratamiento y
presupuesto, para seleccionar el primer hito candidato. Los screens fallidos
quedan conservados. No se han usado resultados finales para esta enmienda.
Se congelan cinco hashes y configuración antes de usar el final. Los
profesores exactos validan todas las cintas bajo los horizontes prospectivos
del estudio previo. Se escribe un marcador de consumo antes de la primera
ejecución estudiantil; no hay reintentos silenciosos de una confirmación.
Un tratamiento desarrollado después de observar un fallo final necesita
nuevo namespace y nueva batería. Los candidatos fallidos permanecen.

PASS requiere cada una de cinco políticas con 1000/1000, invariantes nativos,
paleta, propiedad, terminación en horizonte y ausencia de errores técnicos.
Los conflictos temporales legales de GM no son fallos por sí solos. El gate
debe contener todos los IDs exactamente una vez. Se informan denominadores
por política y 0/5..5/5; no se infiere PASS desde development. Los doce PASS
históricos se conservan como referencia, sin repetirlos ni reinterpretarlos.

Las mediciones de fidelidad se separan en acciones nativas, transiciones
reales, continuaciones contrafactuales, seguridad/progreso y pasos. La
comparación de distribuciones usa bancos independientes y conserva estados
terminales incorrectos. Ni éxito funcional ni no rechazo de un test prueban
identidad de algoritmos o igualdad de leyes. La transferencia N128/256/512,
cuando se ejecuta, es exploratoria y posterior a la selección funcional.

## Registro de decisiones científicas

1. Base integrada reciente para no perder la contabilidad y los testigos del
   estudio anterior; worktree nuevo para preservar historia.
2. Separar corrección de etiquetas de aumento de optimización: el código
   pasivo revela un cambio de contexto COMMIT que merece control propio.
3. Bancos finitos antes de experimentar con consultas propias: permite
   comparar 500/1500/4000 a consultas fijas y medir coste de cf explícitamente.
4. Prefijos verificables en lugar de instantáneas inventadas: especialmente
   necesario en GM y BE.
5. Aislados en training y development prospectivos; regresión histórica
   adicional y test final independiente.
6. No atribuir éxito de Pedro a un solo mecanismo: sus diferencias de
   entrenamiento, representación y evaluación son múltiples.
7. Preservar resultados negativos e incompletos; las extensiones quedan
   identificadas como exploratorias y no sustituyen el protocolo histórico.
