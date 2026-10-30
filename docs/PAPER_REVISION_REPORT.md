# Revisión ejecutada del artículo — 30 de octubre de 2026

Se revisó y modificó el manuscrito científico de Martín, base c3affd16.
La entrega está en la copia aislada `/workspace/paper-revision-20261030`,
rama `codex/paper-revision-20261030`. El usuario autorizó el commit y push de
esta entrega el 30 de octubre, después de las tres auditorías.
La copia original y sus ochenta archivos previos se conservaron.

## Cambios del artículo

- Nuevo título, abstract, introducción y conclusión: fidelidad de continuaciones,
  progreso y reparación; tres contribuciones delimitadas. La cobertura nativa
  heterogénea pasa a ser contexto, con detalles en apéndice.
- Definiciones autocontenidas del oráculo por profesor, fases y admisibilidad,
  con ejemplo KW verificado por replay y diagrama vectorial de las dos
  comunicaciones, componentes aprendidos y trabajo suministrado.
- Reanálisis BE que distingue ocurrencias, tipos suficientes, vistos/no vistos,
  elegibilidad, silencio y ocupación. Se reproducen los quince numeradores
  históricos antes de añadir métricas; las etiquetas y datos originales no cambian.
- Intervenciones de un componente en los dos fallos seleccionados: una propuesta
  libre destraba Johansson y permite completar la reparación GM. Sustituir sólo
  COMMIT por su etiqueta exacta no rescata ninguno. Se conserva el batch original
  de dieciséis casos, la primera etapa y todos los demás mensajes P/C.
- Texto explícito sobre qué cifras son resultados históricos y qué análisis se
  reprodujeron aquí. No se presenta el total histórico de 810.000 como regenerado.

## Nuevo experimento prospectivo

Se entrenaron cuarenta modelos: BE/GM con E1 de contexto nativo, diez semillas
por profesor y dos brazos emparejados. Protocolo, fuentes y configuración
congelados antes del entrenamiento; todos los pesos congelados antes del test.
1.500 actualizaciones fijas, 6.600 objetivos de optimización por modelo,
contextos P/C comunes y costes separados. Veinte por ciento de buckets de
tipos suficientes COMMIT retenidos, sin intersección train/test. Las familias
de intervención siguen siendo las mismas: no se afirma generalización a
familias nuevas ni igualdad de todo el coste del simulador.

Test final: 80.000 consultas emparejadas, 160.000 predicciones de continuación,
40.000 ejecuciones autónomas y 20.000 referencias exactas exitosas.

| Profesor | BA nativo | BA CF | Diferencia media CF−nativo, intervalo bootstrap por semillas | Éxitos autónomos nativo / CF |
|---|---:|---:|---:|---:|
| BE | 84,64% | 90,41% | +5,77 puntos [−13,73; +25,26] | 9.204 / 7.405 de 10.000 |
| GM | 50,00% | 100,00% | +50,00 puntos [+50,00; +50,00] | 10.000 / 10.000 |

El intervalo BE incluye cero y dos semillas CF empeoran fuertemente. En GM,
el intervalo degenerado refleja diez diferencias observadas idénticas; no
es una garantía universal. BE sube de 51,70% a 86,94% por tipo, pero baja en
éxito autónomo. Estos resultados adversos se incluyen y limitan cualquier
afirmación de beneficio uniforme. No hubo selección o reentrenamiento después
de abrir el test. Todos los resultados por semilla están en el artículo.

## Correcciones científicas tras la auditoría adversarial

La métrica primaria, la partición y todos los modelos/resultados anteriores
se conservan. Se añadieron diagnósticos posteriores al test desde los bancos
congelados, marcados como tales en el artículo; no constituyen otro ensayo.

| BE, medias sobre diez semillas | BA por ocurrencias | Exactitud por tipo |
|---|---:|---:|
| Nativo | 84,64% | 51,70% |
| CF | 90,41% | 86,94% |
| Regla programada de elegibilidad y participación | 99,71% | 81,27% |

La regla programada ignora ocupación y obtiene BA mayor que ambos modelos.
Un tipo negativo ocupa el 95,05–96,30% de cada banco BE; hay 17–23 tipos entre
4.000 consultas. Por eso la BA no basta para identificar la condición de
ocupación. Entre consultas elegibles y activas, CF rechaza 228/228 opciones
ocupadas y acepta 814/995 libres; nativo y regla aceptan 995/995 libres pero
rechazan 0/228 ocupadas. Son recuentos descriptivos sobre diez bancos, no
1.223 réplicas de entrenamiento. La ventaja CF por tipos coexiste con 181
falsos negativos en esa franja y menos éxitos autónomos.

GM nativo recibe 4.344–4.354 confirmaciones positivas y cero negativas de
ASSIGN/REPAIR por semilla; CF recibe 1.038–1.098 negativas. DETECT tiene
supervisión de conflicto separada y ambas clases. La BA 50→100% muestra
aprendizaje de rechazos ausentes del control nativo de color, sin comparar
contra otra adquisición que también aporte negativos informativos.

El split fino válido retiene combinaciones de contexto, pero al agrupar por
fase/elegibilidad/participación/ocupación, CF ya vio configuraciones que cubren
99,475–100% de las ocurrencias de test BE y 100% GM. Hay 82–92 tipos finos
por banco GM. Se corrigen abstract, introducción, discusión y conclusión para
no atribuir extrapolación a condiciones lógicas nuevas.

El apéndice especifica el desplazamiento de opción, silencio, cursor previo
al filtro, cuotas, clave completa BE/GM, serialización y hash. Publica
recuentos por fase/semilla y matrices de confusión. La fuente es
`scripts/audit-revision-continuations.py`; resultados e identidades de entrada
están en `results/paper-revision-20261030/continuation-audit.json`.
Las claves y etiquetas se reconstruyen desde los arrays, incluidas las
claves de entrenamiento. No se reentrena ni reejecuta el test neuronal.

El recibo de fuentes original autentica su lista, no todas las dependencias
transitivas o el entorno; no se reescribe retrospectivamente. El control de
COMMIT en los dos rescates se identifica como sustitución de identidad en
esos estados ya correctos. La Figura 4 compacta se conserva. Los README se
actualizan al build vigente y el builder histórico rechaza la fuente revisada
antes de sobrescribir PDF/recibos. Sigue pendiente revisión humana independiente.

## Correcciones tras la segunda auditoría

Se incorporan dos mecanismos BE al cuerpo y al apéndice, mediante el generador
de assets y una sonda portable separada, sin cambiar el ensayo consumido.
En CF 105001/105008, la BA COMMIT sobre las 3.600 consultas auxiliares
adquiridas es 99,510%/100%, frente a 52,756%/52,094% final. Todos los objetivos
auxiliares fueron visitados. El fallo observado es de generalización entre
contextos pese al ajuste adquirido; no identifica una causa única de
optimización ni una imposibilidad arquitectónica.

BE 105010 completa sólo 204/1000 en ambos brazos, con BA COMMIT final
99,451%/99,949%. La primera etapa común omite participación en 13/118 raíces
elegibles del banco final. En estados auxiliares omite 4/128 y activa una
raíz ineligible; estos valores P/C son condicionamiento COMMIT, no objetivos
directos de pérdida de participación. Las propuestas coinciden con las nativas
y son libres en las raíces elegibles de ambos bancos.

El replay conserva el primer lote original de 32 casos y reproduce las 32
salidas. En el primer caso, paso 7, raíz 31, el límite KW baja de 21 a 17 mientras
el auxiliar sigue en 17 por participación falsa. La propuesta 0 decodifica 12
libre; cambiar sólo participación de la raíz hace que COMMIT acepte. Se
localiza esa primera violación sin afirmar que explique los 796 fallos ni
que una ejecución intervenida completa: la intervención fue una consulta
local y las acciones originales siguieron conduciendo el replay.

`scripts/probe-revision-be-failures.py` evalúa doce modelos seleccionados,
102.128 consultas COMMIT y el lote original. Las 48.000 predicciones finales
reproducen las matrices originales; todos los resultados y el witness
coinciden con la auditoría previa. Identidades de entrada y evidencia:
`results/paper-revision-20261030/be-failure-diagnostics.json`.
Son diagnósticos posteriores al test, sin retuning ni estimación de prevalencia.

Se precisa además que se aprenden transformaciones de estado oculto y
funciones de mensajes; el intérprete aplica colores/auxiliares y avanza fases,
cola y reloj. El bloque de reproducción del artículo incluye ahora la
auditoría de continuaciones y esta sonda antes del builder, igual que el README.

## Reproducción y entrega

La tercera auditoría retira la antigua Figura 3 de líneas emparejadas: repite
la tabla y las diez curvas GM coinciden. Retira también las barras de dos
fallos, cuyas cifras y condiciones ya se conservan en el texto y las tablas.
Los archivos gráficos previos permanecen intactos. El artículo conserva el
diagrama de arquitectura y el contraste histórico; la nueva Figura 3 muestra
fragmentos reales de los dos witnesses y el cambio de una propuesta, con
resultados completos en la tabla de intervenciones trasladada al apéndice.
No son gráficos adicionales de los mismos porcentajes. El generador verifica
aristas, colores, propuestas y controles contra los JSON congelados.

La auditoría identifica además una precisión del denominador: 14,53/13,43
pasos BE son medias sobre todas las ejecuciones, no medias condicionadas al
fallo. Todos los éxitos toman 15 pasos en ambos brazos; las medias entre
fallos son 9,0804/8,9576. Se aclara el texto y se guarda la contabilidad en
`be-step-accounting.json`. La tabla por semilla distingue ahora BA en porcentaje
y diferencias en puntos porcentuales. El informe de la tercera auditoría es
`docs/PAPER_REVISION_AUDIT3.md`; los resultados primarios se conservan.

Se recuperaron y comprobaron los 4.261 archivos del paquete CF y los 225/225
checkpoints del cierre por su SHA256, desde archivos públicos anteriores.
Se reconfirmaron los cuarenta pesos originales de Pedro disponibles aquí, sin
atribuirles un nuevo replay. Las rutas históricas pasan a resolverse localmente
o mediante una raíz configurable, verificando la identidad del artefacto.
Los manifiestos históricos quedan intactos; el nuevo manifiesto usa los bytes
entregados realmente y distingue la normalización CRLF/LF de los CSV previos.

Fuente canónico: `paper/ucgnn/main.tex`; PDF: `paper/ucgnn/main.pdf`.
El PDF/fuente anterior permanece en `paper/ucgnn/archive/pre-revision-c3affd16.*`.
La revisión inicial tuvo 23 páginas; la cuenta vigente y los hashes están en
`results/paper-revision-20261030/paper-build.json`. Compilación nativa ARM con
pdfLaTeX/BibTeX y recibo propio,
sin referencias indefinidas ni cajas desbordadas. Se inspeccionaron los renders
de todas las páginas y las tablas/figuras principales a resolución de página.
El compilador x86 histórico no se utilizó en ARM.

Protocolo y comandos: `docs/PAPER_REVISION_PROTOCOL.md` y
`docs/PAPER_REVISION_REPRODUCE.md`. Datos, pesos y estados de reanudación nuevos
quedan en `data/paper-revision-20261030`, con índice y archivo portátil en
`results/paper-revision-20261030`. Los recibos originales de congelación se
conservan; un replay portátil separado comprueba sus hashes al reubicar datos.

## Comprobaciones y límites

En la revisión inicial, **499 pruebas pasaron**, sin fallos, errores ni omisiones en la batería del cierre,
los nuevos contratos y las regresiones de rutas históricas.
Los resultados finales de regresiones y verificación están en
`results/paper-revision-20261030/regressions.xml` y `verification.json`.
Se comprueban fuentes congeladas, identidad de pesos, presupuestos, etiquetas,
separación de tipos, emparejamiento y PDF. La inspección visual y estas pruebas
son revisión propia; queda pendiente una revisión científica humana independiente.
Los 810.000 registros crudos del cierre anterior siguen sin estar publicados.
El replay portátil BE/105002/CF reproduce sus diagnósticos y las 1.000 salidas
finales (colores, estados y pasos), manteniendo los hashes originales.
No se inventa la reproducción del cierre ni una revisión externa.

La primera corrección científica entregó un PDF de 26 páginas, comprobado contra las
fuentes y sus tablas, sin referencias pendientes ni desbordes. Se verificaron
de nuevo los cuarenta modelos, las etiquetas, la partición y 300 archivos del
runtime; los resultados originales y el archivo experimental coinciden con
sus hashes. Esos diagnósticos concuerdan con la primera auditoría y
sus estratos suman exactamente las matrices originales. Su recibo conservado es
`results/paper-revision-20261030/scientific-correction-checks.json`. No se
presentan las 499 regresiones anteriores como ejecutadas otra vez en esta tarea.

Tras la segunda auditoría, el PDF tenía **27 páginas**, sin referencias
pendientes, cajas desbordadas ni caracteres fuera de página. Se inspeccionaron
las tres hojas de contacto y las páginas modificadas en detalle. La verificación
de los 40 modelos, etiquetas, partición y emparejamiento pasa; los 40 resultados
originales, 190 fuentes científicas y 300 archivos runtime siguen intactos,
incluido el archivo experimental por SHA256. Las 102.128 consultas y el witness
coinciden con la segunda auditoría. El recibo de esta corrección es
`results/paper-revision-20261030/audit2-correction-checks.json`; no reemplaza
los recibos anteriores ni constituye revisión humana independiente.

La tercera auditoría mantiene 27 páginas, con la antigua gráfica de pares
y las barras retiradas, un esquema de mecanismos verificado y la tabla de
intervenciones en el apéndice. Se recomputan etiquetas/claves/controles de
80.000 consultas emparejadas y los bootstrap desde las cuarenta matrices,
sin otra inferencia neuronal. El recibo vigente es
`results/paper-revision-20261030/third-audit-checks.json`; el informe completo
es `docs/PAPER_REVISION_AUDIT3.md`. Las congelaciones y los resultados
originales permanecen intactos.

## Entrega en GitHub autorizada

La rama de entrega es `codex/paper-revision-20261030` en
`mrioswilson/distributed-coloring-learning`. Incluye fuentes, PDF revisado,
figuras, generadores, pruebas, protocolos, resultados completos del ensayo,
diagnósticos y archivo portátil del runtime. El manifiesto autentica los
235 archivos de la entrega; sus recibos de congelación originales se conservan.
El SHA del commit publicado se consulta en Git y en el relevo de coordinación
de la publicación, evitando una referencia circular en este mismo archivo.
Esta entrega no crea una release nueva ni modifica la rama main. La revisión
científica humana independiente sigue pendiente.
