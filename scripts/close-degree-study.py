#!/usr/bin/env python3
"""Write the scientific closure from completed immutable outputs."""
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
import numpy as np
from dcl.degree.inventory import ROOT
from dcl.degree.report import table


def fmt_range(xs):
    return str(min(xs)) if min(xs)==max(xs) else f"{min(xs)}–{max(xs)}"


if __name__ == "__main__":
    summary=json.loads((ROOT/'summary.json').read_text())
    rows=list(csv.DictReader((ROOT/'checkpoint-condition.csv').open()))
    main=[r for r in rows if r['stage']=='confirmatory' and r['primary']=='True']
    failures=json.loads((ROOT/'failure-index.json').read_text())
    probes=json.loads((ROOT/'counterfactual-probes/index.json').read_text())
    pairs=list(csv.DictReader((ROOT/'counterfactual-probes/paired-equal-update.csv').open()))
    certs=json.loads((ROOT/'commit-certificates/index.json').read_text())
    native=json.loads((ROOT/'native-21-consolidation.json').read_text())
    primary_bad=[f for f in failures if f['stage']=='confirmatory' and f['primary']]
    native_bad=[f for f in primary_bad if f['delta']==3]
    transfer_bad=[f for f in primary_bad if f['delta']>3]
    failure_counts=Counter((f['stage'],f['failure']) for f in failures)
    ctable=[]
    for t in ('COLOR_TRIALS','SPECULATIVE_GM'):
        for a in ('normalized','deepsets','transformer'):
            vals=[]
            for condition in ('d3-global','d10-global','d200-global'):
                xs=[float(r['accuracy_difference'])*100 for r in pairs if r['teacher']==t and r['architecture']==a and r['control']=='matched' and r['condition']==condition]
                vals.append(f"{np.mean(xs):+.2f} [{min(xs):+.2f},{max(xs):+.2f}]")
            ctable.append([t,a,*vals])
    certtable=[]
    for t in ('LUBY','JOHANSSON','GREEDY'):
        for a in ('normalized','deepsets','transformer'):
            xs=[r for r in certs if r['teacher']==t and r['architecture']==a]
            certtable.append([t,a,sum(x['float32_model']['certified_exact_rule'] for x in xs),
                              sum(x['float32_model']['certified_no_false_accept'] for x in xs),
                              fmt_range([len(x['float32_model']['certified_unsafe_masks']) for x in xs])])
    text='''# Fallos, límites y mecanismos de transferencia

## Separación de causas

**A, profesor.** Seis variantes tienen pruebas condicionales para ambas
paletas. Las referencias exactas validaron todas las cintas confirmatorias
antes de ejecutar estudiantes. BE no es aplicable fuera de grado tres y
no aporta fallos neuronales artificiales.

**B, interfaz.** Los seis profesores extensibles disponen de información
suficiente y evaluación estructural de pesos compartidos. La forma histórica
K4/vecinos3 se reemplaza explícitamente por una organización dispersa
semánticamente conservadora, contrastada con el envoltorio original.
La interfaz algebraica BE está especializada. No se demuestra capacidad
universal de ningún ancho fijo. Que no exista una pérdida informacional
identificada no descarta todas las limitaciones de representación.

**C, política congelada fuera del dominio aprendido.** Un error sobre un
estado legítimo, con referencia correcta y entradas conservadas, es evidencia
de fallo de esa política. No separa automáticamente cobertura, optimización
y capacidad. Las sondas de CT sí aíslan una dependencia concreta de la
comparación aprendida; los demás mecanismos requieren los contrastes
descritos, sin atribuciones causales por el nombre de la arquitectura.

## Fallos observados

'''
    text+=table(['Etapa','Tipo','Ejecuciones'],[[a,b,n] for (a,b),n in sorted(failure_counts.items())])+'\n\n'
    text+=f'En la cohorte principal confirmatoria hubo {len(native_bad)} fallos con Δ3 y {len(transfer_bad)} con Δ>3. '
    text+='Los controles se conservan por separado; un fallo basal K4 no se interpreta como provocado por aumentar grado.\n\n'
    text+='El piloto incluye 42 cutoffs de recursos CT y dos colisiones reales del control matched/E1/s94105. '
    text+='El matched CT a 4000/s94104 también tiene una regresión nativa documentada. '
    text+='El techo de diez segundos sólo se aplicó a celdas exploratorias densas CT aún no completadas; '
    text+='los resultados anteriores largos permanecen, por lo que esos cutoffs no son comparables como una tasa homogénea. '
    text+='Ninguno demuestra no terminación matemática. Las discrepancias de acciones anteriores al cutoff sí son testigos de falta de fidelidad.\n\n'
    text+='Todos los fallos están en `failure-index.json`, con checkpoint, semilla, grado, tamaño, familia y ruta de filas/traza. '
    text+='El primer componente discrepante es una observación temporal. La discordancia alpha>.5 es calibración de una variable latente, no una etiqueta supervisada ni un requisito demostrado de corrección: los scorers pueden usar h1 o compensar otra parametrización de alpha. '
    text+='La confirmación añade la primera divergencia de acción y guarda ambos estados cuando procede. '
    text+='Las métricas de regla sobre estados autónomos no equivalen a fidelidad de toda una trayectoria docente; después de un historial GM inválido no se usan como prueba de fidelidad nativa.\n\n'
    examples=[]
    seen=set()
    for f in sorted(failures,key=lambda f:(f['stage'],f['delta'],f['n'],f['case_id'])):
        key=(f['teacher'],f['architecture'],f['method'],f['primary'],f['stage'])
        if key in seen:continue
        seen.add(key)
        d=f['first_divergence'] or {}
        examples.append([f['stage'],f['teacher'],f['architecture'],f['method'],f['seed'],f['delta'],f['n'],f['family'],f['failure'],d.get('component','—')])
    text+=table(['Etapa','Profesor','Arq.','Método','Semilla','Δ','n','Familia','Resultado','Primer componente'],examples)+'\n\n'
    text+='## Sonda que separa grado local, paleta y preferencias CT\n\n'
    text+='Se construye una arista v–u y, en otra componente, una estrella de grado Δ. '
    text+='En régimen local, ambos extremos de la arista tienen siempre grado uno y paleta {0,1}. '
    text+='Con todos pendientes, pref_v(0)=Δ, pref_u(0)=Δ−1 y las restantes preferencias cero, v gana estrictamente color cero. '
    text+='Son intervenciones privadas discretas válidas para la regla extendida; no se afirma haber encontrado una semilla PCG64 que emita el vector completo construido. '
    text+='La normalización conservadora convierte la diferencia en 1/Δ.\n\n'
    mechanism=json.loads((ROOT/'counterfactual-probes/mechanism-summary.json').read_text())
    text+=table(['Δ','Paleta receptor','Rechazos del ganador exacto /15','Rango logits COMMIT'],
                [[r['delta'],r['endpoint_palette'],r['exact_winner_rejected'],f"[{r['min_forced_logit']:.4f},{r['max_forced_logit']:.4f}]"] for r in mechanism])+'\n\n'
    text+='Los quince pesos CF rechazan el ganador desde Δ10 en esta sonda, aun forzando explícitamente la consulta de color cero. '
    text+='El error persiste con un único vecino y paleta propia constante: no lo explica perder multiplicidades en MAX ni recibir demasiados vecinos. '
    text+='Las preferencias normalizadas más próximas cambian la respuesta de COMMIT, y las magnitudes de los logits negativos son mucho mayores que las discrepancias numéricas observadas en los replays. '
    text+='No es una prueba universal de robustez numérica, pero tampoco evidencia de saturación o redondeo que borre el dato de entrada. '
    text+='Es un testigo de fidelidad de componentes, **no una estimación del fracaso funcional aleatorio**, pues futuras rondas pueden resolver el grafo.\n\n'
    text+='La elección de normalización es explícita: el adaptador principal conserva el template pref/Δ (pref/3 en el dominio nativo). '
    text+='Una sonda posterior mantiene unidades antiguas pref/3 mediante una transformación común sólo de la observación, sin cambiar profesor ni pesos. '
    text+='Con gap crudo1, ese segundo embedding acepta 13/15 ganadores a Δ5, 5/15 a Δ10 y0/15 a Δ50/200, frente a15/15,0/15,0/15 en el principal. '
    text+='Ambos preservan el orden exacto; ninguno resuelve todos los casos medidos. No se escoge un embedding por resultado ni se reescriben las evaluaciones principales. '
    text+='La conclusión es sensibilidad a la escala/rango de las preferencias bajo estos embeddings, no una imposibilidad informacional ni una afirmación sobre todo adaptador posible. '
    text+='El contraste completo está en `ct-normalization-sensitivity.json`.\n\n'
    text+='## Contrafactual frente a matched al mismo hito\n\n'
    text+='Sondas exploratorias sobre prefijos docentes exactos: 120 checkpoints y 838188 consultas. '
    text+='CT se compara a 4000 updates usando los treinta controles adicionales; GM a 500. '
    text+='La siguiente tabla muestra la media descriptiva de las cinco diferencias apareadas en puntos porcentuales y su rango entre semillas, '
    text+='sin promediar políticas ni fingir independencia de consultas que comparten estado.\n\n'
    text+=table(['Profesor','Arquitectura','Δ3 global','Δ10 global','Δ200 global'],ctable)+'\n\n'
    text+='CF mejora estas respuestas frente a matched, pero la ventaja no crece monótonamente con Δ. '
    text+='Los denominadores positivos/negativos y diferencias por checkpoint están en `counterfactual-probes/checkpoint-metrics.csv` y `paired-equal-update.csv`. '
    text+='False accept en esa tabla significa discrepancia de la **cabeza COMMIT**; si participación es falsa no se ejecutaría una escritura insegura. '
    text+='Las propuestas/query se intervienen sólo aquí y jamás corrigen un rollout funcional. '
    text+='Se controlan inicialización, minibatches nativos, arquitectura y updates; CF añade consultas y cambia la pérdida/distribución. '
    text+='No se identifica una mejora causal de la naturaleza contrafactual a igual presupuesto total de etiquetas, ni una garantía de seguridad a cualquier grado.\n\n'
    text+='## Johansson/Transformer: progreso y diversidad de propuestas\n\n'
    text+='La política histórica s83101 conserva 28 cutoffs de 1000 en n64/Δ10/global, todos en la familia de cliques. '
    text+='El replay conserva **todo el batch original** y reproduce colores, fallo y 128 pasos, evitando cambiar el contexto numérico del testigo. '
    text+='Al final quedan dos vértices adyacentes con colores disponibles {0,2}. Ambos proponen color2 en las últimas rondas; '
    text+='COMMIT los rechaza correctamente porque Johansson está siempre activo. Una sonda con 8192 draws nuevos por vértice produce color2 en todos ellos, '
    text+='frente al reparto uniforme de la propuesta exacta entre dos colores. No se demuestra que la masa del otro color sea exactamente cero. '
    text+='El testigo sitúa la pérdida de progreso en la diversidad de propuestas de ese estado, sin convertir un COMMIT correcto en fallo de detección. '
    text+='No hay pérdida del conjunto disponible en la interfaz; sigue abierta la separación de cobertura/optimización frente a capacidad del scorer de atención. '
    text+='La traza y los histogramas están en `failure-replays/JOHANSSON-transformer-historical-primary-passive-s83101/`.\n\n'
    text+='GM sólo exige permisos/paleta en el gate histórico y admite conflictos temporales. '
    text+='Los replays adicionales registran también conflictos entre slots diferentes, omisiones de escrituras programadas y cobertura de detección; '
    text+='no se presentan esas propiedades adicionales como comprobadas para todas las 810000 ejecuciones. '
    text+='Una reparación fortuita o una política final correcta no prueba que se hayan conservado todos los invariantes intermedios del algoritmo exacto.\n\n'
    text+='En el primer testigo GM/Transformer/local/s94102, el paso32 es REPAIR de un vértice con lista {0,1}; '
    text+='la referencia exige escribir color1 y los tres procedimientos proponen color0. Sus scores y alphas son idénticos en ese estado. '
    text+='Matched/pasivo aceptan el color prohibido con logits14.2269/15.2735; CF lo rechaza con logit−13.6503. '
    text+='El rechazo CF evita esa escritura, pero conserva el conflicto previo y el turno de reparación ya se consume. '
    text+='Los tres fallan las mismas 181 instancias locales. Este replay distingue la propuesta incorrecta, la mejor respuesta CF y la insuficiencia de esa mejora para el calendario fijo.\n\n'
    text+='Un segundo control construye una arista, una estrella de grado3 y dos aislados (n8), con IDs únicos y bloques coherentes que sitúan la arista en el mismo slot. '
    text+='Sobre prefijos exactos de GM, la reparación exige color1 bajo globalK4 y bajo la lista local {0,1}. '
    text+='Los tres checkpoints Transformer/s94102 proponen correctamente1 con globalK4 y erróneamente0 con localK2; CF lo rechaza y los otros lo aceptan. '
    text+='Las otras cuatro semillas Transformer y los quince pesos de cada E1/DeepSets eligen1 en ambos regímenes. '
    text+='Este caso válido a **grado máximo tres** separa el cambio de paleta del incremento de grado: los 181 fallos locales no prueban que el grado4 por sí solo sea la causa. '
    text+='Se conserva el vector de IDs y todas las respuestas en `palette-degree-mechanism.json`; es una sonda elegida después del fallo, no una nueva certificación.\n\n'
    text+='## Certificados de COMMIT y garantías que siguen abiertas\n\n'
    text+='Se reutiliza el certificador histórico de intervalos exteriores en 45 núcleos canónicos Luby/Johansson/Greedy. '
    text+='El canal de mensajes binarios tiene seis tipos legales y 64 soportes; MAX no depende de sus multiplicidades. '
    text+='Los certificados conciernen al receptor pendiente que participa y consulta un color, con gather de una sola posición. '
    text+='No se aplican ignorando residuales nativos en JP, CT, BE o GM.\n\n'
    text+=table(['Profesor','Arquitectura','Regla exacta /5','Sin falsa aceptación /5','Nº soportes inseguros certificados'],certtable)+'\n\n'
    text+='Ninguno de estos 45 pesos pasivos satisface el criterio suficiente de COMMIT exacto o sin falsa aceptación en todos los soportes, '
    text+='incluso bajo el modelo de error float32 del certificador. Se conservan intervalos positivos en soportes que deberían rechazarse. '
    text+='Eso es un contraejemplo de la cabeza bajo intervenciones, no un contraejemplo automático de su ejecución autónoma: '
    text+='la política de propuestas y el calendario pueden evitar tales soportes. En Greedy sólo hay un permiso, por lo que soportes con otros participantes no son alcanzables en su calendario. '
    text+='Los soportes con vecino fijo del mismo color también prueban una limitación condicional, sin demostrar que el propio proponente lo elija.\n\n'
    text+='La proposición de seguridad aprendida sigue siendo condicional: si las propuestas pertenecen a su lista y cada escritura cumple disponibilidad y exclusión de ofertas iguales, '
    text+='la coloración permanente es propia por inducción. Para terminación se requiere además una cota positiva de progreso uniforme en todos los estados alcanzables. '
    text+='Validar infinitamente muchos grados o demostrar esa cota no se sustituye por estos finales finitos. '
    text+='Los intervalos reales encierran operaciones por redondeo exterior float64; el certificado numérico float32 asume la cota gamma para affine, IEEE, ausencia de overflow/TF32 y MAX de mensajes no negativos. '
    text+='No es una verificación formal de todos los kernels del hardware.\n'
    Path('docs/DEGREE_GENERALIZATION_FAILURE_ANALYSIS.md').write_text(text)

    closed='''# Revisión de cierre científico del artículo

## Recomendación

**Alternativa 2, con alcance acotado:** incorporar una sección de transferencia
congelada y sus resultados negativos/mecanísticos, conservando el resultado
nativo heterogéneo 21/21. No ampliar esa frase a siete profesores en grado
arbitrario. Seis variantes admiten extensión; BE sigue siendo un caso base.
La pregunta sobre qué regla se aprendió une las pruebas informacionales,
supervisión CF, éxito funcional y contraejemplos de fidelidad. El estudio
no necesita implementar BE recursivo ni entrenar en alto grado para cerrar
esta contribución finita.

| Alternativa | Solidez y unidad | Coste pendiente y riesgo | Decisión |
|---|---|---|---|
| 1. Sólo dominio nativo | 21 combinaciones con evidencia trazable, sin protocolo homogéneo | Menor alcance; dejaría fuera transferencia ya terminada y un contraejemplo explicativo | Defendible si el espacio editorial obliga; conservar extensión como suplemento |
| 2. Añadir transferencia acotada | Pruebas de seis variantes, controles nuevos comunes, pesos congelados y diagnósticos de fidelidad | Explicar adaptador, paletas y cutoffs; no vender el piloto Δ200 como certificación | Recomendada para la versión actual |
| 3. Dos artículos | Otra pregunta más amplia puede separar efectos de Δ, K, familias, entrenamiento y certificación | Requiere grados entrenados nuevos, presupuestos CF igualados y quizá nueva variante BE | Reservar esa investigación, no frenar el artículo actual |

La novedad no se deduce de que el benchmark sea mayor ni se afirma prioridad
bibliográfica exhaustiva. La aportación concreta es separar validez del
profesor, compatibilidad de pesos, funcionalidad y fidelidad mediante
controles matemáticos y experimentales identificados.

## Contribuciones y nivel de evidencia

| Línea | Afirmación defendible | Nivel y límite |
|---|---|---|
| A. Heterogeneidad | 21/21 combinaciones funcionales en sus baterías nativas | Empírico histórico+nuevo; cinco pesos y mil ejecuciones por peso; protocolos diferentes |
| B. Contrafactual | Mejores respuestas a intervenciones en comparaciones apareadas; no igualdad universal del profesor | Empírico; presupuesto total de etiquetas difiere, y persisten errores |
| C. Información | Contraejemplos anónimos para prioridades, necesidad de auxiliares y segundo mensaje CT | Argumentos de indistinguibilidad y contratos de interfaz; no teorema de capacidad de todo ancho |
| D. Muestras/optimización | Consultas, presentaciones y updates se separan; más updates no mejora siempre | Evidencia de curvas y regresiones; no cota universal de complejidad muestral |
| E. Transferencia | Seis profesores parametrizados y ejecución de sus pesos originales fuera de grado tres | Pruebas condicionales docentes y resultados finitos; Δ cambia K en régimen global |
| F. Leyes globales | Éxito funcional y algunas fidelidades locales no igualan distribuciones de coloraciones completas | Se conserva evidencia estadística histórica; aquí no se rehace ni se amplía su objetivo principal |

## Consolidación única del 21/21

La tabla enlaza cada combinación con el inventario de cinco hashes y su
protocolo. Las doce históricas son pasivas del estudio de complejidad muestral;
las nueve nuevas principales son contrafactuales. Los matched certificados
siguen como controles secundarios. La evidencia original detallada de
familias, dominio, criterio, reproducción y limitaciones está en
`native-21-consolidation.csv/json`; no se repitieron entrenamientos.

'''
    ntable=[]
    for r in native:
        labels=[int(x) for x in r['labels_by_seed'].split(',')]
        ntable.append([r['teacher'],r['architecture'],r['cohort'],r['method'],r['seeds'],fmt_range(labels),r['updates'],r['parameters'],r['native_result']])
    closed+=table(['Profesor','Arquitectura','Cohorte','Método','Semillas','Etiquetas/semilla','Updates','Parámetros','Final nativo'],ntable)+'\n\n'
    closed+='Los cuatrocientos y miles de casos nuevos de cada peso no se utilizan para cambiar el peso escogido. '
    closed+='El inventario también conserva treinta controles CT del mismo hito 4000, que no estaban certificados históricamente y cuyo resultado basal se publica. '
    closed+='Pedro y M6B siguen reconstruidos por sus documentos específicos: no se atribuyen a esta cohorte los pesos de Pedro que no se recuperaron, ni la única semilla histórica M6B como cinco réplicas.\n\n'
    closed+='## Respuestas explícitas P1–P15\n\n'
    answers=[
      ('P1','Luby, Johansson siempre activo, ColorTrials repetido, JP síncrono, GM de cuatro bloques y Greedy programado: seis extensiones correctas mediante paleta/grado (B), con pruebas para global Δ+1 y local d(v)+1.'),
      ('P2','BE_BASE_D3 (C). Nuevos campos/reducciones o BE recursivo constituyen otra variante; no se ejecuta ni bloquea la campaña. Los otros seis conservan regla y calendario al parametrizar.'),
      ('P3','Los tres núcleos y 180 checkpoints extensibles inventariados, incluidos controles, funcionan técnicamente mediante adaptador estructural (2). El envoltorio histórico K4/vecinos3 requiere esa reorganización. BE no recibe una extensión algebraica; sus 45 pesos sólo se controlan nativamente.'),
      ('P4',f"La confirmación completa contiene {summary['stages']['confirmatory']['successes']}/{summary['stages']['confirmatory']['executions']} ejecuciones correctas entre políticas distintas. El resultado por cada una de las tres arquitecturas, profesor y peso está en RESULTS y checkpoint-condition.csv; hubo {len(primary_bad)} fallos principales, conservados. El total no describe una política promedio."),
      ('P5','Global n64: las dieciocho combinaciones pasan Δ4 en sus cinco semillas; diecisiete pasan Δ10, mientras Johansson/Transformer s83101 obtiene972/1000 (las otras cuatro1000). Local n64/Δ4: diecisiete pasan las cinco; GM/Transformer s94102 obtiene819/1000 (otras cuatro1000). Global n256/Δ4: todas pasan las cinco. Las otras dieciséis combinaciones pasan las cinco condiciones principales completas. Cinco profesores distintos de CT tienen éxitos exploratorios hasta Δ200, con una ejecución por peso; no se certifica ese rango. No se afirma monotonicidad ni que una lista local que falla a grado4 funcione a grado3: el control GM muestra precisamente un fallo local a grado3.'),
      ('P6','Se comparan Δ3/4/10 a n64 con la misma mezcla y Δ4 a n64/n256. El piloto regular mantiene n256 al variar grado y Δ3/10 al variar n64–1024. Grado y paleta global Δ+1 cambian conjuntamente: su efecto causal separado sigue abierto.'),
      ('P7','El índice de fallos y la tabla de primeros testigos identifican familia por peso; el piloto conserva dos colisiones matched CT y cutoffs CT en regulares densos. No se deduce de ausencia de fallos que una familia completa sea segura.'),
      ('P8','Las trazas distinguen disponibilidad, participación, propuesta, COMMIT y marca. La sonda CT a paleta local constante aísla rechazo COMMIT ante preferencias casi empatadas; el primer desacuerdo estadístico de otra ejecución no prueba su causa suficiente. Véase el análisis de fallos por checkpoint.'),
      ('P9','Sí mejora fidelidad de consultas COMMIT en los contrastes apareados a igual hito CT4000/GM500 frente a matched y pasivo, sin mejora universal o monótona en grado. La funcionalidad se evalúa por separado. Inicialización/batches/updates están controlados, pero las etiquetas totales no: no se aísla la naturaleza CF a presupuesto total igual.'),
      ('P10','MAX es insensible a multiplicidad pero suficiente para predicados existenciales; las tres arquitecturas usan MAX en el grafo. Los prefijos, media/1-K y atención de paleta pueden cambiar fuera de distribución. El contraejemplo CT mantiene un vecino y dos colores locales, por lo que no se explica por aumentar la agregación. Los 45 certificados canónicos no establecen seguridad universal de COMMIT para estos pesos pasivos.'),
      ('P11','JP/GM conservan IDs únicos y relaciones de orden coherentes; retirar IDs elimina información necesaria. GM y Greedy siguen recibiendo calendario/permiso, y BE aritmética y fases explícitas. No se presenta esa ayuda como protocolo anónimo aprendido. No hay ampliación del espacio de IDs más allá de 65536 vértices.'),
      ('P12','Pruebas condicionales: disponibilidad, seguridad y progreso/terminación de seis profesores en ambas listas; JP≤n, Greedy=n y GM≤2ceil(n/4)+n; cotas geométricas ideales para aleatorios. Contraejemplos informacionales y soporte MAX son matemáticos. Los pesos sólo tienen evidencia finita y contraejemplos/certificados de cabeza condicionales; progreso aprendido uniforme y seguridad universal siguen abiertos.'),
      ('P13','Priorizar CT con diferencias de preferencias nuevas, estados de lista heterogénea y controles a igual presupuesto total. Entrenar hasta10 y reservar20/50/100/200; ampliar a20 o50 sólo si la hipótesis lo exige. Otras arquitecturas/profesores se priorizan según fallos reales, no por obligación de llenar benchmarks. BE exige otro algoritmo. El plan no se ejecutó.'),
      ('P14','Integrar 21/21 acotado, CF/funcionalidad separados, información, optimización, transferencia confirmada y el límite CT en el artículo actual. Reservar entrenamiento multigrado, separación causal Δ/K y garantías uniformes/BE general para otra investigación.'),
      ('P15','La evidencia experimental principal queda cerrada al verificar denominadores, hashes, tests y PDF. Para defender el manuscrito se requiere revisión de coautores de alcance, pruebas y tablas y contraste bibliográfico editorial; no repetir entrenamiento ya certificado. Cualquier afirmación más amplia requerirá la confirmación/controles adicionales especificados, no puede obtenerse reinterpretando el piloto.')]
    for q,a in answers:
        closed+=f'**{q}.** {a}\n\n'
    closed+='## Qué queda abierto y qué no bloquea el cierre\n\n'
    closed+='Sigue abierta la extrapolación confirmatoria densa a Δ20/50/100/200 y n1024, '
    closed+='el aislamiento de grado frente a paleta, CF a igual número total de etiquetas, capacidad frente a optimización, '
    closed+='y una variante BE general. Son ampliaciones de alcance, no resultados ya obtenidos. '
    closed+='La compilación y el manifiesto final se verifican en los recibos locales; no se hace push ni publicación externa.\n'
    Path('docs/MULTITEACHER_PAPER_CLOSURE_REVIEW.md').write_text(closed)
    print('Wrote scientific closure and all fifteen answers from completed data.')
