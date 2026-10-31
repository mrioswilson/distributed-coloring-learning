# Tercera auditoría del artículo y decisiones sobre figuras

La revisión no identifica un nuevo error científico central demostrado. Sí
encuentra redundancia visual y dos precisiones de exposición. Se corrigen en
el artículo, manteniendo el ensayo original y todos sus resultados. Es una
revisión propia; sigue pendiente la revisión científica humana independiente.

Objeto inicial: PDF de 27 páginas, SHA256
`4ad1028f4455cd771cb36f64fb528d24871700b0b0c4a27b89bb201de1880de3`,
conservado en `coordination/tasks/paper-adversarial-audit3-20261030-before`
del workspace. Versión resultante: `paper/ucgnn/main.pdf`, 27 páginas,
SHA256 `196e2fccbd3401abebc42a995cf132b5c8a4a9406633172d97eb6623ddec5984`.

1. **La antigua Figura 3 no justifica su espacio. Corregido.** Las diez parejas
   GM tienen exactamente los mismos valores, 50% nativo y 100% CF. Sus diez
   curvas se superponen y visualmente parecen una sola. BE muestra
   heterogeneidad, pero sin identificar semillas y repitiendo la tabla primaria
   y la tabla completa por semilla. No es un error numérico; es una figura poco
   informativa. Se retira la inclusión y su generación automática, conservando
   el PDF anterior y todas las cuarenta salidas. Los dos deterioros BE siguen
   explícitos en el texto y el apéndice.

2. **Las barras de dos fallos también repiten información. Corregido.** Ya eran
   más compactas que la grilla original, pero no explicaban el mecanismo:
   28/1000 cutoffs y 181/1000 colisiones están en el texto y las tablas, con
   condiciones distintas. Se retiran del cuerpo. Permanecen los 450 casos
   checkpoint–condición, las 448 celdas sin fallos y los archivos gráficos
   anteriores. Retirar esa imagen no elimina las observaciones desfavorables.

3. **El mecanismo central sí necesitaba un apoyo visual. Incorporado.** La
   nueva Figura 3 muestra dos fragmentos reales de las ejecuciones copiadas.
   En Johansson, los vértices 28/54 proponen ambos 2; el veto correcto deja ambos
   pendientes. Cambiar sólo la propuesta de 28 a 0 permite confirmar ambos y
   completar en el paso 11, frente al cutoff 128 original. En GM,11 y 40 tienen
   color 0; la propuesta 0 de la reparación se rechaza y conserva el conflicto.
   Cambiar sólo esa propuesta a 1 permite la reparación y el final propio en
   el paso 34. Los dibujos muestran aplicaciones 10→11 y 32→33; los resultados
   terminales corresponden a las ejecuciones completas. El resto de cada
   grafo de 64 vértices se omite del dibujo. Los controles de COMMIT exacto una
   vez conservan los fallos originales. La tabla con los seis resultados pasa
   al apéndice. Son dos diagnósticos seleccionados, no una política de rescate
   ni una tasa poblacional.

   El generador comprueba hashes de los witnesses originales, aristas,
   colores, propuestas, disponibilidad, cambio de un componente y coincidencia
   del control. Lee registros congelados, sin ejecutar otra campaña neuronal.
   Evidencia: `failure-mechanisms.json` y `component-rescues.json` bajo
   `results/paper-revision-20261030`.

4. **El denominador de los pasos BE era ambiguo. Corregido.** Los 14,53/13,43
   son medias sobre todas las ejecuciones, no sobre las fallidas. Los 9204
   éxitos nativos y 7405 CF toman todos 15 pasos. Los 796/2595 fallos promedian
   9,0804/8,9576 pasos. La disminución agregada incorpora salidas anticipadas;
   no es una aceleración de las ejecuciones exitosas. Se aclara el texto y
   se genera `be-step-accounting.json` desde los veinte resultados originales.
   También se precisa en la tabla por semilla que BA usa porcentaje y su
   diferencia puntos porcentuales.

No agregaría otros gráficos de resultados a esta versión. Se conserva la
arquitectura porque delimita lo aprendido y lo suministrado. El contraste
histórico mantiene juntos el diagnóstico completo y el subconjunto de tipos
no vistos, con sus controles y límites explícitos. La comparación de BA con
exactitud por tipo tiene sólo seis filas y se consulta mejor como tabla;
una nueva barra repetiría las cifras. Una grilla de cobertura toda exitosa,
curvas GM coincidentes o un ranking entre condiciones distintas no aportan
una pregunta adicional. La figura nueva se dedica a explicar las decisiones
locales que las tasas finales por sí solas no muestran.

Se revisaron contrato/oráculos, arquitectura e intérprete, definición del
split, presupuestos, unidades de replicación, pruebas condicionadas, alcance
de los diagnósticos posteriores al test y correspondencia de figuras/tablas
con sus fuentes. La recomputación desde arrays de los veinte bancos coincide
exactamente con el audit anterior: etiquetas, claves, partición, frecuencias,
clases, controles y solapamiento lógico. Son 80000 consultas finales emparejadas,
no 80000 réplicas de entrenamiento. Las cuarenta matrices originales reproducen
las medias y los intervalos bootstrap de diez semillas; BE conserva +5,77pp
[−13,73,+25,26] y GM +50pp [50,50]. Esta vuelta no repite inferencia sobre los
102128 ejemplos de la revisión anterior ni vuelve a entrenar modelos.

No se descubrió un contraejemplo nuevo a los argumentos matemáticos
condicionados examinados. Tampoco se convierte el split contextual en una
prueba de generalización lógica nueva, el control GM positivo en una
comparación con otra adquisición informativa, ni los dos witnesses en una
explicación de todos los fallos. Los crudos completos históricos de 810000
observaciones siguen ausentes; sus totales continúan atribuidos. Esta vuelta
no es otra búsqueda bibliográfica exhaustiva ni una revisión externa.

El recibo de cierre y los checks del PDF están en
`results/paper-revision-20261030/third-audit-checks.json`. El manifiesto vigente
identifica los bytes entregados. Las congelaciones científicas originales,
los pesos, el protocolo, la partición y los resultados previos se conservan.
