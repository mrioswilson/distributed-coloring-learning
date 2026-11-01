# Protocolo de verificación del release de Martín

Fecha: 31 de octubre de 2026. Auditoría posterior a los resultados publicados,
sin reentrenamiento, selección adicional ni nuevo test independiente.
Los paquetes se fijan por los tres SHA256 aportados por el usuario y por
los manifiestos internos; no se reemplazan archivos distintos preexistentes.
La entrega integrada previa se conserva completa y se verifican sus bytes.

1. Verificar/restaurar los tres paquetes y sus inventarios. Reconciliar las
   220 referencias entre freezes, las rutas distintas, entrenamientos y
   snapshots adicionales. Los controles reutilizados no se vuelven réplicas.
2. Recalcular etiquetas de las consultas desde arrays y oráculos explícitos,
   validar bancos y respuestas contra hashes originales y recomponer las
   métricas publicadas, sin cambiar su ponderación ni estratos.
3. Re-inferir todos los checkpoints congelados en sus baterías originales:
   200 respuestas del estudio y 20 del follow-up, con los 15 controles
   reutilizados evaluados en la nueva batería. Usar el orden y bloques256
   del código publicado. Comparar logits y signos con las respuestas guardadas.
   Guardar toda discrepancia, aunque no cambie una métrica agregada.
4. Repetir la evaluación funcional primaria de los 100 modelos y de los 20
   modelos/controles del follow-up sobre sus propios casos originales. Los
   15 controles del follow-up no son entrenamientos nuevos. Comparar éxitos,
   fallos por clase/familia y pasos medios con los resúmenes; guardar todas
   las filas nuevas. Las filas funcionales originales por caso no se afirman
   disponibles cuando el código solo guardó un resumen.
5. Auditar las 810 celdas/810.000 filas históricas: hashes, casos, metadatos,
   reconstrucción de grafos, colores terminales, paleta, conflictos y flags.
   Recalcular todos los conteos por celda, incluido el primario449791/450000.
   Se autentican flags intermedios guardados, no se reconstruye toda trayectoria.
   Leer/verificar filas no equivale a reejecutar810000inferencias autónomas.

Los modelos, los bancos y las respuestas son los ya congelados; no se
excluyen semillas ni variantes negativas y no se cambian endpoints. Los
resultados nuevos irán a `results/martin-artifacts-20261031/`. Las corridas
de inferencia se realizan en serie y con un hilo CPU por las restricciones
de memoria observadas en este host. Los checkpoints adicionales no se
evalúan para buscar un hito mejor. Revisión científica independiente pendiente.
