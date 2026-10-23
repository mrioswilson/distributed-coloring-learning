# Resumen factorial

Matriz 4×4 completada: tres grafos regulares nuevos por celda, quince
checkpoints históricos, 500 intentos por generador más 125 de holdout.

| Arquitectura | V2 | MMD antiguo | Sólo V2 | Sólo MMD |
|---|---:|---:|---:|---:|
| E1 NORMALIZED | 115/240 | 49/240 | 66 | 0 |
| DeepSets | 170/240 | 121/240 | 49 | 0 |
| LocalPaletteTransformer | 180/240 | 174/240 | 6 | 0 |

En los tres grafos nuevos n512-d100, la menor cota TV del evento preregistrado
U≤100 entre las quince comparaciones E1 fue 0.023.
Conservando A y T, las quince parejas tienen TV≥0.078, con
la cobertura simultánea de los eventos preregistrados.
La respuesta al tamaño depende del grado y del evento. A grado 50, las bandas
simultáneas respaldan que |D_T| cae al menos 0.138 entre
n128 y n1024. A grado 100, D_A cambia +0.481 [+0.384, +0.579]
entre n256 y n512. Estos cambios en probabilidades no ordenan las TV
completas. Las cinco E1 difieren considerablemente y se mantienen separadas.
Grado y paleta siguen confundidos.

Los testigos independientes muestran también diferencias en propiedades que
el uso total de colores puede ocultar. Las definiciones, probabilidades e
intervalos de los ejemplos E1, DeepSets y Transformer aparecen en el informe.
La intervención separada en candidatos respaldó un efecto sobre la ley completa en 40/60 celdas y TV(I,Luby)≤0,05 en 0/60. La familia suplementaria de testigos respaldó un efecto en 43/60 celdas, con su propia cobertura simultánea del 95%.

Auditoría: 650,656 semillas nuevas sin solapamiento;
10.185/12 horas GPU-proceso; sin entrenamiento ni push.
