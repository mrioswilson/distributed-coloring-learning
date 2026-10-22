# Resumen ejecutivo: pruebas robustas

El instrumento se validó antes de la nueva comparación neuronal. El fallo
histórico de la estrella se reprodujo exactamente: TV=99/199 coexistía con
una señal MMD² 16 veces menor que su desviación nula a m=250. La calibración
primaria antigua usó suficientes permutaciones y el ajuste Holm era correcto.

El nuevo omnibus separa selección y evaluación y calibra conjuntamente
testigos locales, de interacciones, estructurales y globales. Detectó la
estrella en 30/30 ensayos independientes incluso al umbral .05/795; IC95
de potencia [.886,1]. Una segunda versión amplió la búsqueda de pares tras
un fallo de localización de la primera. Se conservaron los fallos frente a
triples remotos y eventos raros: no se afirma potencia universal.

Una réplica nula independiente de 4.000 ensayos, preregistrada después de
congelar el método, produjo 202/4000 rechazos
(5.05%; IC95 4.41–5.77%).
No se modificó el instrumento a partir de esa réplica.

| Arquitectura | Histórico m=250 | MMD antiguo, banco nuevo m=500 | Omnibus nuevo, total m=500 |
|---|---:|---:|---:|
| NORMALIZED (E1) | 10/265 | 21/265 | 52/265 |
| DeepSets | 46/265 | 71/265 | 107/265 |
| LocalPaletteTransformer | 127/265 | 147/265 | 181/265 |


De los 255 no rechazos históricos de E1, 42 ahora son significativos.
El histórico usó m=250; la comparación emparejada nueva presupuesta 500
ejecuciones para ambos procedimientos. El omnibus reserva la mitad de
las completadas para selección y registra los tamaños efectivos.
Los recuentos de rechazo no son distancias distribucionales.

Fase confirmatoria: 2,867,996 coloraciones válidas,
0 inválidas completas, 4 cutoffs y
0 errores técnicos. Las leyes de coloraciones se contrastan
condicionadas a completar antes del límite; los fallos se conservan aparte.

Se respaldó equivalencia aproximada TV<=.05 en 78/105 celdas pequeñas
mediante cotas simultáneas prefijadas. En cliques, la igualdad condicional
ideal procede de un teorema de simetría. En los demás casos, no rechazar no
demuestra igualdad. Hay 188 rechazos con
testigos observables validados en bancos independientes.

Auditoría: 4,469,184 semillas nuevas,
888 bancos; 6.268/12 horas GPU-proceso.
No se reentrenaron redes ni se hizo push. El informe completo responde
P1–P10 y conserva todas las decisiones, incertidumbres y limitaciones.
