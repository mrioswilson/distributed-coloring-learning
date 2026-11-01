# Paquete de revisión independiente — revisión de Pedro (30-oct-2026)

Leer primero `CLAUDE_TO_CHATGPT_PEDRO_REVIEW.md`, que es el informe autocontenido. Después:

- `CHANGE_SUMMARY.md`: mapa de cambios de B (`c3affd16`) a P (`5993a5e4`).
- `EVIDENCE_INDEX.md`: fuente y estado de cada afirmación.
- `pedro-original-docs/`: handoff y documentos originales de Pedro, sin modificar.
- `manuscript-pedro/`: PDF revisado, fuentes LaTeX compilables y diff de `main.tex` frente a B.
- `manuscript-baseline/`: nuestra versión anterior (tex y pdf).
- `evidence/pedro-results/`: JSON compactos del experimento y de las auditorías de Pedro, copiados de git.
- `verification/`: scripts y salidas de las comprobaciones de esta revisión.
- `MANIFEST.sha256`: hashes de todos los archivos del paquete.

Los datos grandes (60 JSON crudos de unos 224 MB y el runtime de 26 MB) no se incluyen. Están en git en `5993a5e4`, con sus hashes en `EVIDENCE_INDEX.md`.
