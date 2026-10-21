# Local palettes and distinguishability — generated report

Graph cases observed: 18848 / 18848 planned. Conditional laws: 3255.
Finite-sample classifier rows: 1260. Task states: {'complete': 160, 'running': 0, 'pending': 0, 'technical_failure': 0}.
GPU-process hours: 3.0239 / 12.

```text
    treatment        regime  successes  cases  mean_rounds  unsafe_commits
     deepsets        GLOBAL        760    760     9.018421               0
     deepsets         LOCAL        760    760     9.172368               0
     deepsets OVERSIZED-1.5        760    760     8.617105               0
     deepsets   OVERSIZED-2        760    760     8.471053               0
         mean        GLOBAL        431    760     8.307895           15586
         mean         LOCAL        431    760     8.426316           18727
         mean OVERSIZED-1.5        453    760     8.271053           10784
         mean   OVERSIZED-2        463    760     8.100000            8648
   normalized        GLOBAL        760    760     8.859211               0
   normalized         LOCAL        760    760     9.025000               0
   normalized OVERSIZED-1.5        760    760     8.596053               0
   normalized   OVERSIZED-2        760    760     8.350000               0
raw_endpoints        GLOBAL        637    760    54.181579               0
raw_endpoints         LOCAL        637    760    54.213158               0
raw_endpoints OVERSIZED-1.5        637    760    54.181579               0
raw_endpoints   OVERSIZED-2        637    760    54.181579               0
          sum        GLOBAL        160    760   203.618421               0
          sum         LOCAL        160    760   203.578947               0
          sum OVERSIZED-1.5        160    760   203.559211               0
          sum   OVERSIZED-2        160    760   203.434211               0
      teacher        GLOBAL        152    152     8.973684               0
      teacher         LOCAL        152    152     8.973684               0
      teacher OVERSIZED-1.5        152    152     8.559211               0
      teacher   OVERSIZED-2        152    152     8.328947               0
  transformer        GLOBAL        760    760     9.821053               0
  transformer         LOCAL        760    760     9.781579               0
  transformer OVERSIZED-1.5        760    760     9.264474               0
  transformer   OVERSIZED-2        760    760     9.121053               0
```

Primary outcome is finite-cohort proper completion. Exact-rule-violating
authorizations, conflicts and cutoffs are separately recorded in CSV. Shared
graphs are not independent observations for each network. See the preregistered
protocol for pairing, intervals, RNG conversion and independent sample splits.
Reports may be regenerated during execution and do not imply full completion.
