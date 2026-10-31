#!/bin/sh
# Count occurrences of historical numbers in baseline vs Pedro main.tex (+ revision inputs)
B=$1; N=$2; D=$3
for pat in '21 teacher' '21/21' '105 selected' '105,000' '810,000' '808,413' '1,559' '28 cutoffs' '449,791' '450,000' '209 failures' '448 primary' '358,622' '181/1,000' '28/1,000' '8,192' '28,800' '3,776' '2.7774' '3.1200' '67.8117' '91$' '99.277' '99.540' '99.1157' '66.5299' '75.3632' '96.6570' '97.2439' '99.7452' '150,000' '4,000 distinct' '20,000 tape' '25,000' '3,526/3,570' '240/240' '195,000' '170 distinct' '0.996318' '0.997009' '22.6404' '9.2686' '99.9932' '60.14' '53.25' '99.99' 'All 45' 'Fifteen' 'fifteen' '83101' '94102' '16-case' 'unavailable' ; do
  b=$(grep -c -- "$pat" $B); n=$(cat $N $D/revision-*.tex | grep -c -- "$pat"); printf '%-16s base=%s pedro=%s\n' "$pat" $b $n; done
