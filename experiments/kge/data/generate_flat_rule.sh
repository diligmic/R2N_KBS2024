#! /bin/bash

DIR="umls"
if [[ -n $1 ]]; then DIR=$1; fi
ODIR="."
if [[ -n $2 ]]; then ODIR=$2; fi

cat ${DIR}/train.txt | tr '()' ' ' | awk '{print $1}' | sort | uniq |     # get all predicates
    awk 'BEGIN{printf("r0:1.0:")} {printf("%s(x,y),%s(y,x),", $1, $1)}' |  # concatenate all atoms
    sed 's/,$//' | sed 's/\(.*\),\([[:alnum:]_]\+.*,.*\)$/\1 -> \2/' > ${ODIR}/rules_flat.txt

cat ${DIR}/train.txt | tr '()' ' ' | awk '{print $1}' | sort | uniq |  # get all predicates
    awk 'BEGIN{r=0}{printf("r%d:1.0:%s(x,y) -> %s(y,x)\n", r, $1, $1); r++; printf("r%d:1.0:%s(x,y),%s(y,z) -> %s(x,z)\n", r, $1, $1, $1); r++;}' > ${ODIR}/rules.txt

cat ${DIR}/train.txt | tr '()' ' ' | awk '{print $1}' | sort | uniq |  # get all predicates
    awk 'BEGIN{r=0;}{P[r] = $1; r++;}END{\
for(t in P) {\
  tail = P[t]; \
  printf("r%d:1.0:", t); \
  for (h in P) {\
    head = P[h]; \
    printf("%s(x,y),", head); \
  } \
  printf("__END__ -> %s(x,y)\n", tail);\
}}' | sed 's/,__END__ / /' > ${ODIR}/rules_flat_dcr.txt
