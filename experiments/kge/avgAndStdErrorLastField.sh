#! /bin/bash
FILE=/dev/stdin

if [[ -n $1 ]]; then FILE=$1; fi

cat $FILE | awk '{s+=$NF; V[NR]=$NF}END{if (NR>0) avg=s/NR; stddev=0.0; for(i in V) {stddev += (avg-V[i])*(abg-V[i]);} if (NR>1) stddev/=(NR-1); stddev=sqrt(stddev); stddev/=sqrt(NR); print avg,stddev,NR}'


