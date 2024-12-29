#! /bin/bash
DIR=${1:-results_nations_002}
METRIC=${2:-mrr}

# Remove the last /.
DIR=$(echo $DIR | sed 's/\(.*\)\/$/\1/')

SIGNATURES=$(ls ${DIR}/log*_?_*.csv | sed -e "s/^${DIR}\///" | sed -e "s/ .*.csv//" | cut -d_ -f1,3- | sed -e "s/_....\-..\-..$//"|sort|uniq)

echo "Processing $DIR $SIGNATURES for metric $METRIC"

echo Output1
for S in $SIGNATURES; do
    FILENAME=$(echo $S | sed 's/_/_?_/')
    echo -n "$S "
    cat ${DIR}/${FILENAME}*.csv | \
    grep -o -e 'test_results:.*$' | \
    grep -o output_1_${METRIC}\':..............| tr ',:' ' ' | awk '{print $2}' | \
    ./avgAndStdErrorLastField.sh 2>/dev/null
done | sort

echo Output2
for S in $SIGNATURES; do
    FILENAME=$(echo $S | sed 's/_/_?_/')
    echo -n "$S "
    cat ${DIR}/${FILENAME}*.csv | \
    grep -o -e 'test_results:.*$' | \
    grep -o output_2_${METRIC}\':..................| tr ',:' ' ' | awk '{print $2}' | \
    ./avgAndStdErrorLastField.sh 2> /dev/null
done | sort
