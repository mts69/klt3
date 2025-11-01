#!/bin/bash
count=${1:-10}
sum=0

echo -e "\n=============================="
echo " GPU Test Results ($count runs)"
echo "=============================="

for ((i=1; i<=count; i++)); do
    t=$(./main_gpu | tail -n 1)
    echo "time $i : $t"
    sum=$(awk "BEGIN {print $sum + $t}")
done

avg=$(awk "BEGIN {print $sum / $count}")
echo
echo "avg time : $avg"
