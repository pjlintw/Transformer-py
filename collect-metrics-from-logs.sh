#!/bin/bash
echo '['
for f in $(find experiments/logs | grep -E "$1" | sort -t- -k4 -n)
do
grep "'eval_f1'" "$f" | tail -n1 | tr "'" '"' | awk '{print $0 ","}'
done
echo '{}'
echo ']'