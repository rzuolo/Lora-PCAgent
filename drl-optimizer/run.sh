#!/bin/bash

/home/rzuolo/source/Lora-PCAgent/drl-optimizer/prep-script.sh

cd /home/rzuolo/source/Lora-PCAgent/drl-optimizer/

pwd 

python run.py

cat scenario_name_log_file.txt | awk -F" "  {'print $2'} | sort -n | uniq -c  | sort | tail -6 | awk -F" " '{ if ($2 >= 0) print $2 ; }'

cat scenario_name_log_file.txt | awk -F" "  {'print $2'} | sort -n | uniq -c  | sort | tail -6 | awk -F" " {'if ($2 >= 0) print $2 ;'} | sort -d| tail -1 > maxoutput.txt
