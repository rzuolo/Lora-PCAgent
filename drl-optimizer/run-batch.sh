#!/bin/bash

IFS="
"
roundtrip=`ls -v batch-experiments/ranges/|grep plus`

for i in $roundtrip
do
	cp batch-experiments/ranges/$i/startup* startup_ext.txt
	cp batch-experiments/ranges/$i/target* targets_ext.txt
	/home/rzuolo/source/Lora-PCAgent/drl-optimizer/prep-script.sh
	cd /home/rzuolo/source/Lora-PCAgent/drl-optimizer/
	cargo=`cat /home/rzuolo/source/Lora-PCAgent/drl-optimizer/targets_ext.txt | awk -F" " '{sum += $1} END {print sum*20}'`
	top_score=`python run.py test nada | grep reward | awk -F" " {'print $2'} | cut -d":" -f2 | sort -d | tail -n 1`
	top_score_int=${top_score%.*}
	let "perf=($top_score_int*100)/$cargo"
	echo "$i max: $cargo -- $i top score: $top_score_int -- performance: $perf%"

done






