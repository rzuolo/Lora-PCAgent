#!/bin/bash

IFS="
"
roundtrip=`ls batch-experiments/ranges/|grep plus`

for i in $roundtrip
do
	cp batch-experiments/ranges/$i/startup* startup_ext.txt
	cp batch-experiments/ranges/$i/target* targets_ext.txt
	/home/rzuolo/source/Lora-PCAgent/drl-optimizer/prep-script.sh
	cd /home/rzuolo/source/Lora-PCAgent/drl-optimizer/
	pwd 
	python run.py test nada

done






