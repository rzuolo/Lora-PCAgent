#!/bin/bash

xargs -n 66 < template/startup-template > startup-template-formatted

for i in {1..20..1};
do

	cat template/targets-template > ranges/plus$i/targets_ext_round_$i.txt
	cat startup-template-formatted >  ranges/plus$i/startup_ext_round_$i.txt
	slot=`head -n $i increment.dat`
	echo "Starting slot"	
	for j in $slot
	do
	  line=`echo -n $j |cut -d"," -f1`
	  value=`echo -n $j |cut -d"," -f2`
	  let packets=$RANDOM%5+1
	  echo $packets >> ranges/plus$i/targets_ext_round_$i.txt
	  newlines=`wc -l ranges/plus$i/targets_ext_round_$i.txt | cut -d" " -f1`
	  echo "$newlines"
	  echo $line--$value
  	  cat ranges/plus$i/startup_ext_round_$i.txt | sed  "s/$/ PLACEHOLDER/" | sed  ""$line"s/PLACEHOLDER/REPLACED/" | sed  "s/PLACEHOLDER/0/"  | sed  "s/REPLACED/$value/" | tr '\n' ' ' > auxiliar 
	  xargs -n $newlines < auxiliar > ranges/plus$i/startup_ext_round_$i.txt
  	done


done

rm -f auxiliar
rm -f startup-template-formatted
