#!/bin/bash


devs=`cat /home/rzuolo/source/Lora-PCAgent/drl-optimizer/targets_ext.txt | wc -l |cut -d" " -f1`
cargo=`cat /home/rzuolo/source/Lora-PCAgent/drl-optimizer/targets_ext.txt | awk -F" " '{sum += $1} END {print sum*10}'`
echo -n $devs > /home/rzuolo/source/Lora-PCAgent/gym/envs/classic_control/number_of_nodes.txt
echo $cargo > /home/rzuolo/source/Lora-PCAgent/gym/envs/classic_control/cargo_load.txt
echo -n $devs > /home/rzuolo/source/Lora-PCAgent/drl-optimizer/number_of_nodes.txt
echo $cargo > /home/rzuolo/source/Lora-PCAgent/drl-optimizer/cargo_load.txt

