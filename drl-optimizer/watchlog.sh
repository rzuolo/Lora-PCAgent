#!/bin/bash

file=`ls -r lora_log* | tail -n 1`

tail -f $file | grep Time
