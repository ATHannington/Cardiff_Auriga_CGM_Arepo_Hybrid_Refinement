#!/bin/bash

declare -a Directories=("./ICs")
#("./Arepo_GFM_Tables_NewAuriga" "./AgeOfAquariusOutputs")

for dir in ${Directories[@]}
  do
    for file in $(find $dir -type f -print0| xargs -0 -L1)
      do
	echo ""
	echo $file
	echo $(lfs getstripe $file)
	tmpfile=$(mktemp)
	cp $file $tmpfile
	rm -rf $file
	mv $tmpfile $file
	echo $(lfs getstripe $file)
      done
  done
