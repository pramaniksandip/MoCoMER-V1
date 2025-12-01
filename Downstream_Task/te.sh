#!/bin/bash

n=4
for (( ; ; )); 
do
    if [ $n -eq 9 ]; then
        break
    fi
    echo $n
    ((n=n+1))
done
