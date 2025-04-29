#!/bin/bash

for item in 200
do
    echo "Running for item=$item"
    python tools/draw_t-SNE.py -item $item
done
