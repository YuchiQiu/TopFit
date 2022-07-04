#!/bin/bash
curl -o DemoData.tar.gz https://weilab.math.msu.edu/Downloads/TopFit/DemoData.tar.gz
tar -zxvf DemoData.tar.gz
rsync -a DemoData/* .
rm -r DemoData/ 
