#!/bin/bash
python exps.py --config bay.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset bay  
python exps.py --config bay.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset bay --modelfolde filename

python exps.py --config metrla.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset metrla  
python exps.py --config metrla.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset metrla --modelfolde filename

python exps.py --config pems03.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset baypems03  
python exps.py --config pems03.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset pems03 --modelfolde filename

python exps.py --config pems04.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset pems04  
python exps.py --config pems04.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset pems04 --modelfolde filename

python exps.py --config pems07.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset pems07  
python exps.py --config pems07.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset pems07 --modelfolde filename

python exps.py --config pems08.yaml --nsample 100 --testmissingratio 0.3 --seed 42 --dataset pems08  
python exps.py --config pems08.yaml --nsample 100 --testmissingratio 0.05 --seed 42 --block --dataset pems08 --modelfolde filename
