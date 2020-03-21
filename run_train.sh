#python -u main_findif.py train --data="FilterNYT" --use_gpu=True > train.log &
# python -u main_findif.py train --data="NYT" --use_gpu=True > train.log &
# python -u main_mil.py train --data="FilterNYT" --use_gpu=True > train.log &
# python -u main_att.py train --data="FilterNYT" --use_gpu=True > train.log &
srun --gres=gpu:1 python -u main_findif.py train --data="NYT" --use_gpu=True --num_epochs=2 > train.log & 
# python -u main_findif.py train --data="NYT" --use_gpu=False --num_epochs=2 > train.log &

