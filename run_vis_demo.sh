# calculate inverse HVP with the trained model
srun --gres=gpu:1 python -u tools/cal_inv_hvp.py go --use_gpu=True

# get influence of selected relation
python -u tools/visualize_demo.py go --rel=1 --use_gpu=False

