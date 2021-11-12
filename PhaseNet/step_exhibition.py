import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time
import run2
import glob

'''
    Before running this script, you should cancel the comment in line 533 of run.py
'''

print('step_exhibition')
os.system('pwd')
start = time.process_time()
# args.plot_figure=True
# args.input_length = 3000
event_id = 'XFJ.GD_201112231459.0002_EV'
prob = 0.5
data_path = '../results_analysis/p4_eqt_process_image/example/npz'
f = os.path.join(data_path, 'waveform.csv')
args = run2.read_args()
args.data_dir = os.path.join(data_path, 'waveform_xfj')
args.batch_size = 1
args.mode = 'pred'
args.model_dir = 'model/190703-214543'
# args.input_mseed = True
args.data_list = f
args.plot_depth=5
args.output_dir = f"../results_analysis/p4_eqt_process_image/example/result{prob}/"
run2.main(args)
# os.system(f"python run.py --mode=pred --model_dir=model/190703-214543  --batch_size={args.batch_size}"
#           f" --input_mseed --ts_prob={prob} --tp_prob={prob} --data_dir={args.data_dir}"
#           f" --data_list={args.data_list} --output_dir={args.output_dir}")
# break
end = time.process_time()
print(end - start)
