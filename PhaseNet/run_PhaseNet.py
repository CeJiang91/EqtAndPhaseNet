import os

# os.system("python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=dataset/waveform_pred
# --data_list=dataset/waveform.csv --output_dir=output --plot_figure --save_result --batch_size=20")
# os.system("python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=TEST_dataset/waveform_xfj "
#           "--data_list=TEST_dataset/waveform.csv --output_dir=output --plot_figure --save_result --batch_size=20 "
#           "--input_length=6000")
os.system("python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=XFJ_dataset/waveform_xfj "
          "--data_list=XFJ_dataset/waveform.csv --output_dir=output --plot_figure --save_result --batch_size=500 "
          "--input_length=6000")
