import os
import time
import run


if __name__ == '__main__':
    start = time.process_time()
    args = run.read_args()
    args.batch_size = 500
    args.data_dir = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform_xfj'
    args.data_list = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform.csv'
    # args.data_dir = '../../../work/SeismicData/XFJ1121/phasenet_input/waveform_xfj'
    # args.data_list = '../../../work/SeismicData/XFJ1121/phasenet_input/waveform.csv'
    args.output_dir = '../../../work/SeismicData/XFJ1121/thiscanbedelete2'
    args.input_length = 6000
    args.mode = 'pred'
    args.model_dir = 'model/190703-214543'
    args.plot_figure=True
    run.main(args)
    # os.system("python run.py --mode=pred --model_dir=model/190703-214543 "
    #           "--data_dir=/home/jc/work/XFJ1121/phasenet_input "
    #           "--data_list=/home/jc/work/XFJ1121/phasenet_input/waveform.csv --output_dir=output --plot_figure "
    #           "--save_result --batch_size=500 --input_length=6000")
    end = time.process_time()
    print(end - start)