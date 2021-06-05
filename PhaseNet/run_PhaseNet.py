import os
import time
import run


if __name__ == '__main__':
    start = time.process_time()
    args = run.read_args()
    args.batch_size = 20
    # args.plot_figure=True
    # args.data_dir = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform_xfj'
    # args.data_list = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform.csv'
    # args.input_length = 6000
    args.data_dir = '/home/jc/disk1/Yangbi.phasenet_input/mseed_xfj'
    args.data_list = '/home/jc/disk1/Yangbi.phasenet_input/fname.csv'
    args.output_dir = '/home/jc/disk1/Yangbi.phasenet_output'
    args.mode = 'pred'
    args.model_dir = 'model/190703-214543'
    args.input_mseed = True
    run.main(args)
    # os.system("python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=demo/mseed "
    #           "--data_list=demo/fname.csv --output_dir=output --batch_size=20 --input_mseed ")
    end = time.process_time()
    print(end - start)