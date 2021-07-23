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
    args.input_length = 3000
    args.data_dir = 'demo/mseed'
    args.data_list = 'demo/fname.csv'
    args.data_dir = '/media/jiangce/work_disk/project/SeismicData/Maduo/QH.DAR_PhaseNet/QH.DAR_PhaseNet_input/mseed'
    args.data_list = '/media/jiangce/work_disk/project/SeismicData/Maduo/QH.DAR_PhaseNet/QH.DAR_PhaseNet_input' \
                     '/fname.csv'
    args.output_dir = '/media/jiangce/work_disk/project/SeismicData/Maduo/QH.DAR_PhaseNet/QH.DAR_PhaseNet_output'
    # args.output_dir = '/media/jiangce/work_disk/project/SeismicData/test_can_be_delete'
    args.mode = 'pred'
    args.model_dir = 'model/190703-214543'
    args.input_mseed = True
    args.tp_prob = 0.5
    args.ts_prob = 0.5
    run.main(args)
    # os.system("python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=demo/mseed "
    #           "--data_list=demo/fname.csv --output_dir=output --batch_size=20 --input_mseed ")
    end = time.process_time()
    print(end - start)