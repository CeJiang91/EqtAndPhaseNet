import os
import time
import run
import glob


def mseed_mode():
    start = time.process_time()
    # args.plot_figure=True
    # args.input_length = 3000
    fname_dir = '/home/fanggrp/data/jc_private/Yangbi.phasenet_input/fnames_2429'

    files = glob.glob(r'%s/*.csv' % fname_dir)
    # for prob in list([0.1, 0.5, 0.7, 0.9]):
    for prob in list([0.3]):
        for i in range(files.__len__()):
            # for f in files:
            f = files[i]
            args = run.read_args()
            args.data_dir = '/home/fanggrp/data/jc_private/Yangbi.phasenet_input/mseed'
            args.batch_size = 40
            args.mode = 'pred'
            args.model_dir = 'model/190703-214543'
            args.input_mseed = True
            args.data_list = f
            args.output_dir = f"/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output_2429{prob}/{f.split('.')[-2]}"
            # run.main(args)
            os.system(f"python run.py --mode=pred --model_dir=model/190703-214543  --batch_size={args.batch_size}"
                      f" --input_mseed --ts_prob={prob} --tp_prob={prob} --data_dir={args.data_dir}"
                      f" --data_list={args.data_list} --output_dir={args.output_dir}")
            # break
        end = time.process_time()
        print(end - start)


def npz_mode():
    start = time.process_time()
    args = run.read_args()
    args.batch_size = 20
    # args.plot_figure=True
    # args.data_dir = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform_xfj'
    # args.data_list = '../../../work/SeismicData/XFJ1121/1daytest/days.phnetinput/waveform.csv'
    # args.input_length = 3000
    args.tp_prob=0.3
    args.ts_prob=0.3
    args.data_dir = 'demo/mseed'
    args.data_list = 'demo/fname.csv'
    args.output_dir = '/home/jc/work/data/this_can_be_delete'
    # args.output_dir = '/home/jc/work/data/Maduo/Maduo.phasenet_output_30s_0.1'
    args.mode = 'pred'
    args.model_dir = 'model/190703-214543'
    # args.input_mseed = True
    # run.main(args)
    os.system(f"python run.py --mode=pred --model_dir=model/model_new  --batch_size={args.batch_size}"
              f"  --input_mseed --ts_prob={args.ts_prob} --tp_prob={args.tp_prob} --data_dir={args.data_dir}"
              f" --data_list={args.data_list} --output_dir={args.output_dir}")
    end = time.process_time()
    print(end - start)


if __name__ == '__main__':
    # mseed_mode()
    npz_mode()
