from yangbi_script import mseed_phnerr, rm_phasenet_overlap, phn_separate_combine, phasenet_stafilter_fnamecsv, \
    mseed_eqterr


def run_yangbi():
    # phn_separate_combine(dir="/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output0.1")
    # rm_phasenet_overlap(input_csv='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output0.1/picks_all.csv',
    #                     output_csv='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output0.1/'
    #                                'picks_rm_overlap.csv')
    # mseed_phnerr(input_file='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output0.3/picks_rm_overlap.csv',
    #              output_dir='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.phasenet_output0.3',
    #              catalog_file='/home/jc/work/data/Yangbi/Yangbi_result/man_catalog.npy')
    # mseed_eqterr(input_dir='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.eqt_output0.1',
    #              output_dir='/home/jc/work/data/Yangbi/Yangbi_result/Yangbi.eqt_output0.1',
    #              catalog_file='/home/jc/work/data/Yangbi/Yangbi_result/man_catalog.npy')
    phasenet_stafilter_fnamecsv(csv_dir='/home/fanggrp/data/jc_private/Yangbi.phasenet_input/')


if __name__ == '__main__':
    run_yangbi()
