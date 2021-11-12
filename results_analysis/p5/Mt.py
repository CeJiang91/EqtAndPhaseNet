from yangbi_script import mseed_phasenet_MT

if __name__ == '__main__':
    mseed_phasenet_MT(input_file='/home/jc/work/data/Yangbi/XG.CHT_PhaseNet/XG.CHT_PhaseNet_output/picks.csv',
                      output_image='mt_cht_test.pdf')
    # mseed_phasenet_MT(input_file='/home/jc/work/data/Maduo/QH.MAD_PhaseNet/QH.MAD_PhaseNet_output/picks_rm_error.csv',
    #                   output_image='mt_dar.pdf')