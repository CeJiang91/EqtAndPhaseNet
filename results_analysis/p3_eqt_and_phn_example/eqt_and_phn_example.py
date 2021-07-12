from yangbi_script import find_cases_mseed_output

if __name__ == '__main__':
    find_cases_mseed_output(eqt_output='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.eqt_output',
                            phn_picks='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.phasenet_output_all/'
                                      'picks.csv',
                            catalog_file='/home/jiangce/work/SeismicData/Yangbi_result/man_catalog.npy')