# This script is used to process the Maduo 521 earthquake
from yangbi_script import phn_continue_sac2mseed, mseed_phasenet_MT, \
    eqt_continue_sac2mseed,gmt_sta_location,gmt_eq_location,man2catalog_npy,mseed_phnerr,\
    mseed_eqterr,err_hist,rm_phasenet_overlap
from EQTransformer.core.mseed_predictor import mseed_predictor


def run_eqt_mseed_predictor():
    mseed_predictor(input_dir='/media/jiangce/Elements SE/Maduo/Maduo.eqt_input/mseeds',
                    input_model='./EQTransformer/data/EqT_model.h5',
                    stations_json='/media/jiangce/Elements SE/Maduo/Maduo.eqt_input/station_list.json',
                    output_dir='/media/jiangce/work_disk/project/SeismicData/Maduo/Maduo_result/Maduo.eqt_output',
                    detection_threshold=0.2,
                    P_threshold=0.1,
                    S_threshold=0.1,
                    number_of_plots=0,
                    plot_mode='time_frequency',
                    batch_size=500,
                    overlap=0.3)


if __name__ == '__main__':
    # phn_continue_sac2mseed(input_dir='/media/jiangce/Elements SE/Maduo/Maduo_sac',
    #                        output_dir='/media/jiangce/Elements SE/Maduo/QH.DAR_PhaseNet/QH.DAR_PhaseNet_input',
    #                        station_filter_file='/media/jiangce/Elements SE/Maduo/sta_filter_list.csv')
    # eqt_continue_sac2mseed(input_dir='/media/jiangce/Elements SE/Maduo/Maduo_sac',
    #                        output_dir='/media/jiangce/Elements SE/Maduo/Maduo.eqt_input',
    #                        station_filter_file='/media/jiangce/Elements SE/Maduo/sta_filter_list.csv')
    # mseed_phasenet_MT(input_file='/home/jc/work/data/Maduo/QH.MAD_PhaseNet/QH.MAD_PhaseNet_output/picks_rm_error.csv',
    #                   output_image='/home/jc/work/data/Maduo/QH.MAD_PhaseNet/mt_dar.pdf')
    # run_mseed_predictor()
    # gmt_sta_location(sac_dir='/media/jiangce/Elements SE/Maduo/Maduo_sac/2021050100.QH.SAC',
    #                  stfile='/media/jiangce/work_disk/project/EqtAndPhaseNet/results_analysis/gmt/st_maduo.dat',
    #                  station_filter_file='/media/jiangce/Elements SE/Maduo/sta_filter_list.csv')
    # man2catalog_npy(input_file='/media/jiangce/Elements SE/Maduo/QH_phase_2021-2021.txt',
    #                 output_file='/media/jiangce/Elements SE/Maduo/man_catalog.npy')
    # gmt_eq_location(catalog_file='/media/jiangce/Elements SE/Maduo/man_catalog.npy',
    #                 eqfile='/media/jiangce/work_disk/project/EqtAndPhaseNet/results_analysis/gmt/eq_maduo.dat')
    # mseed_phnerr(input_file='/home/jc/work/data/Maduo/Maduo_result/Maduo.phasenet_output_30s_0.3/'
    #                         'picks_rm_overlap.csv',
    #              output_file='/home/jc/work/data/Maduo/Maduo_result/Maduo.phasenet_output_30s_0.3/phnerr_30s_0.3.npy',
    #              catalog_file='/home/jc/work/data/Maduo/man_catalog.npy')
    mseed_eqterr(input_dir='/home/jc/work/data/Maduo/Maduo_result/Maduo.eqt_output',
                 output_file='/home/jc/work/data/Maduo/Maduo_result/eqterr.npy',
                 catalog_file='/home/jc/work/data/Maduo/man_catalog.npy')
    # err_hist(eqt_file='/home/jc/work/data/Maduo/Maduo_result/eqterr.npy',
    #          phn_file='/home/jc/work/data/Maduo/Maduo_result/phnerr_30s_0.3.npy',
    #          output_dir='/home/jc/work/EqtAndPhaseNet/results_analysis/p2')
    # rm_phasenet_overlap(input_csv='/home/jc/work/data/Maduo/Maduo_result/Maduo.phasenet_output_30s_0.3/picks.csv',
    #                     output_csv='/home/jc/work/data/Maduo/Maduo_result/Maduo.phasenet_output_30s_0.3/'
    #                                'picks_rm_overlap.csv')