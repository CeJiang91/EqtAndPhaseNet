# This script is used to process the Yangbi 521 earthquake
import glob
import os
import shutil
import numpy as np
from obspy import UTCDateTime, read, Stream
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib as mpl


# mpl.rcParams['pdf.fonttype']=42
# mpl.rcParams['text.usetex']=True


def to_percent(y, position):
    return str(100 * y) + "%"


def gmt_sta_location(sac_dir, stfile, station_filter_file=None):
    stfile = open(stfile, 'w')
    if station_filter_file:
        netstalist = pd.read_csv(station_filter_file, delimiter=',', header=None)
        sta_list = list(netstalist.iloc[:, 1])
        net_list = list(netstalist.iloc[:, 0])
    files = glob.glob(r'%s/*.SAC' % sac_dir)
    net_sta_list = set([])
    for f in files:
        net = f.split('/')[-1].split('.')[6]
        sta = f.split('/')[-1].split('.')[7]
        net_sta_list.add(f'{net}.{sta}')
    net_sta_list = np.array(net_sta_list)
    stnm = []
    for netsta in net_sta_list.item():
        net = netsta.split('.')[0]
        sta = netsta.split('.')[1]
        if station_filter_file:
            if (net not in net_list) or (sta not in sta_list):
                continue
        st = read(f'{sac_dir}/*{netsta}*.SAC')
        st.merge()
        stlo = st[0].stats.sac['stlo']
        stla = st[0].stats.sac['stla']
        stfile.write('%10s' % netsta + '%11.3f' % stla + '%8.3f' % stlo + '\n')
        stnm.append(st)
    stfile.close()


def gmt_eq_location(catalog_file, eqfile):
    catalog = np.load(catalog_file, allow_pickle=True).item()
    eqfile = open(eqfile, 'w')
    for evn in catalog['head']:
        eqlon = catalog['head'][evn]['lon']
        eqlat = catalog['head'][evn]['lat']
        eqmag = catalog['head'][evn]['ML']
        eqfile.write(evn + '%8.3f' % eqlat + '%8.3f' % eqlon + '%8.2f' % eqmag + '\n')
    eqfile.close()


def man2catalog_npy(input_file, output_file):
    fp = open(input_file, encoding='utf-8')
    lines = fp.readlines()
    phase = {}
    evn = 'None'
    mag = {}
    dist = {}
    head = {}
    for line in lines:
        if line[0] == '#':
            words = line.split()
            date = UTCDateTime(line[2:24])
            evn = date.strftime('%Y%m%d.%H%M%S.%f')[:-4] + '.SAC'
            if not phase.__contains__(evn):
                phase[evn] = {}
                mag[evn] = words[10]
                dist[evn] = {}
                head[evn] = {}
                head[evn]['lat'] = float(words[7])
                head[evn]['lon'] = float(words[8])
                head[evn]['starttime'] = UTCDateTime(line[2:24])
                head[evn]['ML'] = float(words[10])
                head[evn]['depth'] = float(words[9])
        else:
            words = line.split()
            st = words[0][-2:] + '/' + words[0][0:-2]
            phase_type = words[3][0]
            arrival = UTCDateTime(head[evn]['starttime'].strftime('%Y-%m-%d') + ' ' + words[5])
            if not phase[evn].__contains__(st):
                phase[evn][st] = {}
                dist[evn][st] = words[1]
            if phase[evn][st].__contains__(phase_type):
                phase[evn][st][phase_type] = min(arrival, phase[evn][st][phase_type])
            else:
                phase[evn][st][phase_type] = arrival
    catalog = {'head': head, 'phase': phase, 'dist': dist, 'mag': mag}
    np.save(output_file, catalog)


def phn_continue_sac2mseed(input_dir, output_dir, station_filter_file=None):
    if station_filter_file:
        netstalist = pd.read_csv(station_filter_file, delimiter=',', header=None)
        sta_list = list(netstalist.iloc[:, 1])
        net_list = list(netstalist.iloc[:, 0])
    if os.path.isdir(output_dir):
        print('============================================================================')
        print(f' *** {output_dir} already exists!')
        inp = input(" --> Type (Yes or y) to create a new empty directory! otherwise it will overwrite!   ")
        if inp.lower() == "yes" or inp.lower() == "y":
            shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'mseed'))
    trn = 0
    csv_file = open(os.path.join(output_dir, "fname.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'E', 'N', 'Z'])
    for root, dirs, files in os.walk(input_dir, topdown=True):
        dirs.sort()
        if len(files) != 0:
            st = Stream()
            datastr = root.split('/')[-1].split('.')[0]
            utc_start = UTCDateTime(datastr) - 8 * 3600
            # if (utc_start < UTCDateTime('2021/05/22/') - 8 * 3600) or \
            #         (utc_start >= UTCDateTime('2021/05/27/') - 8 * 3600):
            #     continue
        for f in sorted(files):
            net = f.split('.')[6]
            sta = f.split('.')[7]
            if net != 'QH' or sta != 'DAR':
                continue
            if station_filter_file:
                if net not in net_list or sta not in sta_list:
                    continue
            tr = read(os.path.join(root, f))
            st.append(tr[0])
        if len(files) != 0:
            nets = []
            stas = []
            st.merge()
            st.trim(utc_start, utc_start + 3600, pad=True, fill_value=0)
            for i in range(len(st)):
                net = st[i].stats.network
                sta = st[i].stats.station
                receiver_type = st[i].stats.channel[:-1]
                if net in nets and sta in stas:
                    continue
                else:
                    nets.append(net)
                    stas.append(sta)
                    fname = datastr + f'.{net}.{sta}.mseed'
                    now_st = st.select(network=net, station=sta)
                    now_st.write(os.path.join(output_dir, 'mseed', fname), format='MSEED')
                    output_writer.writerow([fname, receiver_type + 'E', receiver_type + 'N', receiver_type + 'Z'])
                    csv_file.flush()
            # break


def eqt_continue_sac2mseed(input_dir, output_dir, station_filter_file=None):
    import json
    net_sta_file = 'eqt.net_sta.npy'
    if station_filter_file:
        netstalist = pd.read_csv(station_filter_file, delimiter=',', header=None)
        sta_list = list(netstalist.iloc[:, 1])
        net_list = list(netstalist.iloc[:, 0])
    if not os.path.isfile(net_sta_file):
        files = glob.glob(r'%s/*/*.SAC' % input_dir)
        net_sta_list = set([])
        for f in files:
            net = f.split('/')[-1].split('.')[6]
            sta = f.split('/')[-1].split('.')[7]
            net_sta_list.add(f'{net}.{sta}')
        net_sta_list = np.array(net_sta_list)
        np.save(net_sta_file, net_sta_list, allow_pickle=True)
    else:
        net_sta_list = np.load(net_sta_file, allow_pickle=True).item()
    dic = {}
    for netsta in net_sta_list:
        # netsta = net_sta_list[0]
        net = netsta.split('.')[0]
        sta = netsta.split('.')[1]
        if station_filter_file:
            if (net not in net_list) or (sta not in sta_list):
                continue
        start_date = UTCDateTime('20210522')
        for i in range(5):
            start_time = start_date - 8 * 3600
            start_date_str = start_date.strftime('%Y%m%d')
            if start_date_str == '20210531':
                breakpoint()
            # files = glob.glob(f'{input_dir}/{start_date_str}*/*{netsta}*.SAC')
            st = read(f'{input_dir}/{start_date_str}*/*{netsta}*.SAC')
            st.merge()
            st.trim(start_time, start_time + 24 * 3600, pad=True, fill_value=0)
            filepath = os.path.join(output_dir, 'mseeds', f'{sta}')
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            start_date_str = start_date.strftime('%Y%m%dT%H%M%SZ')
            end_date_str = (start_date + 24 * 3600).strftime('%Y%m%dT%H%M%SZ')
            format = f'{netsta}..{st[0].stats.channel}__{start_date_str}__{end_date_str}.mseed'
            st[0].write(os.path.join(filepath, format), format='MSEED')
            format = f'{netsta}..{st[1].stats.channel}__{start_date_str}__{end_date_str}.mseed'
            st[1].write(os.path.join(filepath, format), format='MSEED')
            format = f'{netsta}..{st[2].stats.channel}__{start_date_str}__{end_date_str}.mseed'
            st[2].write(os.path.join(filepath, format), format='MSEED')
            dic[sta] = {}
            dic[sta]["network"] = st[0].stats.network
            dic[sta]["channels"] = [st[0].stats.channel, st[1].stats.channel, st[2].stats.channel]
            dic[sta]["coords"] = [float(st[0].stats.sac['stla']),
                                  float(st[0].stats.sac['stlo']),
                                  float(st[0].stats.sac['stel'])]
            with open(os.path.join(output_dir, 'station_list.json'), 'w+') as jsonFile:
                jsonFile.write(json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':')))
            start_date += 24 * 3600
        # breakpoint()


def phn_mseed_pick(input_dir, output_dir, station_file):
    files = glob.glob(r'%s/*.mseed' % input_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    csv_file = open(os.path.join(output_dir, '..', "pick_mseed.csv"), 'w', newline='')
    output_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['fname', 'E', 'N', 'Z'])
    netstalist = pd.read_csv(station_file, delimiter=',', header=None)
    sta_list = list(netstalist.iloc[:, 1])
    net_list = list(netstalist.iloc[:, 0])
    for f in files:
        net = f.split('/')[-1].split('.')[1]
        sta = f.split('/')[-1].split('.')[2]
        utc = UTCDateTime(f.split('/')[-1].split('.')[0])
        fname = f.split('/')[-1]
        if net == 'XG' and sta == 'CHT':
            # if (net in net_list and sta in sta_list) and utc < UTCDateTime('20210524'):
            # if (net in net_list and sta in sta_list) and UTCDateTime('20210524')<utc<UTCDateTime('20210528'):
            os.system(f'cp {f} {output_dir}')
            output_writer.writerow([fname, 'BHE', 'BHN', 'BHZ'])
            csv_file.flush()
    csv_file.close()


def mseed_phnerr(input_file, output_file, catalog_file):
    catalog = np.load(catalog_file, allow_pickle=True).item()
    df = pd.read_csv(input_file)
    csv_file = './qh_man_phase.csv'
    if os.path.isfile(csv_file):
        df_cata = pd.read_csv(csv_file)
    else:
        df_cata = pd.DataFrame(columns=['station', 'itp', 'its'])
        for evn in catalog['phase']:
            for sta in catalog['phase'][evn]:
                if 'P' in catalog['phase'][evn][sta]:
                    itp = catalog['phase'][evn][sta]['P']
                else:
                    itp = -999
                if 'S' in catalog['phase'][evn][sta]:
                    its = catalog['phase'][evn][sta]['S']
                else:
                    its = -999
                df_cata = df_cata.append([{'station': sta, 'itp': itp, 'its': its}])
        df_cata.to_csv(csv_file)
    # if os.path.isfile(output_file):
    #     phnerr = np.load(output_file, allow_pickle=True).item()
    #     breakpoint()
    # else:
    phnerr = {'P': [], 'S': []}
    sta_list = []
    TP = 0
    FP = 0
    FN = 0
    label_df = pd.DataFrame()
    for i in range(df.shape[0]):
        sta_list.append(df['fname'][i].split('.')[1].strip() + '/' + df['fname'][i].split('.')[2].strip())
    sta_list = list(set(sta_list))
    for sta in sta_list:
        time_range = UTCDateTime('2021-05-22')
        for i in range(7):
            label_df = label_df.append(df_cata.loc[(df_cata.station == sta) &
                                                   (df_cata.itp.str.contains(time_range.__str__()[0:10]))])
            time_range += 3600 * 24
            label_df.index = range(len(label_df))
    itp_list = []
    its_list = []
    for i in range(df.shape[0]):
        if 0.01 * int(df['fname'][i].split('_')[1]) >= 3600:
            utc_phn = UTCDateTime(df['fname'][i].split('.')[0]) + 0.01 * int(df['fname'][i].split('_')[1]) - 3600 - 15
        else:
            utc_phn = UTCDateTime(df['fname'][i].split('.')[0]) + 0.01 * int(df['fname'][i].split('_')[1])
        sta = df['fname'][i].split('.')[1].strip() + '/' + df['fname'][i].split('.')[2].strip()
        itp = list(map(eval, df['itp'][i].strip('[]').split()))
        its = list(map(eval, df['its'][i].strip('[]').split()))
        FP += len(itp)
        FP += len(its)
        if len(itp) > 0:
            utc_itp = [utc_phn + pt * 0.01 for pt in itp]
        else:
            utc_itp = []
        if len(its) > 0:
            utc_its = [utc_phn + st * 0.01 for st in its]
        else:
            utc_its = []
        selected = df_cata.loc[(df_cata.station == sta) & (df_cata.itp.str.contains(utc_phn.__str__()[0:13]))]
        selected.index = range(len(selected))
        breakrange = False
        continuerange = False
        for j in range(selected.shape[0]):
            if selected.itp[j] != '-999':
                for pt in utc_itp:
                    phnet_erral = pt - UTCDateTime(selected['itp'][j])
                    if abs(phnet_erral) < 0.5:
                        if selected.itp[j] not in itp_list:
                            phnerr['P'].append(phnet_erral)
                            TP += 1
                            FP -= 1
                            itp_list.append(selected['itp'][j])
                        else:
                            FP -= 1
                    if phnet_erral < -5:
                        breakrange = True
                        break
                    if phnet_erral > 5:
                        continuerange = True
                        continue
            if selected['its'][j] != '-999' :
                for st in utc_its:
                    phnet_erral = st - UTCDateTime(selected['its'][j])
                    if abs(phnet_erral) < 0.5:
                        if selected.its[j] not in its_list:
                            phnerr['S'].append(phnet_erral)
                            TP += 1
                            FP -= 1
                            its_list.append(selected['its'][j])
                        else:
                            FP -=1
                    if phnet_erral < -5:
                        breakrange = True
                        break
                    if phnet_erral > 5:
                        continuerange = True
                        continue
            if breakrange:
                break
            if continuerange:
                continue
        # if i>3000:
        #     break
    label_P_num = label_df.loc[label_df.itp != '-999'].shape[0]
    label_S_num = label_df.loc[label_df.its != '-999'].shape[0]
    true_label_num = label_S_num + label_P_num
    with open('phasenet_report.txt', 'a') as the_file:
        the_file.write('================== PhaseNet Info ==============================' + '\n')
        the_file.write('the number of PhaseNet input: ' + str(true_label_num) + '\n')
        the_file.write('P recall of PhaseNet: ' + str(TP / (true_label_num)) + '\n')
        the_file.write('P precision of PhaseNet: ' + str(TP / (TP + FP)) + '\n')
    np.save(output_file, phnerr)


def mseed_eqterr(input_dir, output_file, catalog_file):
    files = glob.glob(r'%s/*/*.csv' % input_dir)
    csv_file = './yn_man_phase.csv'
    if os.path.isfile(csv_file):
        df_cata = pd.read_csv(csv_file)
    else:
        catalog = np.load(catalog_file, allow_pickle=True).item()
        df_cata = pd.DataFrame(columns=['station', 'itp', 'its'])
        for evn in catalog['phase']:
            for sta in catalog['phase'][evn]:
                if 'P' in catalog['phase'][evn][sta]:
                    itp = catalog['phase'][evn][sta]['P']
                else:
                    itp = -999
                if 'S' in catalog['phase'][evn][sta]:
                    its = catalog['phase'][evn][sta]['S']
                else:
                    its = -999
                df_cata = df_cata.append([{'station': sta, 'itp': itp, 'its': its}])
        df_cata.to_csv(csv_file)
    eqterr = {'P': [], 'S': []}
    TP = 0
    FP = 0
    FN = 0
    label_df = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f)
        sta = df.network[0].strip() + '/' + df.station[0].strip()
        time_range = UTCDateTime('2021-05-21')
        for i in range(7):
            label_df = label_df.append(df_cata.loc[(df_cata.station == sta) &
                                                   (df_cata.itp.str.contains(time_range.__str__()[0:10]))])
            time_range += 3600 * 24
            label_df.index = range(len(label_df))
        for i in range(df.shape[0]):
            matched_p = False
            matched_s = False
            if isinstance(df.p_arrival_time[i], str):
                utc_itp = UTCDateTime(df.p_arrival_time[i])
                exist_p = True
            else:
                exist_p = False
            if isinstance(df.s_arrival_time[i], str):
                utc_its = UTCDateTime(df.s_arrival_time[i])
                exist_s = True
            else:
                exist_s = False
            utc_eqt = UTCDateTime(df.event_start_time[i]) + 8 * 3600
            sta = df.network[i].strip() + '/' + df.station[i].strip()
            selected = df_cata.loc[(df_cata.station == sta) & (df_cata.itp.str.contains(utc_eqt.__str__()[0:13]))]
            selected.index = range(len(selected))
            for j in range(selected.shape[0]):
                if selected['itp'][j] != '-999':
                    eqt_erral = utc_itp - UTCDateTime(selected['itp'][j]) + 8 * 3600
                    if abs(eqt_erral) < 0.5:
                        eqterr['P'].append(eqt_erral)
                        TP += 1
                        matched_p = True
                    if eqt_erral < -30:
                        break
                    if eqt_erral > 30:
                        continue
                if selected['its'][j] != '-999':
                    eqt_erral = utc_its - UTCDateTime(selected['its'][j]) + 8 * 3600
                    if abs(eqt_erral) < 0.5:
                        eqterr['S'].append(eqt_erral)
                        TP += 1
                        matched_s = True
                    if eqt_erral < -30:
                        break
                    if eqt_erral > 30:
                        continue
            if (not matched_p) and exist_p:
                FP += 1
            if (not matched_s) and exist_s:
                FP += 1
    # breakpoint()
    label_P_num = label_df.loc[label_df.itp != '-999'].shape[0]
    label_S_num = label_df.loc[label_df.its != '-999'].shape[0]
    true_label_num = label_S_num + label_P_num
    with open('eqt_report.txt', 'a') as the_file:
        the_file.write('================== EQTransformer Info ==============================' + '\n')
        the_file.write('the number of EQTransformer input: ' + str(true_label_num) + '\n')
        the_file.write('P recall of EQTransformer: ' + str(TP / (true_label_num)) + '\n')
        the_file.write('P precision of EQTransformer: ' + str(TP / (TP + FP)) + '\n')
    np.save(output_file, eqterr)


def mseed_phasenet_MT(input_file, output_image):
    df = pd.read_csv(input_file)
    df_count = pd.DataFrame(columns=['date', 'counts'], dtype=str)
    date_list = []
    # for i in range(10000):
    for i in range(df.shape[0]):
        date = df['fname'][i].split('.')[0][:-2]
        itp = list(map(eval, df['itp'][i].strip('[]').split()))
        its = list(map(eval, df['its'][i].strip('[]').split()))
        if date not in df_count.date.to_list():
            date_list.append(UTCDateTime(date).datetime)
            counts = len(itp)
            counts = counts + len(itp)
            df_count = df_count.append([{'date': date, 'counts': counts}])
        else:
            df_count.counts[df_count.date == date] += (len(itp) + len(its))
    plt.vlines(date_list, 0, df_count.counts.to_numpy(dtype=int), colors='black')
    plt.xticks(rotation=70)
    plt.xlabel('Day')
    plt.ylabel('Counts')
    plt.savefig(output_image)
    plt.close()


def err_hist(eqt_file, phn_file, output_dir=None):
    phnerr = np.load(phn_file, allow_pickle=True).item()
    eqterr = np.load(eqt_file, allow_pickle=True).item()
    num_bins = 51
    range = [-0.5, 0.5]
    plt.hist(phnerr['P'], num_bins, weights=[1. / len(phnerr['P'])] * len(phnerr['P']), edgecolor='red', linewidth=1,
             facecolor='red', range=range, alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['P'], num_bins, weights=[1. / len(eqterr['P'])] * len(eqterr['P']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=range, alpha=0.3, label='EQTransformer')
    plt.xlim(range)
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('${T_{AI}}$ - ${T_{Catalog}}$', fontdict={'family': 'Nimbus Roman',
                                                         'weight': 'normal', 'size': 15})
    plt.ylabel('Percentage',fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    plt.title(r'P Picks', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'P_Pick.pdf'))
    else:
        plt.savefig('P_Pick.pdf')
    plt.close()
    # S pick image
    num_bins = 51
    plt.hist(phnerr['S'], num_bins, weights=[1. / len(phnerr['S'])] * len(phnerr['S']), edgecolor='red', linewidth=1,
             facecolor='red', range=range, alpha=0.3, label='PhaseNet')
    plt.hist(eqterr['S'], num_bins, weights=[1. / len(eqterr['S'])] * len(eqterr['S']), edgecolor='blue', linewidth=1,
             facecolor='blue', range=range, alpha=0.3, label='EQTransformer')
    plt.xlim(range)
    formatter = FuncFormatter(to_percent)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('${T_{AI}}$ - ${T_{Catalog}}$', fontdict={'family': 'Nimbus Roman',
                                                         'weight': 'normal', 'size': 15})
    plt.ylabel('Percentage',fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    plt.title(r'S Picks', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'S_Pick.pdf'))
    else:
        plt.savefig('S_Pick.pdf')
    plt.close()


def find_cases_mseed_output(eqt_output, phn_picks, catalog_file):
    """
        Every time you use this function,please modify the condition in the code,  I got no motivation to optimize the
        input parameter.
        eg.
    """
    catalog = np.load(catalog_file, allow_pickle=True).item()
    csv_file = './yn_man_phase.csv'
    if os.path.isfile(csv_file):
        df_cata = pd.read_csv(csv_file)
    else:
        df_cata = pd.DataFrame(columns=['station', 'itp', 'its'])
        for evn in catalog['phase']:
            for sta in catalog['phase'][evn]:
                if 'P' in catalog['phase'][evn][sta]:
                    itp = catalog['phase'][evn][sta]['P']
                else:
                    itp = -999
                if 'S' in catalog['phase'][evn][sta]:
                    its = catalog['phase'][evn][sta]['S']
                else:
                    its = -999
                df_cata = df_cata.append([{'station': sta, 'itp': itp, 'its': its}])
        df_cata.to_csv(csv_file)
    df_phn = pd.read_csv(phn_picks)
    # condition
    sta = 'CHT'
    net = 'XG'
    length = 3000
    df_cata = df_cata.loc[(df_cata.station == f'{net}/{sta}')]
    df_cata.index = range(len(df_cata))
    df_phn = df_phn.loc[(df_phn.fname.str.contains(f'{net}.{sta}'))]
    df_phn.index = range(len(df_phn))
    df_eqt = pd.read_csv(os.path.join(eqt_output, f'{sta}_outputs', 'X_prediction_results.csv'))
    df_eqt.index = range(len(df_eqt))
    fignum = 0
    # 76 1629 1646
    for i in [76, 1629, 1646]:
        # for i in range(len(df_cata)):
        phn_get = False
        eqt_get = False
        if df_cata.itp[i] == '-999':
            continue
        itp = UTCDateTime(df_cata.itp[i])
        if df_cata.its[i] != '-999':
            its = UTCDateTime(df_cata.its[i])
        else:
            its = 0
        itp_eqt_format = itp.strftime('%Y-%m-%d %H:%M')
        df_eqt = df_eqt.fillna('-12345')
        eqt_match = df_eqt.loc[df_eqt.p_arrival_time.str.contains(itp_eqt_format)]
        eqt_match.index = range(len(eqt_match))
        for j in range(len(eqt_match)):
            utc_itp = UTCDateTime(eqt_match.p_arrival_time[j])
            eqt_erral = utc_itp - itp
            if abs(eqt_erral) < 0.5:
                eqt_get = True
                eqt_pt = utc_itp
        itp_phn_format = itp.strftime('%Y%m%d%H')
        start_sample1 = int(100 * (itp.minute * 60 + itp.second + itp.microsecond * 0.000001) // length) * length
        start_sample2 = int(
            100 * (itp.minute * 60 + itp.second + itp.microsecond * 0.000001 + 3600 + length / 200) // length) * length
        start_sample1 = str(start_sample1)
        start_sample2 = str(start_sample2)
        df_phn1 = df_phn.loc[df_phn.fname.str.contains(f'{itp_phn_format}.{net}.{sta}.mseed_{start_sample1}')]
        df_phn2 = df_phn.loc[df_phn.fname.str.contains(f'{itp_phn_format}.{net}.{sta}.mseed_{start_sample2}')]
        df_phn1.index = range(len(df_phn1))
        df_phn2.index = range(len(df_phn2))
        for j in range(len(df_phn1)):
            utc_phn1 = UTCDateTime(df_phn1['fname'][j].split('.')[0]) + 0.01 * int(df_phn1['fname'][j].split('_')[1])
            phn_itp = list(map(eval, df_phn1['itp'][j].strip('[]').split()))
            utc_itp = [utc_phn1 + pt * 0.01 for pt in phn_itp]
            for pt in utc_itp:
                phnet_erral = pt - itp
                if abs(phnet_erral) < 0.5:
                    phn_get = True
                    phn_pt = pt
        for j in range(len(df_phn2)):
            utc_phn2 = UTCDateTime(df_phn2['fname'][j].split('.')[0]) + 0.01 * int(
                df_phn2['fname'][j].split('_')[1]) - 3600 - length / 200
            phn_itp = list(map(eval, df_phn2['itp'][j].strip('[]').split()))
            utc_itp = [utc_phn2 + pt * 0.01 for pt in phn_itp]
            for pt in utc_itp:
                phnet_erral = pt - itp
                if abs(phnet_erral) < 0.5:
                    phn_get = True
                    phn_pt = pt
        if eqt_get and not phn_get:
            phasenet_input_dir = '/media/jiangce/Elements SE/Yangbi/Yangbi.phasenet_input/mseed'
            st = read(os.path.join(phasenet_input_dir, f'{itp_phn_format}.{net}.{sta}.mseed'))
            start_time = itp - 8 * 3600 - 5
            end_time = itp - 8 * 3600 + 10
            itp_0 = itp - 8 * 3600
            st = st.trim(start_time, end_time, pad=True, fill_value=0)
            st.detrend()
            st.normalize()
            fig = plt.figure(figsize=(8, 4), dpi=100)
            fignum += 1
            if fignum > 3:
                break
            for k in range(3):
                ax = fig.add_subplot(3, 1, k + 1)
                ax.plot(st[k].times(), st[k].data, 'k')
                ax.vlines((itp_0 - start_time), -1, 1, 'b', linewidth=2.0)
                if its > 0:
                    its_0 = its - 8 * 3600
                    ax.vlines((its_0 - start_time), -1, 1, 'r', linewidth=2.0, linestyles='dashed')
                ax.yaxis.set_visible(False)
                ax.set_xlim(0, 12)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                if k == 2:
                    ax.set_xlabel('t/s')
            fig.tight_layout()
            print(i)
            # breakpoint()
            fig.savefig(f'eqt1_phn0{i}.pdf')


if __name__ == '__main__':
    # gmt_sta_location(sac_dir='/media/jiangce/Elements SE/Yangbi_sac/2021050100.YN.SAC',
    #                  stfile='/media/jiangce/work_disk/project/EqtAndPhaseNet/results_analysis/gmt/st_yangbi.dat',
    #                  station_filter_file='/media/jiangce/Elements SE/Yangbi.phasenet_input/sta_filter_list.csv')
    # gmt_eq_location(catalog_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/man_catalog.npy',
    #                 eqfile='/media/jiangce/work_disk/project/EqtAndPhaseNet/results_analysis/gmt/eq_yangbi.dat')
    # phn_continue_sac2mseed(input_dir='/media/jiangce/Elements SE/Yangbi/Yangbi_sac',
    #                        output_dir='/media/jiangce/work_disk/project/SeismicData/Yangbi/XG.XBT_PhaseNet/XG.XBT_PhaseNet_input')
    # mseed_phasenet_MT(input_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/'
    #                              'XG.XBT_PhaseNet/XG.XBT_PhaseNet_output/picks.csv',
    #                   output_image='/media/jiangce/work_disk/project/SeismicData/Yangbi/XG.XBT_PhaseNet/mt_xbt.png')
    mseed_phnerr(input_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/Yangbi.phasenet_output_all/picks.csv',
                 output_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/phnerr.npy',
                 catalog_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/man_catalog.npy')
    # mseed_eqterr(input_dir='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/Yangbi.eqt_output',
    #              output_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/eqterr.npy',
    #              catalog_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/man_catalog.npy')
    # err_hist(eqt_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/eqterr.npy',
    #          phn_file='/media/jiangce/work_disk/project/SeismicData/Yangbi/Yangbi_result/phnerr.npy',
    #          output_dir='/media/jiangce/work_disk/project/EqtAndPhaseNet/results_analysis/p2_pick_error')
    # find_cases_mseed_output(eqt_output='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.eqt_output',
    #                         phn_picks='/home/jiangce/work/SeismicData/Yangbi_result/Yangbi.phasenet_output_all/'
    #                                   'picks.csv',
    #                         catalog_file='/home/jiangce/work/SeismicData/Yangbi_result/man_catalog.npy')
    # man2catalog_npy(input_file='/media/jiangce/work_disk/project/SeismicData/Maduo/QH_phase_2021-2021.txt',
    #                 output_file='/media/jiangce/work_disk/project/SeismicData/Maduo/man_catalog.npy')
