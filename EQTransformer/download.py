import os
json_basepath = os.path.join(os.getcwd(),"json/station_list.json")

from EQTransformer.utils.downloader import makeStationList
from EQTransformer.utils.downloader import downloadMseeds

makeStationList(json_path=json_basepath, client_list=["SCEDC"], min_lat=35.50, max_lat=35.60, min_lon=-117.80, max_lon=-117.40, start_time="2019-09-01 00:00:00.00", end_time="2019-09-03 00:00:00.00", channel_list=["HH[ZNE]", "HH[Z21]", "BH[ZNE]"], filter_network=["SY"], filter_station=[])



downloadMseeds(client_list=["SCEDC", "IRIS"], stations_json=json_basepath, output_dir="downloads_mseeds", min_lat=35.50, max_lat=35.60, min_lon=-117.80, max_lon=-117.40, start_time="2019-09-01 00:00:00.00", end_time="2019-09-03 00:00:00.00", chunk_size=1, channel_list=[], n_processor=2)