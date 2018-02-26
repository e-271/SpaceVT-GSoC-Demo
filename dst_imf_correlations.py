import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression


def get_dst_from_csv(filepath):
    """
    Load a DST file in the WDC format into a dictionary with useful key names
    http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html
    :param filepath: path to the csv file
    :return: dictionary of times and dst
    """
    file = open(filepath)
    times = []
    base_values = []
    daily_means = []
    dsts = []

    for line in file:
        year = int(line[14:16] + line[3:5])
        month = int(line[5:7])
        day = int(line[8:10])
        base_value = int(line[16:20])
        daily_mean = int(line[116:120])

        for h in range(20, 116, 4):
            hour = int((h - 20) / 4)
            dstt = int(line[h:h+4])
            times.append(dt.datetime(year, month, day, hour))
            base_values.append(base_value)
            daily_means.append(daily_mean)
            dsts.append(dstt)

    dst = {"time": times, "dst": dsts, "base_value": base_values, "daily_mean": daily_means}
    return dst

def get_omni_from_csv(filepath):
    """
    Get hourly-average OMNI data from a csv file.
    Format is assumed to be this type:
    ftp://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
    Data from:
    ftp://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/

    According to this source, the GSM coordinate system is better for studying the effect of IMF on magnetospheric and ionospheric conditions.
    http://www.mssl.ucl.ac.uk/grid/iau/extra/local_copy/SP_coords/geo_sys.htm

    :param filepath: path to data file
    :return: pandas dataframe with all the raw OMNI data
            dictionary with relevant IMF data and useful key names, similar format to DST data
    """
    omni = pd.read_csv(filepath, delim_whitespace=True, header=None)
    times = []

    for i in range(len(omni[0])):
        year = omni[0][i]
        day_of_year = omni[1][i]
        hour = omni[2][i]
        time = dt.datetime(year, 1, 1, hour) + dt.timedelta(day_of_year - 1)
        times.append(time)

    # units: nT
    # coordinate system: GSM
    Bx = omni[12]
    By = omni[15]
    Bz = omni[16]

    plasma_flow_speed = omni[24] # km / s
    proton_density = omni[23]    # N / cm^3
    proton_temp = omni[22]       # degrees K

    imf = {"time": times, "Bx": Bx, "By": By, "Bz": Bz, "plasma_flow_speed": plasma_flow_speed,
                        "proton_density": proton_density, "proton_temperature": proton_temp}

    return omni, imf


def graph_dst_and_imf(dst, imf):
    """
    Plot DST and IMF data on the same timescale
    :param dst: dictionary of DST data from get_dst_from_csv
    :param imf: dictionary of IMF data from get_omni_from_csv
    """
    # Make sure dst.time == imf.time
    assert np.sum(np.array(dst["time"]) - np.array(imf["time"])) == dt.timedelta(0)

    fig, ax_arr = plt.subplots(5, sharex=True)
    ax_arr[0].plot(dst["time"], dst["dst"])
    ax_arr[0].set_title('Time vs. DST')

    ax_arr[1].plot(imf["time"], imf["Bz"])
    ax_arr[1].set_title('Time vs. Bz component of IMF')

    ax_arr[2].plot(imf["time"], imf["proton_density"])
    ax_arr[2].set_title('Time vs. Proton density')


    ax_arr[3].plot(imf["time"], imf["plasma_flow_speed"])
    ax_arr[3].set_title('Time vs. Flow speed of plasma')


    ax_arr[4].plot(imf["time"], imf["proton_temperature"])
    ax_arr[4].set_title('Time vs. Proton temperature')
    plt.show()

def learn_predict(dst, imf):
    """

    :param dst:
    :param imf:
    :return:
    """
    split = int(len(dst["time"]) / 2)
    train_feats = np.array(imf["Bz"][:split]).reshape(-1,1)
    test_feats = np.array(imf["Bz"][split:]).reshape(-1,1)
    train_labels = np.array(dst["dst"][:split]).reshape(-1,1)
    test_labels = np.array(dst["dst"][split:]).reshape(-1,1)
    test_time = np.array(dst["time"][split:]).reshape(-1, 1)

    linreg = LinearRegression()
    linreg.fit(train_feats, train_labels)

    pred_labels = linreg.predict(test_feats)
    pred_labels = np.array([int(np.round(l)) for l in pred_labels])

    plt.plot(test_time, test_labels, test_time, pred_labels)
    plt.show()


dst = get_dst_from_csv("dst_2015.csv")
omni, imf = get_omni_from_csv("omni2_2015_hourlyavg.dat")
#graph_dst_and_imf(dst, imf)
learn_predict(dst, imf)