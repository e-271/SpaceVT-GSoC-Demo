import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.neural_network import MLPRegressor


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

    dst = {"time": np.array(times), "dst": np.array(dsts), "base_value": np.array(base_values), "daily_mean": np.array(daily_means)}
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

    imf = {"time": np.array(times), "Bx": np.array(Bx), "By": np.array(By), "Bz": np.array(Bz),
           "plasma_flow_speed": np.array(plasma_flow_speed), "proton_density": np.array(proton_density),
           "proton_temperature": np.array(proton_temp)}

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
    # Scale bz
    # RELU likes to have positive inputs
    bz = imf["Bz"]*100
    bz += np.min(bz)
    bz = list(bz)
    time = list(dst["time"])
    dstt = list(dst["dst"])

    layer_size = 5

    n_folds = 4
    ll = len(time)
    fold_l = ll / n_folds
    test_a = 0
    test_b = fold_l+1

    for fold in range(1, n_folds+1):
        #split = int(len(dst["time"]) / 2)
        print "#################"
        print "fold", fold
        print "#################"

        train_feats = np.array(bz[:test_a]+bz[test_b:]).reshape(-1, 1)
        test_feats = np.array(bz[test_a:test_b]).reshape(-1, 1)

        train_labels = np.array(dstt[:test_a]+dstt[test_b:]).reshape(-1, 1)
        test_labels = np.array(dstt[test_a:test_b]).reshape(-1, 1)

        train_time = np.array(time[:test_a]+time[test_b:]).reshape(-1, 1)
        test_time = np.array(time[test_a:test_b]).reshape(-1, 1)

        max_train = -1
        max_test = -1
        layer_train = -1
        layer_test = -1

        for layer_size in range(1,10):
            nn = MLPRegressor(hidden_layer_sizes=layer_size, alpha=0.01, activation="relu", solver="lbfgs",
                              random_state=9)
            nn.fit(train_feats, train_labels)

            pred_labels = nn.predict(train_feats)
            #pred_labels = np.array([int(np.round(l)) for l in pred_labels])
            train_score = nn.score(train_feats, train_labels)
            #print "R^2 on train data: ", train_score

            pred_labels = nn.predict(test_feats)
            #pred_labels = np.array([int(np.round(l)) for l in pred_labels])
            test_score = nn.score(test_feats, test_labels)
            #print "R^2  on test data: ", test_score

            if max_test < test_score:
                max_test = test_score
                layer_test = layer_size
            if max_train < train_score:
                max_train = train_score
                layer_train = layer_size

        print "best training R^2 was", max_train, "with", layer_train, "layers"
        print "best testing R^2 was", max_test, "with", layer_test, "layers"

        test_a += fold_l
        test_b += fold_l

    nn = MLPRegressor(hidden_layer_sizes=layer_train, alpha=0.01, activation="relu", solver="lbfgs",
                      random_state=9)
    nn.fit(train_feats, train_labels)

    pred_labels = nn.predict(train_feats)
    pred_labels = np.array([int(np.round(l)) for l in pred_labels])

    plt.plot(train_time, train_labels, label="training labels")
    plt.plot(train_time, pred_labels, label="predicted labels")
    plt.title("training data")
    plt.legend()
    plt.show()

    pred_labels = nn.predict(test_feats)
    pred_labels = np.array([int(np.round(l)) for l in pred_labels])
    print "R^2  on test data: ", nn.score(test_feats, test_labels)

    plt.plot(test_time, test_labels, label="test labels")
    plt.plot(test_time, pred_labels, label="predicted labels")
    plt.title("test data")
    plt.legend()
    plt.show()

dst = get_dst_from_csv("dst_2015.csv")
omni, imf = get_omni_from_csv("omni2_2015_hourlyavg.dat")
learn_predict(dst, imf)