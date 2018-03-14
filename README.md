# SpaceVT-GSoC-Demo
Demo task for Space@VT GSoC 2018

# Summary
The objective of this project is to predict the Disturbance Storm Time (DST) index given Interplanetary Magnetic Field (IMF) data, particularly the Z component of magnetic field. The project uses a neural network to model a year of IMF data and predict DST data. The resulting model predicts the general trend of DST fairly well, but it usually fails to predict the exact value. However, this is a prototype, and there are a number of improvements that could possibly improve the accuracy.

# Background
The Disturbance Storm Time (DST) index measures a magnetic field produced by ring current flowing around the earth's equator. During periods of disturbed space weather, the ring current will increase, causing a larger magnetic field. Negative values of DSV indicate that the induced magnetic field opposes the Earth's magnetic field.

The Interplanetary Magnetic Field (IMF) is a magnetic field produced by solar wind, which is plasma that has been ejected from the sun (coronal mass ejection). NASA's OMNI database contains IMF measurements taken by near-earth spacecraft and time-shifted to Earth's bow shock nose, which is the region at the tip of the cone-shaped outer region of magnetosphere, closest to the sun.

Since solar winds cause disturbances in the Earth's ring current, these two measurements are closely related. In particular, the Z-component of the IMF is roughly perpindicular to the Earth's ring currents, so it has the greatest influence via electromagnetic induction.

# Problem
Predicting the effects of solar wind on the Earth's magnetosphere could be useful for radio communications and scientific research. We can use machine learning algorithms on IMF data, particularly the Z-component of magnetic field, to try to predict DST. Since solar wind moves much slower than light, taking several days to reach Earth, this could enable us to anticipate what kind of ring currents disturbances will occur in response to an observed solar coronal mass ejection.

# Solution
Since we have both the input data (IMF Bz) and the output labels (DST), this is a supervised learning problem. Neural networks are good at capturing relationships between input and output that may or may not be linear, and they usually work well, so I chose to model the data using a neural network. I implemented it using SciKitLearn's MLPRegressor with the ReLU activation function, and 4-fold cross-validation to test the accuracy. The only feature was IMF Bz data from the full year of 2015 (1-hour averages), and the label was the DST for the same time period. I used the 1-hour average Bz data because the DST is only reported once per hour.

At first, the predictions followed similar trends to the ground truth labels for negative DST, but would not make any prediction above a certain threshhold, so it did not predict positive DST well. This turned out to be a problem with the ReLU activiation function, which sometimes has problems if input data is a mix of positive and negative values or is not scaled a certain way. To fix this problem, I multiplied all the Bz data by 100, and added the minimum Bz value so that all the data was positive or 0. This resulted in predictions that did not cut off at a threshhold value. The R^2 value was rather low at around 0.1-0.2, because the predictions tend to follow the same trend, but are rarely exactly the same as the ground truth labels. I tested neural network layer sizes from 1-100, and found anything 5 or above had similar accuracy, so I used 5 layers for best efficiency and accuracy. I also experimented with adding different features including proton density and plasma flow speed from the IMF data set, but found they decreased accuracy. This may indicate that they also need some scaling and preprocessing, or they may not be well-correlated with the DST.

Images below show the predictions vs. truth over a year, as well as two zoomed-in time periods of the neural network working well and not working well. If you are viewing this report as plain text, you can find the graphs at:

https://github.com/e-271/SpaceVT-GSoC-Demo/tree/master/report

# Conclusions
Neural networks are a promising method for predicting the general trend of DST data, but a neural network trained using only Bz usually fails to predict the exact value of DST. It is possible that adding new features could increase the accuracy. The new features will need to be carefully selected and scaled for the neural network. It's also possible that there is some time lag between the Bz and the DST, but since we are using 1-hour averages it would need to be on that same timescale to have a significant effect. Another potential way of improving this algorithm is simply to give it more data - I used just one year, but IMF data is available as far back as 1963, and DST data is available from 1957. Training the network on 50 years of data instead of just 1 might increase the accuracy a lot. In conclusion, this algorithm is fairly good at predicting the general trend of DST, but is inaccurate in predicting the actual values. However, there are several ways that the value predictions could potentially be improved. 

# References
OMNI data documentation

https://omniweb.gsfc.nasa.gov/html/HROdocum.html

Source of DST data

http://wdc.kugi.kyoto-u.ac.jp/index.html

Source of IMF data

https://cdaweb.gsfc.nasa.gov/pub/data/omni/

GitHub page for this project

https://github.com/e-271/SpaceVT-GSoC-Demo

