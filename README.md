# SpaceVT-GSoC-Demo
Demo task for Space@VT GSoC 2018

# Background


The Disturbance Storm Time (DST) index measures a magnetic field produced by ring current flowing around the earth's equator. During periods of disturbed space weather, the ring current will increase, causing a larger magnetic field. Negative values of DSV indicate that the induced magnetic field opposes the Earth's magnetic field.


The Interplanetary Magnetic Field (IMF) is a magnetic field produced by solar wind, which is plasma that has been ejected from the sun (coronal mass ejection). NASA's OMNI database contains IMF measurements taken by near-earth spacecraft and time-shifted to Earth's bow shock nose, which is the region at the tip of the cone-shaped outer region of magnetosphere, closest to the sun.


Since solar winds cause disturbances in the Earth's ring current, these two measurements are closely related. In particular, the Z-component of the IMF is roughly perpindicular to the Earth's ring currents, so it has the greatest influence via electromagnetic induction.


To determine the effects of solar events on the Earth's magnetosphere, we can use machine learning algorithms on existing data to create a predictive model. This will allow us to run simulations, and possibly predict ring-current changes in response to solar wind.


# Resources
OMNI data documentation
https://omniweb.gsfc.nasa.gov/html/HROdocum.html
