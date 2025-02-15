G. Hayes 2024

These are the scripts I used for the ramp cerebrovascular reactivity (CVR) analysis in:

G. Hayes, S. Sparks, J. Pinto, and D. P. Bulte, “Ramp protocol for non-linear cerebrovascular reactivity with transcranial doppler ultrasound,” Journal of Neuroscience Methods, vol. 416, p. 110381, Apr. 2025, doi: 10.1016/j.jneumeth.2025.110381.

The data consisted of 2 MHz pulsed transcranial Doppler ultrasound blood flow velocity, acquired in the middle cerebral artery.
The CO2 data was acquired using a thin nasal cannula placed into both nostrils and an infrared gas analyser.
These signals were acquired during a protocol of consisted of 3 repetitions of 5 deep breaths, followed by 30 s of regular breathing on synthetic medical air, 40 s breathing a 5% CO2 balanced gas mixture, and 40 s breathing a 10% CO2 balanced gas mixture.

Feel free to download and alter this code to suit your own analysis and reference/acknowledge the publication above if you do so.

Adjust parameters as needed for your data. Notably, pay attention to your file paths, sampling rate, the peak promience for peak identification, and the barometric pressue in your location.

It should be noted that our input powerlabs (PWL) data had the following channels: channel 1 of the pwl data is the CO2, channel 2 is the O2 data, channel 3 is the raw TCD data, channel 4 is the PPG data, and channel 5 is the comment data.
