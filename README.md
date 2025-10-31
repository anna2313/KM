# KM

This repository aims to get win proportion (WP) and win ratio (WR) estimate from an image of the Kaplan-Meier estimate of the cumulative distribution function of failure (or event). It also contains an example.

## Data

The example comes the the BAATAF trial from the 1990s. (https://www.nejm.org/doi/full/10.1056/NEJM199011293232201). We took Figure 1 and used https://web.eecs.utk.edu/~dcostine/personal/PowerDeviceLib/DigiTest/index.html to digitalize the image (add at least one point pair step to capture the step function structure). The data we anded up with is in placebo.csv and treatment.csv.

If you would like to use the code for a different trial with two arms, just clone the repository, follow the steps above and change the placebo.csv and treatment.csv to your own data.

## Data processing and visualization

First run process_visualize.py. This file takes the placebo.csv and treatment.csv, and adds boundaries (0 and 5 in our example), fixes the noise that was introduced by digitalizing the curve by making it into a step function. Finally, it samples additional points and visualizes both curves.

As a second step run sampling.py. This is the python file, that makes WP and WR approximations from the curves. It uses the new points created by process_visualize.py, so run that first. It samples from the cdf with uniform inverse sampling and approximates WP and WR from these samples. It reruns this process numerous times and prints out the histogram of the sampled WR values as well as the mean and median. You can change the number of trials and the number of patients in each arm.
