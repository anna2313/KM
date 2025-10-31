# KM

This repository aims to get WP and WR estimate from an image of the Kaplan-Meier estimate of the cumulative distribution function of failure (or event). It also contains an example.

## Data

The example comes the the BAATAF trial from the 1990s. (https://www.nejm.org/doi/full/10.1056/NEJM199011293232201). We took Figure 1 and used https://web.eecs.utk.edu/~dcostine/personal/PowerDeviceLib/DigiTest/index.html to digitalize the image (add at least one point pair step to capture the step function structure). The data we anded up with is in placebo.csv and treatment.csv.

If you would like to use the code for a different trial with two arms, just clone the repository, follow the steps above and change the placebo.csv and treatment.csv to your own data.

## Data processing and visualization

First run 
