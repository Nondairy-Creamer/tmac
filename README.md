## Two-channel motion artifact correction (TMAC)

### Installation:
Navigate to the python project directory
```
git clone https://github.com/Nondairy-Creamer/tmac
cd tmac
pip install -e .
```

### General description:
For a full description please consult the paper
https://arxiv.org/abs/2204.12595

Formally this is a motion correction method that performs inference over a generative model of two simultaneously measured channels to remove shared motion artifacts. One channel is activity-dependent and the other must be activity-independent.

Informally, this method simply subtracts the activity-independent channel from the activity-dependent channel while accounting for noise.

### Usage:
To see an example on synthetic data, install the package into a python project, then run the script examples/tmac_on_synthetic_data.py

The model can be called in a python project with the following commands, where red and green are numpy matricies of the red and green fluorescence data with dimensions [time, neurons].

```
import tmac.models as tm

trained_variables = tm.tmac_ac(red, green)
```

Also included are two preprocessing functions. A function to linearly interpolate over time to remove NaNs from the data and one to correct for photobleaching by dividing by an exponential with a single decay constant fit with all the neural data. TMAC assumes a constant mean and no NaNs, so adjustments like this are necessary preprocessing steps, though any method for data imputation and bleach correction will suffice. Using the preprocessing steps:

```
import tmac.models as tm
import tmac.preprocessing as tp

red_interp = tp.interpolate_over_nans(red)[0]
green_interp = tp.interpolate_over_nans(green)[0]

red_corrected = tp.photobleach_correction(red_interp)
green_corrected = tp.photobleach_correction(green_interp)

trained_variables = tm.tmac_ac(red_corrected, green_corrected)
```

### Output:
The output dictionary contains
* **a**: The neural activity (time, neurons)
* **m**: The motion artifact (time, neurons)
* **length\_scale\_a**, **length\_scale\_m**: the timescale of the gaussian process for a and m in units of time indices (neurons,)
* **variance\_a**, **variance\_m**: the amplitude of a and m (neurons,)
* **variance\_g\_noise**, **variance\_r\_noise**: the amplitude of the channel noise for r and g (neurons,)

### Notes
* Every neuron is processed separately. The input is a matrix for convenience.

* The photobleach correction provided shares a bleaching tau across neurons, so is not independent across neurons.
* Do not temporally filter the data. The Gaussian process prior over a and m will perform the necessary smoothing without reducing temporal resolution.
* The activity a has mean 1 and units of fold change from the mean. If you want it to be unitful, you can multiply each neurons activity by the mean over time of the green channel input.

### Assumptions and limitations
* The math is performed using a Fourier basis which treats the data as circular. This means that the inferred activity at time 0 a[0] is smoothed together with the data the the end of the vector a[-1]. For most data sets the timescale of a should be far smaller than the length of the data so this effect should be small.

* Caution should be used on very small data sets. If the total measurement time is < 4x the timescale of the activity, the hyperparameters may have trouble fitting accurately. Furthermore the correlations at the ends of the data will be pronounced in such a data set. This is generally uncommon as most recordings are long enough to observe many fluctuations of neural activity. 
* All the hyperparameters of the data (see output section) are assumed to be constant over the recording. For instance, if the timescale of your activity changes during the experiment consider splitting your data into separate inputs so the hyperparameters will fit each segment separately. Otherwise the fit hyperparameters will be some average of the changing values.
* TMAC assumes a constant mean. If for some reason the mean of your data shifts TMAC will not work properly. A shifting mean could occur from changing laser power during an experiment. The most common example of shifting mean is photobleaching which is why it must be corrected before running TMAC
* Bleedthrough is not accounted for by TMAC. If 10% of your channel intensity is bleedthrough you will lose 10% of the amplitude of your signal.
