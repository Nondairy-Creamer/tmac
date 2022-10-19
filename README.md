## Two-channel motion artifact correction (TMAC)

### Installation:
Navigate to the python project directory
```
git clone https://github.com/Nondairy-Creamer/tmac
cd tmac
pip install -e .
```

### General description:
For a full description please consult the paper:
https://doi.org/10.1371/journal.pcbi.1010421

Modified from the Author Summary in the paper:

Optical imaging of neural activity using fluorescent indicators is a powerful technique for studying the brain, yet despite the widespread success of this technique many difficulties remain. Imaging moving animals can be challenging, because the animal’s movements may introduce changes in fluorescence intensity that are unrelated to neural activity, known as motion artifacts. We focus on the special case of two-channel imaging, where two indicators are present in the same cell: an activity-dependent indicator and an activity-independent indicator. This is the code for our Two-channel Motion Artifact Correction (TMAC) method. TMAC uses a generative model of the fluorescence to infer the latent neural activity without motion artifacts. A more intuitive description would be that this method simply subtracts the activity-independent channel from the activity-dependent channel while accounting for channel-independent noise.

Below we refer to the activity-depdendent channel as the "green channel" and the activity-independent channel as the "red channel" because it relates to the common use case where researchers measure calcium dependent GCaMP fluorescence of a red fluorophore and activity-independent fluoresence of a red fluorophore.

Importantly, TMAC will remove motion artifacts in any type of two-channel imaging, so long as one channel is activity-independent and both channels share
the same motion artifact component. TMAC could therefore be applied to a wide range of two-channel imaging modalities including for voltage imaging, fiber photometry when using an isosbestic wavelength, or two-channel two-photon imaging.

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
* **a**: [time, neurons], the neural activity 
* **m**: [time, neurons], the motion artifact
* **length\_scale\_a**, **length\_scale\_m**: [neurons,], the timescale of the gaussian process for a and m in units of time indices
* **variance\_a**, **variance\_m**: [neurons,], the amplitude of a and m
* **variance\_g\_noise**, **variance\_r\_noise**: [neurons,], the amplitude of the channel noise for r and g

### Notes
* Every neuron is processed separately. The input is a matrix for convenience.

* TMAC is deterministic. You will get the same answer even if you rerun it.
* The photobleach correction provided shares a bleaching tau across neurons, so is not independent across neurons.
* Do not temporally filter the data. The Gaussian process prior over a and m will perform the necessary smoothing without reducing temporal resolution.
* The activity a has mean 1 and units of fold change from the mean. If you want it to be unitful, you can multiply each neuron's activity by the mean over time of the green channel input i.e. 
```
a_unitful = a * np.mean(green_corrected, axis=0, keepdims=True)
```

### Assumptions and limitations
* The math is performed using a Fourier basis which treats the data as circular. This means that the inferred activity at the beginning of a is smoothed together with the values the end of the vector a. For most data sets the recording should be many times longer than timescale of a, so this effect should be small.

* TMAC works when the fluorophores for the two channels are expressed and measured in the same place and time. Different expression patterns, segmentation, or measuring the channels at different times may lead to different motion artifacts in the two channels.
* Caution should be used on very small data sets. If the total measurement time is < 4x the timescale of the activity, the hyperparameters may have trouble fitting accurately. Furthermore the correlations at the ends of the data will be pronounced in such a data set (see previous bullet). This is generally uncommon as most recordings are long enough to observe many fluctuations of neural activity. 
* All the hyperparameters of the data (see output section) are assumed to be constant over the recording. For instance, if the timescale of your activity changes during the experiment consider splitting your data into separate recordings. Then feed these recordings into TMAC individually so that the hyperparameters will fit each segment separately. Otherwise the fit hyperparameters will be some average of the changing values.
* TMAC assumes a constant mean. If for some reason the mean of your data shifts TMAC will not work properly. A shifting mean could occur from changing laser power during an experiment. The most common example of shifting mean is photobleaching which is why it must be corrected before running TMAC
* Bleedthrough is not accounted for by TMAC. If 10% of the channel intensity in your activity-independent channel is from your activity-dependent channel you will lose ~10% of your amplitude in the inferred activity.
