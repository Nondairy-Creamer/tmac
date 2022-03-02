## Motion correction in two-channel calcium imaging

This is a python package implementing the additive calcium inference model (ACIM).

### Installation:
Navigate to the python project directory
```
git clone https://github.com/Nondairy-Creamer/calcium_inference
cd calcium_inference
pip install -e .
```

### Usage:
To see an example on synthetic data, install the package into a python project, then run the script examples/ACIM_on_synthetic_data.py

The model can be called in a python project with the following commands, where red and green are numpy matricies of the red and green fluorescence data.

```
import calcium_inference.models as cim

trained_variables = cim.additive_calcium_inference_model(red, green)
```

Also included are two preprocessing functions. A function to linearly interpolate over time to remove NaNs from the data and one to correct for photobleaching by dividing by an exponential with a single decay constant fit with all the neural data. AIM assumes a constant mean and no NaNs, so adjustments like this are necessary preprocessing steps, though any method for data imputation and bleach correction will suffice. Note: one should not temporally filter the data as the gaussian process will do the necessary smoothing. Using the preprocessing steps:

```
import calcium_inference.calcium_inference_models as cim
import calcium_inference.preprocessing as cip

red_interp = cip.interpolate_over_nans(red)[0]
green_interp = cip.interpolate_over_nans(green)[0]

red_corrected = cip.photobleach_correction(red_interp)
green_corrected = cip.photobleach_correction(green_interp)

trained_variables = cim.additive_calcium_infernece_model(red_corrected, green_corrected)
```
