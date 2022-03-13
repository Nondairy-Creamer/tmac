## Two-channel motion artifact correction (TMAC)

### Installation:
Navigate to the python project directory
```
git clone https://github.com/Nondairy-Creamer/tmac
cd tmac
pip install -e .
```

### Usage:
To see an example on synthetic data, install the package into a python project, then run the script examples/TMAC_on_synthetic_data.py

The model can be called in a python project with the following commands, where red and green are numpy matricies of the red and green fluorescence data.

```
import tmac.models as tm

trained_variables = tm.tmac_ac(red, green)
```

Also included are two preprocessing functions. A function to linearly interpolate over time to remove NaNs from the data and one to correct for photobleaching by dividing by an exponential with a single decay constant fit with all the neural data. TMAC assumes a constant mean and no NaNs, so adjustments like this are necessary preprocessing steps, though any method for data imputation and bleach correction will suffice. Note: one should not temporally filter the data as the gaussian process will do the necessary smoothing. Using the preprocessing steps:

```
import tmac.models as tm
import calcium_inference.preprocessing as tp

red_interp = tp.interpolate_over_nans(red)[0]
green_interp = tp.interpolate_over_nans(green)[0]

red_corrected = tp.photobleach_correction(red_interp)
green_corrected = tp.photobleach_correction(green_interp)

trained_variables = tm.tmac_ac(red_corrected, green_corrected)
```
