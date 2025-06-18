#### Experiments ####

## preshower v2 (may 8-14) ##
The experiments between may 10th and may 14th are all using the same geometry/parameters
The difference is only in the objectives
We ran 2 objectives at a time and ran all combos, giving 5 experiments
    The only combination we didn't run was the 2 mu/pi objective experiment
    Ian had already done this study, and also because the focus is neutral hadrons
    I just didn't do it. We could do it though. But we probably already have results
    from Ian.
    
## preshower v1 (april 26) ##
github commit: https://github.com/aid2e/dRICH-MOBO/tree/ff52d61ca5658131822330e729aefca440f7fa3e
Here we used preshower ratios instead of values
Three params:
    1. preshower ratio: ratio of preshower section thickness relative to postshower section
    2. preshower steel ratio: ratio of steel thickness relative to scint in preshower section
    3. postshower steel ratio: ratio of steel to scint in postshower section
We used all four objectives:
    1. High energy Neutron RMSE
    2. Low energy Neutron RMSE
    3. 1GeV pion vs muon ID
    4. 5GeV pion vs muon ID
        
## linear ratio (april 18) ##
Here we add some complexity by allowing the thickness of the layer to vary (linearly)
with the layer number (effectively the radius)
Thus, we have three objectives
    1. Number of layers
    2. Thick diff ratio: this was kinda complicated, so you can see the source here:
        https://github.com/simons27/epic_klm/blob/linear_ratio/src/KLMWS_geo.cpp#L151
    3. Steel ratio (ratio of steel to scint)

We use three objectives:
    1. High energy Neutron RMSE
    2. Low energy Neutron RMSE
    3. 1GeV pion vs muon ID

## Basic (april 4) ##
This experiment was the most basic: we vary:
    1. the ratio of steel to scint in each layer, and 
    2. the number of total layers
We use all four objectives:
    1. High energy Neutron RMSE
    2. Low energy Neutron RMSE
    3. 1GeV pion vs muon ID
    4. 5GeV pion vs muon ID

