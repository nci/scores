import numpy as np

def apply_weights(values, weights=None):
    '''
    Returns:
        A new array with the elements of values multiplied by the specified weights.
        
    Args:
        - weights: The weightings to be used at every location in the values array. If weights contains additional 
        dimensions, these will be taken to mean that multiple weightings are wanted simultaneoulsy, and these 
        dimensions will be added to the new array.
        - values: The unweighted values to be used as the basis for weighting calculation


    Note - this weighting function is different to the .weighted method contained in xarray in that xarray's 
    method does not allow NaNs to be present in the weights or data.    
    '''
    
    if weights is not None:
        result = values * weights
        return result

    return values


def create_latitude_weights(latitudes):
    '''
    A common way of weighting errors is to make them proportional to the amount of area
    which is contained in a particular region. This is approximated by the cosine
    of the latitude on an LLXY grid. Nuances not accounted for include the variation in
    latitude across the region, or the irregularity of the surface of the earth.
    '''
    weights = np.cos(np.deg2rad(latitudes))
    return weights