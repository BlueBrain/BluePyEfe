
BluePyEfe Change Log
=====================


v2.0.0
-------

**Architectural changes**:

* Restructuring of the code to a class-based code that can be used as an API
* Implementation of a hierarchy of the handling of the metadata associated with each trace
* Implementation of extraction of feature for non-step protocols
* BluePyEfe can now output the protocols as time series that can be read by BluePyOpt as protocols
* changes in the structure of the input config dictionary
* changes in plotting function of both features and traces

**Changes in the way the mean and standard deviations are computed**:

* The mean and std for a feature are saved if the number of data point for this target +- tolerance is above the threshold_nvalue_save option. NaNs are not taken into accound when comparing the number of points with threshold_nvalue_save.
* Any cell for which the rheobase could not be computed is not used when computing the mean or std of the features. In this case, it is however possible to set this rheobase value by hand before calling the mean_efeature function.
* Instead of saving protocols and efeatures having NaNs as mean or std, BPE2 removes them and issues a warning instead. If for a given protocol/target, there are no non-Nan values, then the protocol is not saved at all (a warning is issued as well).

**Implementation of an automatic step detection when ton/toff/tend/amp is not known**:

* An automatic step detection has been implemented for some of the simple eCodes (see bluepyefe/ecode/), 
it provides as an output ton, toff, tend, amp and hypamp. 
This automatic detection only works when the signal to noise ratio of the stimuli is good enough. 
Therefore, before exploiting the efeatures, the user should check that the automatic step detection has indeed found the correct step. 
This can be checked by plotting the real current data on top of the reconstruction of the current resulting from the step detection. 
For the non-step protocols, the timing information need to be provided by the user.
