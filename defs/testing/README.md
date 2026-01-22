The following code includes all the functions that are used for the testing procedures, to acquire comparable metrics for the synthesized data.

The two methods of use are:
 - Nearest Neighbor: Where the synthesizer models are evaluated how spectrally resembling the synthesized data is compared to the true spectra. For more detail as to how such models work, please check the 'nearest_neighbor' folder's README.
 - Unmix: The synthesizer models are evaluated by how well we can unmix them and how much the unmixing makes sense. We do this by training an unmixer on synthesized data, which is then evaluated on ground truth data. This tells us how much useful the synthesized data are to train an unmixer. For more detail of the specific procedure, please check the 'unmix' folder's README.

Code in 'master' folder allows us to do both o these methods in the same script