
Returns:
        gamma (float): A measure of how similar the datapoints are, closer to 0 means they are distributed among each other, more means they are clustered
        zeta (int): Clustering measure of the synthesized data, closer to 0 means none is more clustered than other, more positive is real is more clustered, more negative means synthesized is more clustered
        eta (int): Density measure of the synthesized data, closer to 0 means none is more dense than other, more positive is real is more dense (mode collapse, model has found a safe spot, much more clustered than real), more negative means synthesized is more dense (noisy generation, synth data are not as clustered as real)
        gammas (list[int]): The individual gamma_ij values; [gamma_11, gamma_12, gamma_21, gamma_22]