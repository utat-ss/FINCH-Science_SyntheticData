This readme file defines what each of the training runs are for.

- 3: The basis in this series is better oversampling hyperparams
    - 3-0: Compared to 2-4, this one only has the proper oversampling hyperparams; 0.075 (instead of 1) for amp and 0.0016 (instead of 0.01) for jitter
        - results are worse than 2-4
    - 3-1: Compared to 2-4, 0.2 for amp and 0.05 for jitter
        - results are worse than 2-4
        - since the results for 3-0 and 3-1 are worse than 2-4, the best hyperparam for oversampling is between 0.075/0.0016-0.2/0.05
        - for now, 0.1/0.01 are probably good enough
    - 3-2: Compared to 2-4, has stronger EMA of 0.9999 compared to 0.999
        - the stronger EMA resulted in more variance in recons (thus producing worse results), which was expected
        - should not go lower than 0.999 since window for 0.999 is already at about 1k steps, which is 8k datapoints, which is a good memory range since it'd have seen the dataset 5.3 times, for 0.9999 this is 54 times
        - 0.999 is probably the best hyperparam
    - 3-3: Compared to 2-4 this one has stronger guidance, 4.0 rather than 3.0  with uncond probaility of 0.2 instead of 0.1
    - 3-4: Compared to 2-4, this one has more ddim steps 250 rather than 100
    - 3-5: Compared to 2-4, this one has less ddim steps 50 rather than 100
    - 3-6: Compared to 2-4, this one has more uncond probability 0.2 than 0.1