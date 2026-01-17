This readme file defines what each of the training runs are for.

- 3: The basis in this series is better oversampling hyperparams 0.075 (instead of 1) for amp and 0.0016 (instead of 0.01) for jitter
    - 3-0: Compared to 2-4, this one only has the proper oversampling hyperparams
    - 3-1: Compared to 3-0, this one has stronger guidance 4.0 rather than 3.0
    - 3-2: Compared to 3-0, this one has more ddim steps 250 rather than 100
    - 3-3: Compared to 3-0, this one has less ddim steps 50 rather than 100
    - 3-4: Compared to 3-0, this one has stronger EMA .99995 instead of .999
    - 3-5: Compared to 3-0, this one has more uncond probability 0.2 than 0.1