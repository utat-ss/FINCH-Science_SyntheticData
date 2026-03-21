# to run the pipeline on google cloud vm

## copy directory onto vm
```text
gcloud compute scp --recurse --zone=northamerica-northeast2-a "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/

gcloud compute scp --recurse --zone=northamerica-northeast2-b "C:\work\project\git-repo\FINCH-Science_SyntheticData" micha@gpu-ddim-run:/home/micha/
```

## start vm:
```text
gcloud compute instances start gpu-ddim-run --zone=northamerica-northeast2-a

gcloud compute instances start gpu-ddim-run --zone=northamerica-northeast2-b
```

## copy configs onto vm:
```text
gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion\ExpAnalysis_002-4\config.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4/config.yaml --zone=northamerica-northeast2-a

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion\testing_cfg_3000.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_3000.yaml --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\testing\unmix\critic_train\train.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/testing/unmix/critic_train/train.py --zone=northamerica-northeast2-b
```

## ssh into vm:
```text
gcloud compute ssh micha@gpu-ddim-run --zone=northamerica-northeast2-a

gcloud compute ssh micha@gpu-ddim-run --zone=northamerica-northeast2-b
```

## vm commands:
```text
source ~/venv/bin/activate
pkill -f 'defs.synthesis.script' || true
cd ~/FINCH-Science_SyntheticData
LOG=/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4/test_logs.txt
nohup python -m defs.diffusion.script --config training/diffusion/ExpAnalysis_002-4/config.yaml > "$LOG" 2>&1 &
echo "Started PID:$! LOG:$LOG"

export WANDB_API_KEY=YOUR_API_KEY_HERE
nohup python -m defs.synthesis.script --config synthesis/isprs/diffusion/synthesis_cfg.yaml &

nohup python -m defs.testing.master.script --config testing/isprs/diffusion/testing_cfg_3000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_3000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_9000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_27000.yaml &

pkill -f 'defs.testing.master.script' || true
```

### show logs
```text
tail -n 200 -f "$LOG"

tail -n 20 nohup.out
```

## grab files from vm
```text
gcloud compute scp --recurse --zone=northamerica-northeast2-a "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion" "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\vm"

gcloud compute scp --recurse --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion" "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion\vm"
```

## shut down vm
```text
gcloud compute instances stop gpu-ddim-run --zone=northamerica-northeast2-a
gcloud compute instances describe gpu-ddim-run --zone=northamerica-northeast2-a --format="get(status)"
```
