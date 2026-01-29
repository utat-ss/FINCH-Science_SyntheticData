# to run the pipeline on google cloud vm

## copy directory onto vm
```text
gcloud compute scp --recurse --zone=northamerica-northeast2-a "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/
```

## start vm:
```text
gcloud compute instances start gpu-ddim-run --zone=northamerica-northeast2-a
```

## copy configs onto vm:
```text
gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion\ExpAnalysis_002-4\config.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4/config.yaml --zone=northamerica-northeast2-a
```

## ssh into vm:
```text
gcloud compute ssh micha@gpu-ddim-run --zone=northamerica-northeast2-a
```

## vm commands:
```text
source ~/venv/bin/activate
pkill -f 'defs.diffusion.script' || true
cd ~/FINCH-Science_SyntheticData
LOG=/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4/test_logs.txt
nohup python -m defs.diffusion.script --config training/diffusion/ExpAnalysis_002-4/config.yaml > "$LOG" 2>&1 &
echo "Started PID:$! LOG:$LOG"
```

## grab files from vm
```text
gcloud compute scp --recurse --zone=northamerica-northeast2-a "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion" "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\vm"
```

## shut down vm
```text
gcloud compute instances stop gpu-ddim-run --zone=northamerica-northeast2-a
gcloud compute instances describe gpu-ddim-run --zone=northamerica-northeast2-a --format="get(status)"
```
