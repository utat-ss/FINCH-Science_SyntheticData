# to run the pipeline on google cloud vm

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
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/micha/train.log.$TS
nohup python -m defs.diffusion.script --config training/diffusion/ExpAnalysis_002-4/config.yaml > "$LOG" 2>&1 &
echo "Started PID:$! LOG:$LOG"
```
