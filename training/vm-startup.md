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

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion_2\testing_cfg_27000.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_27000.yaml --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion\testing_cfg_nn.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_nn.yaml --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\testing\unmix\critic_train\train.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/testing/unmix/critic_train/train.py --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\testing\unmix\script.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/testing/unmix/script.py --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\testing\master\script.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/testing/master/script.py --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\testing\unmix\critics\models\mlp.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/testing/unmix/critics/models/mlp.py --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\synthesis\loaders.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/synthesis/loaders.py --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion\generated_gdstreamline_3000_denoised.csv" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion/generated_gdstreamline_3000_denoised.csv --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion\psi1_gdstreamline_hybrid.csv" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/psi1_gdstreamline_hybrid.csv --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion_2\testing_cfg_27000_hybrid.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_27000_hybrid.yaml --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion\ExpAnalysis_002-4_20m_minmax\config_20m_minmax.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4_20m_minmax/config_20m_minmax.yaml --zone=northamerica-northeast2-b

gcloud compute ssh micha@gpu-ddim-run --zone=northamerica-northeast2-b --command="mkdir -p /home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4_20m_minmax"

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion\synthesis_cfg_minmax.yaml" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion/synthesis_cfg_minmax.yaml --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion\cfg_diffusion_setup_3.json" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion/cfg_diffusion_setup_3.json --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion\gdstreamline_statedict_3.pth" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion/gdstreamline_statedict_3.pth --zone=northamerica-northeast2-b

gcloud compute scp "C:\work\project\git-repo\FINCH-Science_SyntheticData\defs\diffusion\data\data_preperation.py" micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/defs/diffusion/data/data_preperation.py --zone=northamerica-northeast2-b


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

nohup python -m defs.diffusion.script --config training/diffusion/ExpAnalysis_002-4_20m_minmax/config_20m_minmax.yaml &

export WANDB_API_KEY=YOUR_API_KEY_HERE
nohup python -m defs.synthesis.script --config synthesis/isprs/diffusion/synthesis_cfg_3.yaml &

nohup python -m defs.testing.master.script --config testing/isprs/diffusion/testing_cfg_3000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_27000_hybrid.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_3000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_9000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion_2/testing_cfg_27000.yaml &

nohup python -m defs.testing.master.script --config /home/micha/FINCH-Science_SyntheticData/testing/isprs/diffusion/testing_cfg_nn.yaml &

pkill -f 'defs.testing.master.script' || true
```

### show logs
```text
tail -n 200 -f "$LOG"

tail -n 40 nohup.out

tail -f nohup.out
```

## grab files from vm
```text
gcloud compute scp --recurse --zone=northamerica-northeast2-a "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion" "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\vm"

gcloud compute scp --recurse --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/gdstreamlined_3k_critic.pth" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/gdstreamlined_27k_critic_hybrid.pth" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion_2"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/preds_27k_hybrid.parquet" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion_2"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/logs_9k.txt" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion_2"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/logs_nearneigh.txt" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/metric_save.csv" "C:\work\project\git-repo\FINCH-Science_SyntheticData\testing\isprs\diffusion"

gcloud compute scp --recurse --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/training/diffusion/ExpAnalysis_002-4_20m_3" "C:\work\project\git-repo\FINCH-Science_SyntheticData\training\diffusion\ExpAnalysis_002-4_20m_3\vm"

gcloud compute scp --zone=northamerica-northeast2-b "micha@gpu-ddim-run:/home/micha/FINCH-Science_SyntheticData/synthesis/isprs/diffusion/generated_gdstreamline_3.csv" "C:\work\project\git-repo\FINCH-Science_SyntheticData\synthesis\isprs\diffusion"


```

## shut down vm
```text
gcloud compute instances stop gpu-ddim-run --zone=northamerica-northeast2-a
gcloud compute instances describe gpu-ddim-run --zone=northamerica-northeast2-a --format="get(status)"

source ~/venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install pyarrow
# (optional) also install fastparquet:
pip install fastparquet

tail -n 400 /home/micha/FINCH-Science_SyntheticData/nohup.out

```
