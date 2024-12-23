cd /scratch/homes/sfan/uncertainty/EE724
pip install -r requirements.txt
python run.py --dataset gsm8k --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/gsm8k/spec_TGT[8B]_DFT[3B].out
python run.py --dataset gsm8k --target_model_name meta-llama/Llama-3.2-3B --draft_model_name meta-llama/Llama-3.1-8B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/gsm8k/spec_TGT[3B]_DFT[8B].out
# ... add other model combinations similar to other scripts 