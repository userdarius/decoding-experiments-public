cd /scratch/homes/sfan/uncertainty/EE724
pip install -r requirements.txt
python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-70B --draft_model_name meta-llama/Llama-3.2-1B --run_base > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/base_TGT[70B].out
python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-70B --draft_model_name meta-llama/Llama-3.2-1B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[70B]_DFT[1B].out
python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-70B --draft_model_name meta-llama/Llama-3.2-3B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[70B]_DFT[3B].out
python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-70B --draft_model_name meta-llama/Llama-3.1-8B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[70B]_DFT[8B].out

# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[8B]_DFT[3B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-3B --draft_model_name meta-llama/Llama-3.1-8B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[3B]_DFT[8B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-1B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[8B]_DFT[1B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-1B --draft_model_name meta-llama/Llama-3.1-8B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[1B]_DFT[8B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-3B --draft_model_name meta-llama/Llama-3.2-1B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[3B]_DFT[1B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-1B --draft_model_name meta-llama/Llama-3.2-3B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[1B]_DFT[3B].out

# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-1B --run_base > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/base_TGT[1B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-1B --draft_model_name meta-llama/Llama-3.2-3B --run_base > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/base_TGT[3B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-3B --draft_model_name meta-llama/Llama-3.1-8B --run_base > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/base_TGT[8B].out

# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-1B --draft_model_name meta-llama/Llama-3.2-1B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[1B]_DFT[1B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.2-3B --draft_model_name meta-llama/Llama-3.2-3B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[3B]_DFT[3B].out
# python run-full.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.1-8B --run_spec > /scratch/homes/sfan/uncertainty/EE724/logs/trivia_qa-full/spec_TGT[8B]_DFT[8B].out