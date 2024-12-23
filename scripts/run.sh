cd /scratch/homes/sfan/uncertainty/EE724
pip install -r requirements.txt
python run.py --dataset trivia_qa --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec --num_fewshot_data 5 > /scratch/homes/sfan/uncertainty/EE724/logs/test_trivial_qa.out
python run.py --dataset squad --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec --num_fewshot_data 5 > /scratch/homes/sfan/uncertainty/EE724/logs/test_squad.out
python run.py --dataset nq --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec --num_fewshot_data 5 > /scratch/homes/sfan/uncertainty/EE724/logs/test_nq.out
# python run.py --dataset bioasq --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec --num_fewshot_data 5 > /scratch/homes/sfan/uncertainty/EE724/logs/test_bioasq.out
# python run.py --dataset svamp --target_model_name meta-llama/Llama-3.1-8B --draft_model_name meta-llama/Llama-3.2-3B --run_spec --num_fewshot_data 5 > /scratch/homes/sfan/uncertainty/EE724/logs/test_svamp.out

