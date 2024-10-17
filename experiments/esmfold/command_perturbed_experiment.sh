# Baselines
python experiments/perturbed/predict_gearnet_perturbed.py 
python experiments/perturbed/predict_gearnet_perturbed.py dataset=gearnet_go_bp
python experiments/perturbed/predict_gearnet_perturbed.py dataset=gearnet_go_cc
python experiments/perturbed/predict_gearnet_perturbed.py dataset=gearnet_go_mf
python experiments/perturbed/predict_mutation_perturbed.py --outdir logs/pst_perturbation/baselines
python experiments/perturbed/predict_scop_perturbed.py dataset=scop

# Complete graphs
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt data.edge_perturb=complete
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt dataset=gearnet_go_bp data.edge_perturb=complete
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt dataset=gearnet_go_cc data.edge_perturb=complete
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt dataset=gearnet_go_mf data.edge_perturb=complete
python experiments/perturbed/predict_mutation_perturbed.py --outdir logs/pst_perturbation/complete --perturbation complete --pretrained ./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt
python experiments/perturbed/predict_scop_perturbed.py dataset=scop data.edge_perturb=complete pretrained=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt

# Sequence graphs
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt data.edge_perturb=sequence
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt dataset=gearnet_go_bp data.edge_perturb=sequence
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt dataset=gearnet_go_cc data.edge_perturb=sequence
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt dataset=gearnet_go_mf data.edge_perturb=sequence
python experiments/perturbed/predict_mutation_perturbed.py --outdir logs/pst_perturbation/sequence --perturbation sequence --pretrained ./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt
python experiments/perturbed/predict_scop_perturbed.py dataset=scop data.edge_perturb=sequence pretrained=./logs_pst/random_ablation/edge_perturb_sequence/esm2_t6_8M_UR50D/runs/2024-07-20_17-54-40/model.pt

# Random graphs
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt data.edge_perturb=random
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt dataset=gearnet_go_bp data.edge_perturb=random
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt dataset=gearnet_go_cc data.edge_perturb=random
python experiments/perturbed/predict_gearnet_perturbed.py pretrained=./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt dataset=gearnet_go_mf data.edge_perturb=random
python experiments/perturbed/predict_mutation_perturbed.py --outdir logs/pst_perturbation/random --perturbation random --pretrained ./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt
python experiments/perturbed/predict_scop_perturbed.py dataset=scop data.edge_perturb=random pretrained=./logs_pst/random_ablation/edge_perturb_random/esm2_t6_8M_UR50D/runs/2024-07-21_18-41-49/model.pt

# python experiments/perturbed/predict_proteinshake_perturbed.py logs.prefix=logs_pst/proteinshake_perturbed perturbation=complete model_path=./logs_pst/random_ablation/edge_perturb_complete/esm2_t6_8M_UR50D/runs/2024-07-17_17-41-55/model.pt task=binding_site_detection,enzyme_class,gene_ontology,pfam_task,structural_class
