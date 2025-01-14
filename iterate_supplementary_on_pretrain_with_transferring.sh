learning_rates=(1e-2 5e-3 1e-3 1e-4 5e-4)
target_datasets=("photo")

backbone="fagcn"
backbone_tuning=1 
split_method="RandomWalk"

few_shot=1
batch_sizes=(100) 


for target_dataset in "${target_datasets[@]}"; do
    source_dataset_str=""
    datasets=( "cornell"  "cora" "citeseer" )
    for dataset in "${datasets[@]}"; do
        if [ "$dataset" != "$target_dataset" ]; then
            source_dataset_str+="${dataset},"
        fi
    done

    source_dataset_str="${source_dataset_str%,}"
    echo $source_dataset_str
    echo $target_dataset
    echo "storage/reconstruct/${source_dataset_str}_pretrained_model.pt"    

    python src/exec.py --config-file pretrain.json --general.save_dir "storage/fagcn/reconstruct" --general.reconstruct 0.2 --data.name "cora,citeseer,cornell" --pretrain.split_method RandomWalk --model.backbone.model_type fagcn
    
    
    for lr in "${learning_rates[@]}"
    do
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func adapt  --general.save_dir "storage/fagcn/balanced_few_shot_fine_tune_backbone_with_rec" --general.few_shot 1  --general.reconstruct 0.0 --data.node_feature_dim 100 --data.name "photo" --adapt.method finetune --model.backbone.model_type fagcn --model.saliency.model_type "none"  --adapt.pretrained_file "storage/fagcn/reconstruct/cora,citeseer,cornell_pretrained_model.pt" --adapt.finetune.learning_rate 1e-2 --adapt.batch_size 100 --adapt.finetune.backbone_tuning 1        
        done 
    done
done