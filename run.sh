run_list='land cifer'

for r in $run_list
do
    python main.py --yaml_config ./configs/$r.yaml
done 
