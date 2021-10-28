#!/bin/bash
datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
metrics=(equal_opportunity equalized_odds predictive_equality statistical_parity)
rseed=(0 1 2 3 4 5 6 7 8 9)

echo "#!/bin/bash"
echo "#SBATCH --time=02:00:00"
echo "#SBATCH --ntasks=301"
echo "#SBATCH --mem-per-cpu=4G"
echo "export TMPDIR=/tmp"
echo "cd ../../core"

for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        for metric in "${metrics[@]}"
                            do 
                                FILE_DT="../../results_dt/${dataset}/${model}/${metric}_${r}.csv"
                                CMD_DT="srun python LaundryML_Fairlearn.py --dataset=${dataset} --rseed=${r} --metric=${metric} --explainer=dt  --model_class=${model} --transfer"
                                if [ ! -f $FILE_DT ]; then
                                    echo "${CMD_DT}"
                                fi

                                FILE_LM="../../results_lm/${dataset}/${model}/${metric}_${r}.csv"
                                CMD_LM="srun python LaundryML_Fairlearn.py --dataset=${dataset} --rseed=${r} --metric=${metric} --explainer=lm  --model_class=${model} --transfer"
                                if [ ! -f $FILE_LM ]; then
                                    echo "${CMD_LM}"
                                fi

                                FILE_RL="../../results_rl/${dataset}/${model}/${metric}_${r}.csv"
                                CMD_RL="srun python LaundryML.py --dataset=${dataset} --rseed=${r} --metric=${metric} --model_class=${model} --transfer"
                                if [ ! -f $FILE_RL ]; then
                                    echo "${CMD_RL}"
                                fi
                            done
                    done
            done
    done


