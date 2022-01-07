while getopts ":h:s:d:" o; do
    case "${o}" in
        h) opt_name=${OPTARG};;
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

fork_of_runs () {
    opt_name=${1}
    cmd=${2}

    for percentile in 10 20 30 40 50 60 70 80 90
    do
        if [[ "$opt_name" == "KA_tpe" ]]
        then
            # Knowledge augmentation is only available for TPE
            next_cmd="${cmd} --knowledge_augmentation True --feasible_domain ${percentile} --opt_name tpe"
        elif [[ "$opt_name" == "normal_tpe" ]]
        then
            # non constraint TPE
            next_cmd="${cmd} --constraint False --opt_name tpe"
        else
            next_cmd="${cmd} --feasible_domain ${percentile} --knowledge_augmentation False --opt_name ${opt_name}"
        fi

        echo `date '+%y/%m/%d %H:%M:%S'`
        echo $next_cmd
        $next_cmd
    done
}

run_optimization_on_bench () {
    bench=${1}
    datasets=(${2})
    opt_name=${3}
    cidx=${4}
    seed=${5}

    for dataset in "${datasets[@]}"
    do
        cmd="python optimize_${bench}.py --exp_id ${seed} --max_evals 200 --constraint_mode ${cidx} --dataset ${dataset}"
        fork_of_runs "${opt_name}" "${cmd}"
    done
}

end=$((${start}+${dx}))
for seed in `seq $start $end`
do
    for cidx in 0 1 2
    do
        bench=hpolib
        datasets=('slice_localization' 'protein_structure' 'naval_propulsion' 'parkinsons_telemonitoring')
        run_optimization_on_bench "${bench}" "${datasets[*]}" "${opt_name}" "${cidx}" "${seed}"

        bench=nasbench101
        datasets=('cifar10A' 'cifar10B' 'cifar10C')
        run_optimization_on_bench "${bench}" "${datasets[*]}" "${opt_name}" "${cidx}" "${seed}"

        bench=nasbench201
        datasets=('cifar10' 'cifar100' 'imagenet')
        run_optimization_on_bench "${bench}" "${datasets[*]}" "${opt_name}" "${cidx}" "${seed}"
    done
done

echo `date '+%y/%m/%d %H:%M:%S'`
