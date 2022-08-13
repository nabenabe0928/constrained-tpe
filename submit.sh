while getopts ":s:d:" o; do
    case "${o}" in
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

for opt_name in tpe naive_tpe KA_tpe hm normal_tpe random_search nsga2 cbo
do
    echo ./run.sh -s $start -d $dx -h $opt_name
    ./run.sh -s $start -d $dx -h $opt_name
done

echo `date '+%y/%m/%d %H:%M:%S'`
