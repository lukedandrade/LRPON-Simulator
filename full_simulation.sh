for n_onus in 5 10 15 20
do
    #for model in ols ridge lasso mlp
    for model in ols
    do
        python l-sim_w_parser.py ipact_pred -O $n_onus -M $model &
    done
    sleep 100
done