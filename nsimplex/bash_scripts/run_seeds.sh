cd ..
for j in 20
do
        for i in $(seq 1 10)
        do
                nohup python nsimplex_sim_runner.py --structure_path 5simplex/5simplex --save_path 5simplexSeed/5simplex_${j}_seed${i}/5simplex --stimulus_type dc --stimulus_strength ${j} --stimulus_length 190 --stimulus_start 5 --time 200 --binsize 3 --p_transmit 0.8 --noise_strength 0. --n 5 --seed ${i} &> /dev/null &
		pid=$!
		wait $pid
        done
done
cd bash_scripts
