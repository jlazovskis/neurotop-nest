cd ..
for j in 10 20 30
do
        for i in 3 4 5 6
        do
                nohup python nsimplex_sim_runner.py --structure_path ${i}simplex/${i}simplex --save_path ${i}simplex_${j}/$ --stimulus_type dc --stimulus_strength ${j} --stimulus_length 90 --stimulus_start 5 --time 100 &> /dev/null &
        done
        for i in 3 4 5 6
        do
                nohup python nsimplex_sim_runner.py --structure_path ${i}simplex/${i}simplex --save_path ${i}simplex_${j}k/$ --stimulus_type poisson --stimulus_strength ${j}000 --stimulus_length 90 --stimulus_start 5 --time 100 &> /dev/null &
        done
done
cd bash_scripts
