cd ..
    nohup python nsimplex_seed_analyse.py --n 5 --structure_path 5simplex/5simplex --save_path  5simplexSeed/5simplex --gaussian_variance 0.8 --reliability_type GK &> /dev/null &
    nohup python nsimplex_seed_analyse.py --n 5 --structure_path 5simplex/5simplex --save_path  5simplexSeed/5simplex --gaussian_variance 0.5 --reliability_type GK &> /dev/null &
    nohup python nsimplex_seed_analyse.py --n 5 --structure_path 5simplex/5simplex --save_path  5simplexSeed/5simplex --gaussian_variance 1 --reliability_type GK &> /dev/null &
    nohup python nsimplex_seed_analyse.py --n 5 --structure_path 5simplex/5simplex --save_path  5simplexSeed/5simplex  --shift 2 --reliability_type CC &> /dev/null &
    nohup python nsimplex_seed_analyse.py --n 5 --structure_path 5simplex/5simplex --save_path  5simplexSeed/5simplex  --shift 3 --reliability_type CC &> /dev/null &
cd bash_scripts
