cd ..
for j in 10 20 30 10k 20k 30k
do
	for i in 3 4 5 6
	do
		nohup python nsimplex-analyse.py --structure_path ${i}simplex/${i}simplex --save_path ${i}simplex_${j}/${i}simplex --n $i --time 100 --binsize 3 &> /dev/null &
	done
done
cd bash_scripts
