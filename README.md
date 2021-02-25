# hashing

To run the code use command below:
In the command below just the dataset name is useful. Keep the rest of the parameters the same.

python learned_hash.py <dataset_name> <output_file> <some number keep 1000> <some number keep 1000>
python learned_hash.py fb temp.csv 1000 1000 

If datasets not avaialble you can run sequential and lognormal distributions:
python learned_hash.py seq temp.csv 1000 1000
python learned_hash.py log_normal temp.csv 1000 1000

Benchmark_binary_stuff function is useful for hashing.
get_slope_bias function is where the slope and bias is calculated.
