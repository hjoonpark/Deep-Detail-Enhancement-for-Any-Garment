clear;
nohup python teGeometry_Opt.py --folder "fear" --start-idx 0 --end-idx 52 --cuda-idx 0 > output_ours/test/log_euc5.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 52 --end-idx 104 --cuda-idx 0 > output_ours/test/log_euc6.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 104 --end-idx 156 --cuda-idx 1 > output_ours/test/log_euc7.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 156 --end-idx 220 --cuda-idx 1 > output_ours/test/log_euc8.txt &