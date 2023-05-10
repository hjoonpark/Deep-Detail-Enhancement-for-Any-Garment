clear;
nohup python teGeometry_Opt.py --folder "fear" --start-idx 0 --end-idx 110 --cuda-idx 0 > output_ours/test/log1.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 110 --end-idx 220 --cuda-idx 0 > output_ours/test/log2.txt &