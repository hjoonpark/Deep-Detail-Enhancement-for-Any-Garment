clear;
# mkdir output_ours/test
# nohup python teGeometry_Opt.py --folder "anger" --start-idx 0 --end-idx 52 --cuda-idx 0 > output_ours/test/log_anger1.txt &
# nohup python teGeometry_Opt.py --folder "anger" --start-idx 52 --end-idx 104 --cuda-idx 0 > output_ours/test/log_anger2.txt &
# nohup python teGeometry_Opt.py --folder "anger" --start-idx 104 --end-idx 156 --cuda-idx 0 > output_ours/test/log_anger3.txt &
# nohup python teGeometry_Opt.py --folder "anger" --start-idx 156 --end-idx 220 --cuda-idx 0 > output_ours/test/log_anger4.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 0 --end-idx 52 --cuda-idx 1 > output_ours/test/log_fear1.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 52 --end-idx 104 --cuda-idx 1 > output_ours/test/log_fear2.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 104 --end-idx 156 --cuda-idx 1 > output_ours/test/log_fear3.txt &
nohup python teGeometry_Opt.py --folder "fear" --start-idx 156 --end-idx 220 --cuda-idx 1 > output_ours/test/log_fear4.txt &
#  python teGeometry_Opt.py --folder "anger" --start-idx 70 --end-idx 80 --cuda-idx 0 
