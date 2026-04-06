import subprocess
import sys
import numpy as np
import os
import re
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm


turn_omp = True
omp_nums = [1,2,4,8,16,32,64]
KEY_COLS = ['field', 'type', 'threads']
def upsert_csv(csv_path, new_df):
    if not os.path.exists(csv_path):
        new_df.to_csv(csv_path, index=False)
        return

    old_df = pd.read_csv(csv_path)
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=KEY_COLS,
        keep='last'
    )

    merged.to_csv(csv_path, index=False)

def append_row(csv_file, row):
    with open(csv_file, "a", newline="") as f:  # "a" = append
        writer = csv.writer(f)
        writer.writerow(row)

def ceil_power_of_2(n):
    n = int(n)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def compute_range(input_file, ddtype):
    num_elements = np.prod(shape)
    original = np.fromfile(input_file, dtype=ddtype, count=num_elements)
    return original.max() - original.min()

def compute_psnr(input_file, decompressed_file, ddtype, shape):
    num_elements = np.prod(shape)
    original = np.fromfile(input_file, dtype=ddtype, count=num_elements).reshape(shape)
    reconstructed = np.fromfile(decompressed_file, dtype=ddtype, count=num_elements).reshape(shape)
    dtype = np.longdouble
    diff = original.astype(dtype) - reconstructed.astype(dtype)
    mse = np.mean(np.square(diff, dtype=dtype), dtype=dtype)
    data_range = original.max() - original.min()
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    rmse = np.sqrt(mse)
    nrmse = rmse / data_range
    diff = np.abs(original - reconstructed)
    max_diff = diff.max()
    # print(f"  Max_E = {max_diff}\n  Max_RE = {max_diff / data_range}\n  PSNR = {psnr}")
    return max_diff, max_diff / data_range, rmse, nrmse, psnr

def run_zfp(shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".zfp"
    decompressed_file = input_file + ".zfp.out"
    # os.environ["OMP_NUM_THREADS"] =  str(nums)
    cmd = [
        "../zfp/bin/zfp", data_type_para,
        "-i", input_file,
        "-z", compressed_file,
        "-o", decompressed_file, 
        "-a", str(e), '-s',
        '-' + str(len(shape)), *[str(s) for s in (shape)], '-x', f'omp={nums},'
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return compressed_file, decompressed_file, result

def run_sz3(shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".sz"
    decompressed_file = input_file + ".sz.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    config = ['-c', 'sz3config']
    if nums == 1:
        config = []
    cmd = [
        "../SZ3/build/bin/sz3", data_type_para,
        "-i", input_file,
        "-z", compressed_file,
        "-o", decompressed_file, 
        "-M", mode, str(e),
        "-a", 
        '-' + str(len(shape)), *[str(s) for s in (shape)]
    ] + config
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(cmd)
    return compressed_file, decompressed_file, result


def run_sz3_para(cmp, shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".sz"
    decompressed_file = input_file + ".sz.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    cmp_dir = f"../{cmp}/build/bin/sz3"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir, data_type_para,
        "-i", input_file,
        "-z", compressed_file,
        "-o", decompressed_file, 
        "-M", mode, str(e),
        "-a",
        '-' + str(len(shape)), *[str(s) for s in (shape)]
    ] 
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(cmd)
    return compressed_file, decompressed_file, result

def run_szo_para(cmp, shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".sz"
    decompressed_file = input_file + ".sz.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    cmp_dir = f"../{cmp}/build/bin/szo"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir, data_type_para,
        "-i", input_file,
        "-z", compressed_file,
        "-o", decompressed_file, 
        "-M", mode, str(e),
        "-a",
        '-' + str(len(shape)), *[str(s) for s in (shape)]
    ] 
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(cmd)
    return compressed_file, decompressed_file, result

def run_sleek_para(cmp, shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'single'
    if(data_type == "float"):
        data_type_para = "single"
    else:
        data_type_para = "double"
    compressed_file = input_file + ".sleek"
    decompressed_file = input_file + ".sleek.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    cmp_dir = f"../{cmp}/sleek_{data_type_para}_compressor_lossy_cpu"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir,
        input_file,
        compressed_file, 
        str(e),
        "y",
    ] 
    cmp_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    decmp_dir = f"../{cmp}/sleek_{data_type_para}_decompressor_lossy_cpu"
    decmd = [
        "taskset", "-c", cpus,
        decmp_dir,
        compressed_file, 
        decompressed_file,
        input_file,
        str(e),
        "y",
    ] 
    dec_result = subprocess.run(decmd, check=True, capture_output=True, text=True)
    return compressed_file, decompressed_file, [cmp_result, dec_result]

def run_pfpl_para(cmp, shape, data_type, input_file, e, mode = 'abs', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "f32"
    else:
        data_type_para = "f64"
    compressed_file = input_file + ".pfpl"
    decompressed_file = input_file + ".pfpl.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    t = ['serial', 'ser']
    if nums == 1:
        t = ['serial', 'ser']
    else :
        t = ['openmp', 'omp']
    cmp_dir = f"../{cmp}/bin/{data_type_para}/{t[0]}/{mode}_compress_{t[1]}"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir,
        input_file,
        compressed_file, 
        str(e),
    ] 
    cmp_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    decmp_dir = f"../{cmp}/bin/{data_type_para}/{t[0]}/{mode}_decompress_{t[1]}"
    decmd = [
        "taskset", "-c", cpus,
        decmp_dir,
        compressed_file, 
        decompressed_file,
    ] 
    dec_result = subprocess.run(decmd, check=True, capture_output=True, text=True)
    return compressed_file, decompressed_file, [cmp_result, dec_result]

def run_sperr_para(cmp, shape, data_type, input_file, e, mode = 'REL', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = '32'
    if(data_type == "float"):
        data_type_para = "32"
        decom_para = "--decomp_f"
    else:
        data_type_para = "64"
        decom_para = "--decomp_d"
    compressed_file = input_file + ".sperr"
    decompressed_file = input_file + ".sperr.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    cmp_dir = f"../SPERR/build/bin/sperr{len(shape)}d"

    chunks_config = []
    omp_config = [ "--omp", str(nums)]
    if len(shape) !=3 :
        omp_config = []
    else:     
        i = nums
        k = 2
        chunks = [ceil_power_of_2((shape[0])) , ceil_power_of_2((shape[1])), ceil_power_of_2((shape[2]))]
        chunks[0] = chunks[0] if chunks[0] < 256 else 256
        chunks[1] = chunks[1] if chunks[1] < 256 else 256
        chunks[2] = chunks[2] if chunks[2] < 256 else 256
        while i > 1:
            chunks[2] =  chunks[2] / k
            i = i / 2
            if i > 1:
                chunks[1] = chunks[1] / k
                i = i / 2
            else:
                break
            if i > 1:
                chunks[0] = chunks[0] / k
                i = i / 2
            else:
                break
            k = k * 2
        chunks_config = [ "--chunks", *[str(int(s)) for s in (chunks)]]

    cmd_cmp = [
        "taskset", "-c", cpus,
        cmp_dir, input_file,
        "--ftype", data_type_para,
        "-c", "--bitstream", compressed_file,
        "--pwe", str(e),
        '--dims', *[str(s) for s in (shape)], 
    ] + omp_config + chunks_config
    cmp_result = subprocess.run(cmd_cmp, check=True, capture_output=True, text=True)
    cmd_dec = [
        "taskset", "-c", cpus,
        cmp_dir, compressed_file,
        "--ftype", data_type_para,
        "-d", decom_para, decompressed_file,
        "--pwe", str(e),
        '--dims', *[str(s) for s in (shape)], 
    ] + omp_config
    dec_result = subprocess.run(cmd_dec, check=True, capture_output=True, text=True)

    return compressed_file, decompressed_file, [cmp_result, dec_result]

def run_szx_para(cmp, shape, data_type, input_file, e, mode = 'ABS', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = '-f'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".szx"
    decompressed_file = input_file + ".szx.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    t = '1' if nums == 1 else '4'
    cmp_dir = f"../{cmp}/bin/szx"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir, "-z",
        data_type_para, "-i", input_file, 
        '-' + str(len(shape)), *[str(s) for s in (shape)],
       "-M", mode, '-' + mode[0], str(e),
       '-m', t
    ] 
    cmp_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    decmd = [
        "taskset", "-c", cpus,
        cmp_dir, "-x",
        data_type_para, "-s", compressed_file, 
        '-' + str(len(shape)), *[str(s) for s in (shape)],
        "-M", mode, '-' + mode[0], str(e),
        '-m', t
    ] 
    dec_result = subprocess.run(decmd, check=True, capture_output=True, text=True)
    return compressed_file, decompressed_file, [cmp_result, dec_result]

def run_tthresh_para(cmp, shape, data_type, input_file, e, mode = 'ABS', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = '-f'
    if(data_type == "float"):
        data_type_para = "float"
    else:
        data_type_para = "double"
    compressed_file = input_file + ".tthresh"
    decompressed_file = input_file + ".tthresh.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    cmp_dir = f"../{cmp}/build/tthresh"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir, "-t", data_type_para,
        "-i", input_file, '-c', compressed_file, '-o', decompressed_file,
       "-e", f"{float(e):.10f}".rstrip('0').rstrip('.'),
        '-s', *[str(s) for s in (shape)],
    ] 
    if len(shape) == 2:
        cmd += ['1']
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return compressed_file, decompressed_file, result

def run_mgard_para(cmp, shape, data_type, input_file, e, mode = 'abs', nums = 1):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = '-f'
    if(data_type == "float"):
        data_type_para = "s"
    else:
        data_type_para = "d"
    compressed_file = input_file + ".mgard"
    decompressed_file = input_file + ".mgard.out"
    os.environ["OMP_NUM_THREADS"] =  str(nums)
    os.environ["OMP_PROC_BIND"] = "close"
    cpus = f"0-{nums-1}" 
    t = 'serial' if nums == 1 else 'openmp'
    cmp_dir = f"../{cmp}/build-openmp-cpu/mgard/bin/mgard-x"
    cmd = [
        "taskset", "-c", cpus,
        cmp_dir, "-z", '-dt', data_type_para,
        "-i", input_file, "-o", compressed_file,
        '-dim', str(len(shape)), *[str(s) for s in (shape)][::-1],
       "-em", mode, '-e',str(e),
       '-s', 'inf', '-d', t,
       '-v', '2',
       '-l', 'huffman-zstd',
    ] 
    cmp_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    decmd = [
        "taskset", "-c", cpus,
        cmp_dir, "-x",
        "-i", compressed_file, 
        "-o", decompressed_file, '-d', t, '-v', '2'
    ] 
    dec_result = subprocess.run(decmd, check=True, capture_output=True, text=True)
    print(cmp_result.stdout,dec_result.stdout )
    return compressed_file, decompressed_file, [cmp_result, dec_result]



def run_compressor(shape, data_type, data_path, compressor):
    global omp_nums
    ddtype = np.float64 if data_type == "double" else np.float32
    errors = ['1E-1', '1E-2','1E-3', '1E-4', '1E-5', '1E-6', '1E-7', '1E-8'] if data_type == "double" else ['1E-1', '1E-2','1E-3', '1E-4', '1E-5']
    # errors = ['1E-1']
    byte_num = 8 if data_type == "double" else 4
    directory = Path(data_path)
    target_files = [str(p) for p in directory.iterdir() if p.suffix in (".f32", ".d64")]

    dataset = sys.argv[1]

    if turn_omp is True:
        omp = "_omp"
        header = ["dataset", "field", "type", "threads"] + [str(e) for e in errors] 
        # omp_nums = [1,2,4,8,16,32,64]
    else:
        # omp_nums = [1]
        omp = ""
        header = ["dataset", "field", "type"] + [str(e) for e in errors]
    cmp_size_csv = './experiment_csv/cmp_size/'  + '/' + dataset + omp + '.csv'
    ratio_csv = './experiment_csv/ratio/'  + '/' + dataset + omp + '.csv'
    comth_csv = './experiment_csv/cmpth/'  + '/' + dataset  + omp + '.csv'
    decth_csv = './experiment_csv/decth/' + '/' + dataset + omp + '.csv'

    psnr_csv = './experiment_csv/psnr/'  + '/' + dataset + omp + '.csv'
    nrmse_csv = './experiment_csv/nrmse/'  + '/' + dataset + omp + '.csv'
    maxre_csv = './experiment_csv/maxre/'  + '/' + dataset  + omp + '.csv'
    maxe_csv = './experiment_csv/maxe/'  + '/' + dataset  + omp + '.csv'

    odata_size = []
    odata_ratio = []
    odata_cmpth = []
    odata_decth = []

    odata_psnr = []
    odata_nrmse = []
    odata_maxre = []
    odata_maxe = []
    # odata_len_hit = {i: [] for i in range(1, 11)}
    print(f"Running {compressor} on dataset {sys.argv[1]}...")
    data_num = np.prod(shape)
    for input_file in target_files:
        # if input_file.find("pressure.d64") == -1:
        #     continue
        # data = np.memmap(input_file, dtype=ddtype, mode="r")
        # value_range = data.max() - data.min()
        print(f"Processing file: {input_file} compressor: {compressor}")
        filename = os.path.basename(input_file).replace('.f32', '').replace('.d64', '')
        temp_size = [dataset, filename, compressor]
        temp_ratio = [dataset, filename, compressor]
        temp_cmpth = [dataset, filename, compressor]
        temp_decth = [dataset, filename, compressor]

        temp_psnr = [dataset, filename, compressor]
        temp_nrmse = [dataset, filename, compressor]
        temp_maxre = [dataset, filename, compressor]
        temp_maxe = [dataset, filename, compressor]
        if filename in ['SFCLDLIQ_1_1800_3600', 'SFCLDICE_1_1800_3600','ODV_dust3_1_1800_3600','ODV_dust4_1_1800_3600','ODV_sulf_1_1800_3600',
                        'ODV_ocar1_1_1800_3600','ODV_bcar1_1_1800_3600','ODV_SSLTC_1_1800_3600', 'ODV_bcar2_1_1800_3600','ODV_ocar2_1_1800_3600','ODV_dust2_1_1800_3600','ODV_dust1_1_1800_3600',
                        'AEROD_v_1_1800_3600','ODV_SSLTA_1_1800_3600']:
            continue
        # if filename in ['SFCLDLIQ_1_1800_3600', 'SFCLDICE_1_1800_3600']:
        #     continue
        if compressor == 'SZ3' and (dataset == 'Hurricane' or dataset == 'SCALE'):
            omp_nums = [1,2,4,8,16,32]
        else :
            omp_nums = [1,2,4,8,16,32,64]
        for nums in omp_nums:
            print(f"Processing file: {input_file}, threads: {nums}")
            temp_size = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_ratio = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_cmpth = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_decth = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            
            temp_psnr = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_maxre = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_maxe = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            temp_nrmse = [dataset, filename, compressor, nums] if turn_omp is True else [dataset, filename, compressor]
            # temp_len_hit = {i: [dataset, filename, compressor, nums] if turn_omp else [dataset, filename, compressor] for i in range(1, 11)}
            error_mode = 'ABS'
            if compressor == 'SPERR' or compressor == 'ZFP' and compressor != "SLEEK":
                error_mode = 'ABS'
            for e in tqdm(errors):
                if error_mode == 'ABS' and compressor != "SLEEK" and compressor != "tthresh":
                    e = float(e) * compute_range(input_file, ddtype)
                if compressor.find("SZ3") != -1 or compressor.find("SZo") != -1:    
                    if compressor == "SZ3.3" :
                        compressed_file, decompressed_file, result = run_sz3(shape, data_type, input_file,  e, error_mode, nums)
                    elif compressor == "SZ3":
                        compressed_file, decompressed_file, result = run_sz3(shape, data_type, input_file,  e, error_mode, nums)
                    elif compressor == "SZo":
                        compressed_file, decompressed_file, result = run_szo_para("SZo", shape, data_type, input_file,  e, error_mode, nums)
                    else :
                        compressed_file, decompressed_file, result = run_sz3(shape, data_type, input_file,  e, error_mode, nums)
                    
                    ratio = re.search(r"compression ratio =\s*([0-9.+Ee-]+)", result.stdout).group(1)
                    dec_th = float(re.search(r"decompression time =.*?(\d+\.\d+)", result.stdout).group(1))
                    cmp_th = float(re.search(r"(?<!de)compression time =.*?(\d+\.\d+)", result.stdout).group(1))
                    
                    temp_ratio += [float(ratio)]
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    cmp_size = os.path.getsize(compressed_file)
                    #print(byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th), byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th))
                    # maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    psnr = re.search(r"PSNR\s*=\s*([0-9.+Ee-]+)", result.stdout).group(1)
                    nrmse = re.search(r"NRMSE\s*=\s*([0-9.+Ee-]+)", result.stdout).group(1)
                    maxre = re.search(r"Max relative error = ([0-9.+Ee-]+)", result.stdout).group(1)
                    maxe = re.search(r"Max absolute error = ([0-9.+Ee-]+)", result.stdout).group(1)
                    # print(cmp_size)
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
    
                    # pattern = r"lenCount\[(\d+)\]\s*=\s*(\d+)\s*hit ratio:\s*([0-9\.eE+-]+)"
                    # matches = re.findall(pattern, result.stdout)
                    # len_counts = {}
                    # hit_ratios = {}
                    # for n, count, ratio in matches:
                    #     n = int(n)
                    #     len_counts[n] = int(count)
                    #     hit_ratios[n] = float(ratio)
                    # for i in range(1, 11):
                    #     temp_len_hit[i].append(hit_ratios.get(i, 0.0))

                elif compressor == 'ZFP':
                    if error_mode != 'ABS':
                        print(f"Error Mode in {compressor} not supported.")
                        sys.exit(1)
                    compressed_file, decompressed_file, result = run_zfp(shape, data_type, input_file,  e, error_mode, nums)
                    ratio = re.search(r"ratio=\s*([0-9.+Ee-]+)", result.stderr).group(1)
                    cmp_th = float(re.search(r"Compression_time=.*?(\d+\.\d+)", result.stdout).group(1))
                    dec_th = float(re.search(r"Decompression_time=.*?(\d+\.\d+)", result.stdout).group(1))
                    
                    temp_ratio += [float(ratio)]
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    cmp_size = os.path.getsize(compressed_file)
                    #print(byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th), byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th))
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    # psnr = re.search(r"PSNR\s*=\s*([0-9.+Ee-]+)", result.stdout).group(1)
                    # nrmse = re.search(r"NRMSE\s*=\s*([0-9.+Ee-]+)", result.stdout).group(1)
                    # maxre = re.search(r"Max relative error = ([0-9.+Ee-]+)", result.stdout).group(1)
                    # maxe = re.search(r"Max absolute error = ([0-9.+Ee-]+)", result.stdout).group(1)
                    # print(cmp_size)
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                elif compressor == 'tthresh':
                    compressed_file, decompressed_file, result = run_tthresh_para('tthresh', shape, data_type, input_file,  e, error_mode, nums)
                    
                    cmp_th = float(re.search(r"Compression time =.*?(\d+\.\d+)", result.stdout).group(1))
                    dec_th = float(re.search(r"Decompression time =.*?(\d+\.\d+)", result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape) 

                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]

                elif compressor == 'SPERR':
                    if error_mode != 'ABS':
                        print(f"Error Mode in {compressor} not supported.")
                        sys.exit(1)
                    compressed_file, decompressed_file, [cmp_result, dec_result] = run_sperr_para("SPERR", shape, data_type, input_file,  e, error_mode, nums)
                    cmp_th = float(re.search(r"Compression time =.*?(\d+\.\d+)", cmp_result.stdout).group(1))
                    dec_th = float(re.search(r"Decompression time =.*?(\d+\.\d+)", dec_result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape) 
                    # print(cmp_th)
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                elif compressor == 'SLEEK':
                    compressed_file, decompressed_file, [cmp_result, dec_result] = run_sleek_para('SLEEK', shape, data_type, input_file,  e, error_mode, nums)
                    cmp_th = float(re.search(r"encoding time:.*?(\d+\.\d+)", cmp_result.stdout).group(1))
                    dec_th = float(re.search(r"decoding time:.*?(\d+\.\d+)", dec_result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                elif compressor == 'PFPL':
                    compressed_file, decompressed_file, [cmp_result, dec_result] = run_pfpl_para('PFPL',shape, data_type, input_file,  e, 'abs', nums)
                    cmp_th = float(re.search(r"lc comp ecltime,.*?(\d+\.\d+)", cmp_result.stdout).group(1))
                    dec_th = float(re.search(r"lc decomp ecltime,.*?(\d+\.\d+)", dec_result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                elif compressor == 'SZx':
                    compressed_file, decompressed_file, [cmp_result, dec_result] = run_szx_para('SZx',shape, data_type, input_file,  e, error_mode, nums)
                    cmp_th = float(re.search(r"compression time =.*?(\d+\.\d+)", cmp_result.stdout).group(1))
                    dec_th = float(re.search(r"decompression time =.*?(\d+\.\d+)", dec_result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                elif compressor == 'MGARD':
                    compressed_file, decompressed_file, [cmp_result, dec_result] = run_mgard_para('MGARD',shape, data_type, input_file,  e, 'abs', nums)
                    cmp_th = float(re.search(r"High-level compression time: .*?(\d+\.\d+)", cmp_result.stdout).group(1))
                    dec_th = float(re.search(r"High-level decompression time: .*?(\d+\.\d+)", dec_result.stdout).group(1))
                    cmp_size = os.path.getsize(compressed_file)
                    maxe, maxre, rmse, nrmse, psnr = compute_psnr(input_file, decompressed_file, ddtype, shape)
                    temp_cmpth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(cmp_th)]
                    temp_decth += [byte_num * data_num / 1024 / 1024 / 1024 /  float(dec_th)]
                    temp_ratio += [byte_num * data_num / cmp_size]
                    temp_size += [cmp_size]
                    temp_maxe += [maxe]
                    temp_maxre += [maxre]
                    # temp_rmse += [rmse]
                    temp_nrmse += [nrmse]
                    temp_psnr += [psnr]
                else:
                    print(f"Compressor {compressor} not supported.")
                    sys.exit(1)
            os.remove(compressed_file)
            os.remove(decompressed_file)
            odata_ratio.append(temp_ratio)
            odata_cmpth.append(temp_cmpth)
            odata_decth.append(temp_decth)
            odata_size.append(temp_size)
            odata_psnr.append(temp_psnr)
            odata_nrmse.append(temp_nrmse)
            odata_maxre.append(temp_maxre)
            odata_maxe.append(temp_maxe)
            # print(odata_cmpth)
            # for i in range(1, 11):
            #     odata_len_hit[i].append(temp_len_hit[i])
            # print(odata_ratio)

    odata_ratio=np.array(odata_ratio).reshape(-1,len(header)).T
    odata_cmpth=np.array(odata_cmpth).reshape(-1,len(header)).T
    odata_decth=np.array(odata_decth).reshape(-1,len(header)).T
    odata_size=np.array(odata_size).reshape(-1,len(header)).T

    odata_psnr=np.array(odata_psnr).reshape(-1,len(header)).T
    odata_nrmse=np.array(odata_nrmse).reshape(-1,len(header)).T
    odata_maxre=np.array(odata_maxre).reshape(-1,len(header)).T
    odata_maxe=np.array(odata_maxe).reshape(-1,len(header)).T

    # odata_len_hit_np = {}

    # for i in range(1, 11):
    #     odata_len_hit_np[i] = np.array(odata_len_hit[i]).reshape(-1, len(header)).T

    for file_csv in [ratio_csv, comth_csv, decth_csv, cmp_size_csv, psnr_csv, maxre_csv, maxe_csv, nrmse_csv]:
        if not os.path.exists(file_csv):
            continue
        #df = pd.read_csv(file_csv)
        #df = df[(df['dataset'] != dataset) | (df['type'] != compressor)]
        #df.to_csv(file_csv, index=False)

    w_mode = 'a'
    hearder_mode = (not os.path.exists(ratio_csv)) or w_mode == 'w'
    pd.DataFrame(dict(zip(header, odata_ratio))).to_csv(ratio_csv, mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_cmpth))).to_csv(comth_csv, mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_decth))).to_csv(decth_csv,mode=w_mode, header=hearder_mode, index=False)

    pd.DataFrame(dict(zip(header, odata_size))).to_csv(cmp_size_csv,mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_psnr))).to_csv(psnr_csv, mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_maxre))).to_csv(maxre_csv, mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_maxe))).to_csv(maxe_csv, mode=w_mode, header=hearder_mode, index=False)
    pd.DataFrame(dict(zip(header, odata_nrmse))).to_csv(nrmse_csv, mode=w_mode, header=hearder_mode, index=False)
    # # for i in range(1, 11):
    #     len_csv = f'./experiment_csv/len_hit/len{i}_{dataset}{omp}.csv'
    
    #     pd.DataFrame(dict(zip(header, odata_len_hit_np[i]))).to_csv(len_csv, mode=w_mode, header='w', index=False)


def test_compressor(shape, data_type, data_path):
    #run_compressor(shape, data_type, data_path, 'ZFP')
    #run_compressor(shape, data_type, data_path, 'PFPL')
    #run_compressor(shape, data_type, data_path, 'SZo')
    #run_compressor(shape, data_type, data_path, 'SZ3')
    #run_compressor(shape, data_type, data_path, 'SPERR')
    run_compressor(shape, data_type, data_path, 'MGARD')

if __name__ == "__main__":
    # input_file = sys.argv[2]
    data_type = "float"
    dim = 3
    shape = [384, 384, 256]
    data_path = "/data1/lb/sdrbench/" + sys.argv[1]
    if sys.argv[1] == "Miranda" or sys.argv[1] == "Miranda-float":
        shape = [384, 384, 256]
        if sys.argv[1] == "Miranda":
            data_type = "double"
    elif sys.argv[1] == "CH4":
        data_type = "double"
        shape = [500, 500, 500]
        bit_rate = [1,1,1,4,3,6,4,5,5]
    elif sys.argv[1] == "Hurricane":
        shape = [500, 500, 100]
        omp_nums = [1,2,4,8,16,32,64]
    elif sys.argv[1] == "SCALE":
        shape = [1200, 1200, 98]
        omp_nums = [1,2,4,8,16,32,64]
    elif sys.argv[1] == "NYX" or sys.argv[1] == "JHTDB":
        shape = [512, 512, 512]
    elif sys.argv[1] == "CESM":
        shape = [3600, 1800]
        dim = 2
    elif sys.argv[1] == "EXAFEL":
        shape = [388, 5837120]
        dim = 2
    elif sys.argv[1] == "tomobank":
        shape = [2048, 2048]
        dim = 2
    test_compressor(shape, data_type, data_path)
