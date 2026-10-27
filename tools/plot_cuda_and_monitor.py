from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as h: return list(csv.DictReader(h))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cuda-csv', required=True); p.add_argument('--monitor-csv', required=True); p.add_argument('--output-dir', required=True); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    data=rows(a.cuda_csv)
    cases=sorted({int(row['case']) for row in data})
    fig,ax=plt.subplots(figsize=(9,5),dpi=160)
    for index, backend in enumerate(('numpy','cupy','cuda_c')):
        selected={int(row['case']):float(row['mean_ms']) for row in data if row['backend']==backend}
        ax.bar([case+(index-1)*0.26 for case in cases],[selected[case] for case in cases],width=0.25,label=backend)
    ax.set_xticks(cases,[f"Case {case}" for case in cases])
    ax.set_yscale('log')
    ax.set_ylabel('Mean kernel/compute time (ms, log scale)')
    ax.set_title('NumPy / CuPy / CUDA C convolution by input size')
    ax.legend()
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(out/'numpy_cupy_cuda_time.png'); plt.close(fig)
    m=rows(a.monitor_csv); t=list(range(len(m))); fig,axs=plt.subplots(2,1,figsize=(9,6),dpi=160,sharex=True); axs[0].plot(t,[float(x['gpu_util_pct']) for x in m],label='GPU util %'); axs[0].plot(t,[float(x['memory_used_mib'])/float(x['memory_total_mib'])*100 for x in m],label='VRAM used %'); axs[0].set_ylabel('%'); axs[0].legend(); axs[0].grid(alpha=.2); axs[1].plot(t,[float(x['temperature_c']) for x in m],label='Temp C'); axs[1].plot(t,[float(x['power_w']) for x in m],label='Power W'); axs[1].set_xlabel('Sample'); axs[1].legend(); axs[1].grid(alpha=.2); fig.suptitle('GPU resource monitor during optimized PS training'); fig.tight_layout(); fig.savefig(out/'gpu_resource_monitor.png'); plt.close(fig)

if __name__=='__main__': main()
