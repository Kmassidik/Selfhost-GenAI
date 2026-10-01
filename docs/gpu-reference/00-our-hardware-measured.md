# Our GPU Hardware — Measured (the real silicon we build for)

_Mon Aug 31 08:25:58 PM WIB 2026 on dalang-Z9PE-D8-WS_

## Per-GPU properties (torch.cuda.get_device_properties)
```
device: NVIDIA GeForce RTX 3060 Ti | count: 3
  major = 8
  minor = 6
  total_memory = 8221.1 MB
  multi_processor_count = 38
  max_threads_per_multi_processor = 1536
  warp_size = 32
  regs_per_multiprocessor = 65536
  shared_memory_per_multiprocessor = 0.1 MB
  -> SMs x threads = 38 SMs, cc 8.6
```
## Clocks / power / PCIe link
```
index, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W], pcie.link.gen.max, pcie.link.width.max, pcie.link.gen.current, pcie.link.width.current
0, 8192 MiB, 2100 MHz, 7001 MHz, 200.00 W, 3, 16, 2, 16
1, 8192 MiB, 2100 MHz, 7001 MHz, 200.00 W, 3, 16, 1, 16
2, 8192 MiB, 2100 MHz, 7001 MHz, 200.00 W, 3, 16, 1, 16
```
## Inter-GPU topology (no NVLink expected)
```
	[4mGPU0	GPU1	GPU2	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	SYS	0-31	0-1		N/A
GPU1	SYS	 X 	PHB	0-31	0-1		N/A
GPU2	SYS	PHB	 X 	0-31	0-1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
## NUMA / CPU
```
CPU(s):                                  32
On-line CPU(s) list:                     0-31
Model name:                              Intel(R) Xeon(R) CPU E5-2665 0 @ 2.40GHz
BIOS Model name:                         Intel(R) Xeon(R) CPU E5-2665 0 @ 2.40GHz To Be Filled By O.E.M. CPU @ 1.2GHz
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               2
CPU(s) scaling MHz:                      47%
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-7,16-23
NUMA node1 CPU(s):                       8-15,24-31
```
