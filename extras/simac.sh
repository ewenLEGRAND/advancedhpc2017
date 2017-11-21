#!/bin/bash

cpu=`sysctl -n machdep.cpu.brand_string`;
cores=`sysctl -n machdep.cpu.core_count`;
threads=`sysctl -n machdep.cpu.thread_count`;
mem=`sysctl -n hw.memsize`;
mems=$(($mem / 1024 / 1024));
#cols=`system_profiler SPDisplaysDataType | grep Resolution`;
storage=`diskutil list | grep 0: | awk '{ print $5, $3$4 }' | sed 's/[*]//'`;
vga=`system_profiler SPDisplaysDataType | grep Chipset | while read -e line; do echo $line | tail -c +16; done`
ssd=`diskutil list | grep SSD | head -n 1 | awk '{ print $6 }' | head -c 5`
nand_written_lba=`smartctl -A /dev/$ssd | grep "Total_LBAs_Written" | awk '{print $10}'`
nand_written_mb=`bc <<< "$nand_written_lba * 512 / 1048576"`
echo === SysInfo: $cpu \[$cores cores $threads threads\], $(($mems))MB RAM, $vga, $storage, SSDWritten for $ssd: ${nand_written_mb}MB;
