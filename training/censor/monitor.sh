#!/bin/bash
# Black-box monitor for the shared Mac Studio: every 30 s, append memory, swap, GPU and disk
# state and the biggest processes (any user) to data/interim/monitor/monitor-YYYYMMDD.log, so a
# freeze like the one on 2026-10-30 (no OOM messages, system log just stops) leaves a trail.
# Keeps 14 days. Needs no sudo. Run detached: screen -dmS monitor training/censor/monitor.sh
cd "$(dirname "$0")/../.." || exit 1
DIR=data/interim/monitor
mkdir -p "$DIR"
while true; do
  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')"
    echo "free_level=$(sysctl -n kern.memorystatus_level)% $(sysctl -n vm.swapusage)"
    vm_stat | awk -F: '/Pages (free|active|inactive|wired down|occupied by compressor)/ {
      gsub(/[ .]/, "", $2); printf "%s=%.1fGB ", $1, $2 * 16384 / 1e9 } END { print "" }'
    ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1
    df -h / | awk 'NR == 2 { print "disk_free=" $4 }'
    ps -axo rss=,user=,pid=,etime=,comm= | sort -rn | head -6 |
      awk '{ printf "  %6.1fGB %-10s %6s %12s %s\n", $1 / 1e6, $2, $3, $4, $5 }'
  } >> "$DIR/monitor-$(date +%Y%m%d).log" 2>&1
  find "$DIR" -name 'monitor-*.log' -mtime +14 -delete
  sleep 30
done
