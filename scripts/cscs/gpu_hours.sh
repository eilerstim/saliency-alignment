#!/bin/bash
# GPU-hours and node-hours consumed by this user's SLURM jobs, from sacct.
#
#   scripts/cscs/gpu_hours.sh [START_DATE] [JOB_NAME_FILTER]
#
# START_DATE defaults to 7 days ago (YYYY-MM-DD). Running jobs count their
# elapsed time so far. GPU-hours are summed from the gres/gpu allocation of
# each job; node-hours from the node count (CSCS bills GH200 nodes, 4 GPUs
# per node). Job steps are excluded (-X) so nothing is double counted.
set -euo pipefail

START="${1:-$(date -d '7 days ago' +%F 2>/dev/null || date -v-7d +%F)}"
FILTER="${2:-}"

sacct -X -n -P --starttime "$START" \
    --format=JobID,JobName%40,State,ElapsedRaw,AllocTRES,NNodes \
| awk -F'|' -v filter="$FILTER" '
    {
        name = $2; state = $3; secs = $4 + 0; tres = $5; nodes = $6 + 0
        if (filter != "" && index(name, filter) == 0) next
        gpus = 0
        n = split(tres, parts, ",")
        for (i = 1; i <= n; i++) {
            if (parts[i] ~ /^gres\/gpu(:[^=]*)?=/) {
                split(parts[i], kv, "="); gpus = kv[2] + 0
            }
        }
        gpu_h[name] += secs * gpus / 3600.0
        node_h[name] += secs * nodes / 3600.0
        count[name] += 1
        if (state ~ /^(RUNNING|PENDING)/) active[name] += 1
        tg += secs * gpus / 3600.0; tn += secs * nodes / 3600.0; tc += 1
    }
    END {
        # sort key prefix: 0 = header, 1 = body (sorted by name), 2 = total
        printf "0|%-42s %5s %6s %10s %10s\n", "job name", "jobs", "active", "GPU-h", "node-h"
        for (k in gpu_h)
            printf "1|%-42s %5d %6d %10.1f %10.1f\n", k, count[k], active[k] + 0, gpu_h[k], node_h[k]
        printf "2|%-42s %5d %6s %10.1f %10.1f\n", "TOTAL since " "'"$START"'", tc, "", tg, tn
    }' | sort -t'|' -k1,1 -k2,2 | cut -d'|' -f2-
