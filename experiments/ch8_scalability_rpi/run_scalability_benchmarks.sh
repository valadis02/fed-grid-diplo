#!/bin/bash
ALGORITHMS=("fedavg" "krum" "multi_krum" "tmean")
TOPOLOGIES=(5 10 20)
MAX_ROUNDS=10
TIMEOUT=600
OUTPUT_DIR="$HOME/fed-grid-diplo/results/scalability"
YML_DIR="$HOME/fed-grid-diplo/experiments_yml/rpi_scalability"

mkdir -p "$OUTPUT_DIR"

collect_metrics() {
    local container=$1
    local n_nodes=$2
    docker logs "$container" 2>&1 \
        | grep "\[AGG METRICS\]" \
        | grep "n_clients=${n_nodes}"
}

parse_and_save() {
    local algo=$1
    local n=$2
    local container=$3
    local outfile="$OUTPUT_DIR/metrics_${algo}_${n}.json"

    echo "[" > "$outfile"
    local first=1
    collect_metrics "$container" "$n" | while read line; do
        agg_time=$(echo "$line"   | grep -oP 'agg_time=\K[0-9.]+')
        total_time=$(echo "$line" | grep -oP 'total_time=\K[0-9.]+')
        peak_ram=$(echo "$line"   | grep -oP 'peak_ram=\K[0-9.]+')
        cpu_temp_b=$(echo "$line" | grep -oP 'cpu_temp_before=\K[0-9.]+')
        cpu_temp_a=$(echo "$line" | grep -oP 'cpu_temp_after=\K[0-9.]+')
        energy=$(echo "$line"     | grep -oP 'estimated_energy=\K[0-9e.+-]+')

        [ "$first" -eq 0 ] && echo "," >> "$outfile"
        first=0
        echo "  {\"strategy\":\"$algo\",\"topology\":$n,\"agg_time\":$agg_time,\"total_time\":$total_time,\"peak_ram\":$peak_ram,\"cpu_temp_before\":$cpu_temp_b,\"cpu_temp_after\":$cpu_temp_a,\"energy\":$energy}" >> "$outfile"
    done
    echo "]" >> "$outfile"

    local count=$(collect_metrics "$container" "$n" | wc -l)
    echo "  ✓ Αποθηκεύτηκαν $count μετρήσεις → $outfile"
}

run_experiment() {
    local algo=$1
    local n=$2
    local CONTAINER="scale_${algo}_${n}_fog_app"
    local YML="$YML_DIR/scale_${algo}_${n}_rpi.yml"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Algorithm : $algo | Topology : $n nodes"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    docker compose -f "$YML" down 2>/dev/null
    echo "  Εκκίνηση fog..."
    docker compose -f "$YML" up -d
    sleep 8

    echo ""
    echo "  >> Στον PC τρέξε:"
    echo "     docker-compose -f experiments_yml/pc_scalability/scale_${algo}_${n}_pc.yml up"
    echo ""
    echo "  Πάτα ENTER όταν ξεκινήσουν τα edges στον PC..."
    read

    echo "  Συλλογή metrics (max ${MAX_ROUNDS} rounds ή ${TIMEOUT}s)..."
    elapsed=0
    collected=0

    while [ $elapsed -lt $TIMEOUT ] && [ $collected -lt $MAX_ROUNDS ]; do
        collected=$(collect_metrics "$CONTAINER" "$n" | wc -l)
        echo "  Progress: $collected/$MAX_ROUNDS rounds (${elapsed}s elapsed)"
        if [ $collected -ge $MAX_ROUNDS ]; then
            echo "  ✓ Συλλέχθηκαν $MAX_ROUNDS μετρήσεις!"
            break
        fi
        sleep 15
        elapsed=$((elapsed + 15))
    done

    [ $elapsed -ge $TIMEOUT ] && echo "  ⚠ Timeout! Συλλέχθηκαν $collected μετρήσεις."

    parse_and_save "$algo" "$n" "$CONTAINER"

    echo "  Κατέβασμα fog..."
    docker compose -f "$YML" down

    echo ""
    echo "  >> Στον PC κάνε down τα edges:"
    echo "     docker-compose -f experiments_yml/pc_scalability/scale_${algo}_${n}_pc.yml down"
    echo ""
    echo "  Πάτα ENTER για να συνεχίσεις στον επόμενο συνδυασμό..."
    read
}

echo "============================================"
echo " Scalability Benchmark - RPi Orchestrator"
echo " Algorithms : ${ALGORITHMS[@]}"
echo " Topologies : ${TOPOLOGIES[@]}"
echo " Max rounds : $MAX_ROUNDS ή ${TIMEOUT}s"
echo " Output     : $OUTPUT_DIR"
echo "============================================"

for algo in "${ALGORITHMS[@]}"; do
    for n in "${TOPOLOGIES[@]}"; do
        run_experiment "$algo" "$n"
    done
done

echo ""
echo "============================================"
echo " ΟΛΟΚΛΗΡΩΘΗΚΕ!"
echo " Αποτελέσματα:"
for f in "$OUTPUT_DIR"/metrics_*.json; do
    count=$(grep -c "agg_time" "$f" 2>/dev/null || echo 0)
    echo "  $f: $count μετρήσεις"
done
echo "============================================"