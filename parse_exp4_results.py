"""
parse_exp4_results.py — Πείραμα 4: AES-256-GCM Overhead Analysis
"""
import os
from statistics import mean, stdev

LOG_EDGE  = "logs_edge_exp4.txt"
LOG_FOG   = "logs_fog_exp4.txt"
LOG_CLOUD = "logs_cloud_exp4.txt"

def read_lines(path):
    if not os.path.exists(path):
        print(f"  [WARNING] Δεν βρέθηκε: {path}")
        return []
    with open(path, "rb") as f:
        bom = f.read(2)
    enc = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    with open(path, encoding=enc, errors="ignore") as f:
        return f.readlines()

def extract_value(line, key):
    """Εξάγει αριθμό μετά από key= π.χ. 'encrypt=0.008s' → 0.008"""
    try:
        idx = line.index(key + "=")
        start = idx + len(key) + 1
        end = start
        while end < len(line) and (line[end].isdigit() or line[end] == '.'):
            end += 1
        return float(line[start:end])
    except (ValueError, IndexError):
        return None

def stats(values):
    if not values:
        return
    print(f"    count : {len(values)}")
    print(f"    mean  : {mean(values)*1000:.3f} ms")
    if len(values) > 1:
        print(f"    std   : {stdev(values)*1000:.3f} ms")
    print(f"    min   : {min(values)*1000:.3f} ms")
    print(f"    max   : {max(values)*1000:.3f} ms")

def main():
    print("=" * 60)
    print("  Πείραμα 4: Ανάλυση AES-256-GCM Overhead")
    print("=" * 60)

    edge_lines  = read_lines(LOG_EDGE)
    fog_lines   = read_lines(LOG_FOG)
    cloud_lines = read_lines(LOG_CLOUD)

    # ── Edge ──────────────────────────────────────────────────────
    edge_enc, edge_ser, edge_send, payload_kb = [], [], [], []
    rounds = []

    for line in edge_lines:
        if "Edge: model encrypted" in line:
            v = extract_value(line, "encrypt")
            if v is not None: edge_enc.append(v)
        elif "Model sent to Fog" in line:
            s = extract_value(line, "ser")
            e = extract_value(line, "enc")
            n = extract_value(line, "send")
            p = extract_value(line, "payload")
            if s: edge_ser.append(s)
            if n: edge_send.append(n)
            if p: payload_kb.append(p)
        elif "SUMMARY" in line and "Round" in line:
            r = extract_value(line, "train")
            ram = extract_value(line, "RAM")
            enc = extract_value(line, "enc")
            snd = extract_value(line, "send")
            tot = extract_value(line, "total")
            # round number
            try:
                rn = int(line.split("[Round ")[1].split("]")[0])
            except:
                rn = len(rounds) + 1
            if r and tot:
                rounds.append({"round": rn, "train": r, "ram": ram or 0,
                               "enc": enc or 0, "send": snd or 0, "total": tot})

    print(f"\n{'='*60}")
    print("  Edge → Fog: Κρυπτογράφηση")
    print(f"{'='*60}")
    print(f"  Γύροι: {len(edge_enc)}")
    if payload_kb:
        print(f"  Payload: {mean(payload_kb):.1f} KB")
    print("  Χρόνος κρυπτογράφησης:")
    stats(edge_enc)

    # ── Fog decrypt ───────────────────────────────────────────────
    fog_dec = []
    for line in fog_lines:
        if "decrypted model from edge" in line:
            v = extract_value(line, "decrypt")
            if v is not None: fog_dec.append(v)

    fog_enc = []
    for line in fog_lines:
        if "encrypted model for cloud" in line:
            v = extract_value(line, "encrypt")
            if v is not None: fog_enc.append(v)

    print(f"\n{'='*60}")
    print("  Fog: Αποκρυπτογράφηση από Edge")
    print(f"{'='*60}")
    print(f"  Γύροι: {len(fog_dec)}")
    print("  Χρόνος αποκρυπτογράφησης:")
    stats(fog_dec)

    print(f"\n{'='*60}")
    print("  Fog → Cloud: Κρυπτογράφηση")
    print(f"{'='*60}")
    print(f"  Γύροι: {len(fog_enc)}")
    print("  Χρόνος κρυπτογράφησης:")
    stats(fog_enc)

    # ── Cloud decrypt ─────────────────────────────────────────────
    cloud_dec = []
    for line in cloud_lines:
        if "decrypted model from fog" in line:
            v = extract_value(line, "decrypt")
            if v is not None: cloud_dec.append(v)

    print(f"\n{'='*60}")
    print("  Cloud: Αποκρυπτογράφηση από Fog")
    print(f"{'='*60}")
    print(f"  Γύροι: {len(cloud_dec)}")
    print("  Χρόνος αποκρυπτογράφησης:")
    stats(cloud_dec)

    # ── Round summaries ───────────────────────────────────────────
    if rounds:
        print(f"\n{'='*60}")
        print("  Ανά Γύρο")
        print(f"{'='*60}")
        print(f"  {'Round':<6} {'Train(s)':<10} {'RAM(MB)':<10} {'Enc(ms)':<9} {'Send(s)':<9} {'Total(s)'}")
        print("  " + "-" * 58)
        for r in rounds:
            print(f"  {r['round']:<6} {r['train']:<10} {r['ram']:<10} {r['enc']*1000:<9.2f} {r['send']:<9} {r['total']}")

        # ── Συνολικό overhead ─────────────────────────────────────
        all_enc   = [r['enc']   for r in rounds]
        all_total = [r['total'] for r in rounds]

        # Συνολικό crypto per round (edge_enc + fog_dec + fog_enc + cloud_dec)
        n = min(len(edge_enc), len(fog_dec), len(fog_enc), len(cloud_dec))
        if n > 0:
            total_crypto = [
                edge_enc[i] + fog_dec[i] + fog_enc[i] + cloud_dec[i]
                for i in range(n)
            ]
        else:
            total_crypto = edge_enc[:len(all_total)]

        print(f"\n{'='*60}")
        print("  Συνολική Ανάλυση")
        print(f"{'='*60}")
        print(f"  Μέσος χρόνος encrypt (edge)     : {mean(all_enc)*1000:.3f} ms")
        if total_crypto:
            print(f"  Μέσος συνολικός crypto overhead : {mean(total_crypto)*1000:.3f} ms")
            pct = mean(total_crypto) / mean(all_total) * 100
            print(f"  Crypto ως % γύρου               : {pct:.4f}%")
        print(f"  Μέσος χρόνος γύρου              : {mean(all_total):.2f} s")
        print(f"  Baseline Πείραμα 3 (PC)         : ~50.56 s")

    print(f"\n{'='*60}")
    print("  Ανάλυση ολοκληρώθηκε!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()