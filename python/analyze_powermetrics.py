#!/usr/bin/env python3
"""
Resume um relatorio do `sudo powermetrics --samplers cpu_power` (Apple
Silicon) em uma tabela por cluster: % online medio, frequencia media
(MHz) e % idle medio, ao longo de todas as amostras do arquivo.

Uso:
    python3 python/analyze_powermetrics.py results/powermetrics_p6.txt [outros arquivos...]

Serve para investigar por que uma contagem especifica de threads
(ex.: OMP_NUM_THREADS=8 no M2 Pro, que tem os P-cores divididos em
dois clusters P0/P1 de 3 nucleos cada, mais o E-Cluster de 4 nucleos)
pode ficar mais lenta que uma contagem menor -- comparando o quanto
cada cluster ficou ocupado (online/idle) e em que frequencia rodou.
"""
import re
import sys

CLUSTERS = ["E-Cluster", "P0-Cluster", "P1-Cluster"]


def parse(path):
    text = open(path).read()
    samples = text.split("*** Sampled system activity")[1:]
    rows = []
    for s in samples:
        row = {}
        for cluster in CLUSTERS:
            def grab(field, cast=float):
                m = re.search(rf"{re.escape(cluster)} {field}:\s*([\d.]+)", s)
                return cast(m.group(1)) if m else None
            row[cluster] = {
                "online": grab("Online", float),
                "freq": grab("HW active frequency", float),
                "idle": grab("idle residency", float),
            }
        rows.append(row)
    return rows


def summarize(path):
    rows = parse(path)
    n = len(rows)
    print(f"=== {path} (n={n} amostras) ===")
    for cluster in CLUSTERS:
        def avg(field):
            vals = [r[cluster][field] for r in rows if r[cluster][field] is not None]
            return sum(vals) / len(vals) if vals else None
        online = avg("online")
        freq = avg("freq")
        idle = avg("idle")
        if online is None and freq is None and idle is None:
            continue
        print(f"  {cluster:<12} online_medio={online:6.1f}%  "
              f"freq_media={freq:7.0f}MHz  idle_medio={idle:6.1f}%")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for path in sys.argv[1:]:
        summarize(path)
