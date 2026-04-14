"""
Cache Simulator: Two-Level (L1 + L2) with LRU, Write-Through & Write-Back
ENCS302 - Computer Organization and Architecture
Assignment 4: Cache Misses Are Expensive — Optimize or Pay the Price
"""

import json
import math
import os
from collections import OrderedDict


# ─────────────────────────────────────────────────
# CacheSet: An n-way set with LRU replacement
# ─────────────────────────────────────────────────
class CacheSet:
    def __init__(self, associativity, write_policy):
        self.ways = associativity
        self.write_policy = write_policy          # "write-back" | "write-through"
        # OrderedDict used as LRU queue: key=tag, value=dirty_bit
        self.lru = OrderedDict()

    def access(self, tag, is_write):
        """
        Returns (hit: bool, evicted_dirty: bool)
        """
        if tag in self.lru:
            # ── HIT ──
            self.lru.move_to_end(tag)
            if is_write and self.write_policy == "write-back":
                self.lru[tag] = True              # mark dirty
            return True, False

        # ── MISS ──
        evicted_dirty = False
        if len(self.lru) >= self.ways:
            # evict LRU entry
            _, dirty = self.lru.popitem(last=False)
            evicted_dirty = dirty

        dirty_on_insert = is_write and self.write_policy == "write-back"
        self.lru[tag] = dirty_on_insert
        return False, evicted_dirty

    def invalidate(self):
        self.lru.clear()


# ─────────────────────────────────────────────────
# CacheLevel: A full cache (many sets)
# ─────────────────────────────────────────────────
class CacheLevel:
    def __init__(self, size_kb, block_size_bytes, associativity, hit_time, write_policy, name):
        self.name         = name
        self.size_b       = size_kb * 1024
        self.block_size   = block_size_bytes
        self.assoc        = associativity
        self.hit_time     = hit_time
        self.write_policy = write_policy

        num_blocks        = self.size_b // self.block_size
        self.num_sets     = num_blocks // self.assoc
        self.index_bits   = int(math.log2(self.num_sets))
        self.offset_bits  = int(math.log2(self.block_size))

        self.sets = [CacheSet(associativity, write_policy) for _ in range(self.num_sets)]

        # Stats
        self.hits   = 0
        self.misses = 0
        self.reads  = 0
        self.writes = 0

    def _decode(self, address):
        offset_mask = (1 << self.offset_bits) - 1
        index_mask  = (1 << self.index_bits) - 1
        index = (address >> self.offset_bits) & index_mask
        tag   = address >> (self.offset_bits + self.index_bits)
        return index, tag

    def access(self, address, is_write):
        """Returns (hit: bool, evicted_dirty: bool)"""
        if is_write:
            self.writes += 1
        else:
            self.reads += 1

        index, tag = self._decode(address)
        hit, evicted_dirty = self.sets[index].access(tag, is_write)

        if hit:
            self.hits += 1
        else:
            self.misses += 1

        return hit, evicted_dirty

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def miss_rate(self):
        return 1.0 - self.hit_rate

    @property
    def total_accesses(self):
        return self.hits + self.misses

    def reset_stats(self):
        self.hits = self.misses = self.reads = self.writes = 0

    def summary(self):
        return {
            "name":         self.name,
            "hits":         self.hits,
            "misses":       self.misses,
            "total":        self.total_accesses,
            "hit_rate":     round(self.hit_rate * 100, 2),
            "miss_rate":    round(self.miss_rate * 100, 2),
        }


# ─────────────────────────────────────────────────
# TwoLevelCache: orchestrates L1 + L2
# ─────────────────────────────────────────────────
class TwoLevelCache:
    def __init__(self, cfg):
        l1c = cfg["L1"]
        l2c = cfg["L2"]
        self.L1 = CacheLevel(
            l1c["size_kb"], l1c["block_size_bytes"],
            l1c["associativity"], l1c["hit_time_cycles"],
            l1c["write_policy"], "L1"
        )
        self.L2 = CacheLevel(
            l2c["size_kb"], l2c["block_size_bytes"],
            l2c["associativity"], l2c["hit_time_cycles"],
            l2c["write_policy"], "L2"
        )
        self.mem_time   = cfg["main_memory"]["access_time_cycles"]
        self.total_cycles = 0
        self.total_accesses = 0

    def access(self, address, is_write):
        self.total_accesses += 1
        cycles = 0

        l1_hit, _ = self.L1.access(address, is_write)
        if l1_hit:
            cycles = self.L1.hit_time
        else:
            # L1 miss → go to L2
            l2_hit, _ = self.L2.access(address, is_write)
            if l2_hit:
                cycles = self.L1.hit_time + self.L2.hit_time
            else:
                # L2 miss → main memory
                cycles = self.L1.hit_time + self.L2.hit_time + self.mem_time

        self.total_cycles += cycles
        return cycles

    def run_trace(self, trace_path):
        """Read and simulate a trace file (format: 'R|W 0xADDR' per line)."""
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                op   = parts[0].upper()
                addr = int(parts[1], 16)
                self.access(addr, is_write=(op == 'W'))

    def amat(self):
        """Average Memory Access Time (cycles)."""
        if self.total_accesses == 0:
            return 0.0
        l1_miss = self.L1.miss_rate
        l2_miss = self.L2.miss_rate
        # AMAT = L1_hit_time + L1_miss_rate*(L2_hit_time + L2_miss_rate*Mem_time)
        return (self.L1.hit_time
                + l1_miss * (self.L2.hit_time
                             + l2_miss * self.mem_time))

    def reset(self):
        self.L1.reset_stats()
        self.L2.reset_stats()
        for s in self.L1.sets:
            s.invalidate()
        for s in self.L2.sets:
            s.invalidate()
        self.total_cycles   = 0
        self.total_accesses = 0

    def report(self):
        amat = self.amat()
        print(f"\n{'='*55}")
        print(f"  Cache Simulation Report")
        print(f"{'='*55}")
        for lvl in (self.L1, self.L2):
            s = lvl.summary()
            print(f"\n  [{s['name']}]")
            print(f"    Total Accesses : {s['total']}")
            print(f"    Hits           : {s['hits']}")
            print(f"    Misses         : {s['misses']}")
            print(f"    Hit  Rate      : {s['hit_rate']} %")
            print(f"    Miss Rate      : {s['miss_rate']} %")
        print(f"\n  AMAT              : {amat:.2f} cycles")
        print(f"  Total Cycles      : {self.total_cycles}")
        print(f"{'='*55}")
        return amat


# ─────────────────────────────────────────────────
# I/O Strategy Comparison
# ─────────────────────────────────────────────────
def io_strategy_analysis(
    transfer_size_bytes=4096,   # 4 KB transfer
    cpu_freq_ghz=3.0,
    programmed_io_cycles_per_byte=6,
    dma_setup_cycles=500,
    dma_transfer_cycles_per_byte=1,
    dma_interrupt_cycles=200
):
    """
    Estimate CPU cycles consumed for Programmed I/O vs DMA.
    """
    # Programmed I/O: CPU is busy for every byte transferred
    pio_cycles = transfer_size_bytes * programmed_io_cycles_per_byte

    # DMA: CPU pays setup + interrupt; transfer happens in background
    dma_cycles_cpu = dma_setup_cycles + dma_interrupt_cycles
    dma_cycles_total = dma_setup_cycles + (transfer_size_bytes * dma_transfer_cycles_per_byte) + dma_interrupt_cycles

    cycle_time_ns = 1e9 / (cpu_freq_ghz * 1e9)   # ns per cycle

    pio_time_us  = pio_cycles  * cycle_time_ns / 1000
    dma_time_us  = dma_cycles_total * cycle_time_ns / 1000
    dma_cpu_us   = dma_cycles_cpu   * cycle_time_ns / 1000

    print(f"\n{'='*55}")
    print(f"  I/O Strategy Comparison  ({transfer_size_bytes} bytes @ {cpu_freq_ghz} GHz)")
    print(f"{'='*55}")
    print(f"\n  Programmed I/O")
    print(f"    CPU cycles used   : {pio_cycles:,}")
    print(f"    Elapsed time      : {pio_time_us:.2f} µs")
    print(f"    CPU utilization   : 100% (busy-wait)")

    print(f"\n  DMA Transfer")
    print(f"    CPU cycles (total): {dma_cycles_total:,}")
    print(f"    CPU cycles (busy) : {dma_cycles_cpu:,}  (setup + interrupt)")
    print(f"    Total elapsed     : {dma_time_us:.2f} µs")
    print(f"    CPU busy time     : {dma_cpu_us:.2f} µs")
    print(f"    CPU utilization   : {100*dma_cycles_cpu/dma_cycles_total:.1f}%  (freed for other work)")

    speedup = pio_cycles / dma_cycles_cpu
    print(f"\n  DMA CPU-efficiency speedup : {speedup:.1f}x over Programmed I/O")
    print(f"{'='*55}")

    return {
        "pio_cycles": pio_cycles,
        "pio_time_us": round(pio_time_us, 3),
        "dma_total_cycles": dma_cycles_total,
        "dma_cpu_cycles": dma_cycles_cpu,
        "dma_total_time_us": round(dma_time_us, 3),
        "dma_cpu_time_us": round(dma_cpu_us, 3),
        "cpu_efficiency_speedup": round(speedup, 2),
    }


# ─────────────────────────────────────────────────
# Main Driver
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    with open("config.json") as f:
        cfg = json.load(f)

    traces = {
        "Sequential": "traces/trace_sequential.txt",
        "Random":     "traces/trace_random.txt",
        "Strided":    "traces/trace_strided.txt",
    }

    print("\n" + "="*55)
    print("  ENCS302 Assignment 4 — Cache Simulator")
    print("="*55)

    results = {}
    for name, path in traces.items():
        print(f"\n>>> Trace: {name}")
        sim = TwoLevelCache(cfg)
        sim.run_trace(path)
        amat = sim.report()
        results[name] = {
            "L1": sim.L1.summary(),
            "L2": sim.L2.summary(),
            "AMAT": round(amat, 3),
        }

    # I/O strategy analysis
    io_results = io_strategy_analysis(
        transfer_size_bytes=4096,
        cpu_freq_ghz=cfg["cpu_frequency_ghz"]
    )

    print("\nAll simulations complete.")
