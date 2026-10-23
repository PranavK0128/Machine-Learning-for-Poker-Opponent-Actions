"""Pack the label-free incumbent preflop feature export for Notebook 09."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


STAGE6_ROOT = Path(
    "C:/Users/konda/Stack Society PokerNow Analyzer/"
    "Poker Assistant Model/Stage 6 - Qualified Layers Improvement"
)
SOURCE = STAGE6_ROOT / "data/exports/ml/stage6-incumbent-preflop-native-features.csv"
TARGET = STAGE6_ROOT / "data/exports/ml/stage6-incumbent-preflop-native-features.parquet"
KEEP_COLUMNS = (
    "hand_id", "decisionType", "stakeLabel", "position", "positionGroup",
    "primaryOpportunity", "facingSituation", "openerPosition", "threeBettorPosition",
    "seat", "stackBb", "rfiOpportunity", "stealOpportunity", "coldCallOpportunity",
    "threeBetOpportunity", "squeezeOpportunity", "foldToStealOpportunity",
    "facedOpenRaise", "facedThreeBet", "facedFourBet", "foldToFourBetOpportunity",
    "openRaiseAmountBb", "threeBetToAmountBb", "priorLimperCount", "priorCallerCount",
    "playersLeftToAct",
)


def main() -> None:
    if Path.cwd().resolve() != STAGE6_ROOT or "onedrive" in str(STAGE6_ROOT).lower():
        raise RuntimeError(f"run from the canonical Stage 6 root: {STAGE6_ROOT}")
    reader = csv.open_csv(
        SOURCE,
        read_options=csv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=csv.ConvertOptions(strings_can_be_null=True),
    )
    temporary = TARGET.with_name(f".{TARGET.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch]).select(KEEP_COLUMNS)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    compression_level=9,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=100_000)
            rows += len(table)
        if writer is None:
            raise RuntimeError("incumbent preflop native feature export is empty")
    finally:
        if writer is not None:
            writer.close()
    if rows != 1_285_167:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"incumbent preflop native feature row count changed: {rows}")
    temporary.replace(TARGET)
    print({"rows": rows, "path": str(TARGET), "bytes": TARGET.stat().st_size})


if __name__ == "__main__":
    main()
