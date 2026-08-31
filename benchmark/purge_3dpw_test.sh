#!/usr/bin/env bash
# Remove every 3DPW *test* sequence crop from a distillation folder, so the
# 3DPW test split can be used as a real benchmark.
#
# Dry run (default — prints what it would remove, deletes nothing):
#     ./benchmark/purge_3dpw_test.sh /path/to/data/sam3d_distill_mix
# Actually delete:
#     ./benchmark/purge_3dpw_test.sh /path/to/data/sam3d_distill_mix --delete
#
# Crops are named  imageFiles_<sequence>_image_<frame>_p<n>.{npz,png}
# and the pattern anchors on the literal "_image_" that follows the sequence
# name, so no sequence can match another one's files by prefix.

set -euo pipefail

ROOT="${1:?usage: $0 <distill_dir> [--delete]}"
MODE="${2:---dry-run}"

# The 24 official 3DPW test sequences.
TEST_SEQS="
downtown_arguing_00 downtown_bar_00 downtown_bus_00 downtown_cafe_00
downtown_car_00 downtown_crossStreets_00 downtown_downstairs_00
downtown_enterShop_00 downtown_rampAndStairs_00 downtown_runForBus_00
downtown_runForBus_01 downtown_sitOnStairs_00 downtown_stairs_00
downtown_upstairs_00 downtown_walkBridge_01 downtown_walking_00
downtown_walkUphill_00 downtown_warmWelcome_00 downtown_weeklyMarket_00
downtown_windowShopping_00 flat_guitar_01 flat_packBags_00
office_phoneCall_00 outdoors_fencing_01
"

ANN="$ROOT/annotations"
IMG="$ROOT/images"
[[ -d "$ANN" && -d "$IMG" ]] || { echo "ERROR: $ROOT needs annotations/ and images/"; exit 1; }

before_a=$(find "$ANN" -maxdepth 1 -type f | wc -l)
before_i=$(find "$IMG" -maxdepth 1 -type f | wc -l)
total=0

for s in $TEST_SEQS; do
    n=$(find "$ANN" -maxdepth 1 -name "imageFiles_${s}_image_*" | wc -l)
    total=$((total + n))
    printf '  %-30s %6d\n' "$s" "$n"
    if [[ "$MODE" == "--delete" ]]; then
        find "$ANN" -maxdepth 1 -name "imageFiles_${s}_image_*" -delete
        find "$IMG" -maxdepth 1 -name "imageFiles_${s}_image_*" -delete
    fi
done

echo
echo "  test-sequence crops found: $total"
if [[ "$MODE" == "--delete" ]]; then
    after_a=$(find "$ANN" -maxdepth 1 -type f | wc -l)
    after_i=$(find "$IMG" -maxdepth 1 -type f | wc -l)
    echo "  annotations: $before_a -> $after_a"
    echo "  images     : $before_i -> $after_i"
    [[ "$after_a" -eq "$after_i" ]] \
        && echo "  OK: annotations and images still paired" \
        || echo "  WARNING: counts diverged — inspect before training"
else
    echo "  DRY RUN — nothing deleted. Re-run with --delete to apply."
fi
