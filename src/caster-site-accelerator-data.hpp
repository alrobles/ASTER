#ifndef CASTER_SITE_ACCELERATOR_DATA_HPP
#define CASTER_SITE_ACCELERATOR_DATA_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef __HIPCC__
#define CASTER_ACCELERATOR_HD __host__ __device__
#else
#define CASTER_ACCELERATOR_HD
#endif

namespace caster_accelerator {

enum class OperationKind : uint8_t {
    Update,
    ScoreTripartition
};

struct Operation {
    uint32_t taxon;
    int8_t from;
    int8_t to;
    OperationKind kind;
    uint8_t reserved;
};

struct SpeciesRange {
    uint32_t begin;
    uint32_t end;
};

struct Partition {
    uint64_t sequenceOffset;
    uint64_t siteOffset;
    uint32_t siteCount;
    uint32_t rangeOffset;
    float frequencies[4];
};

static_assert(sizeof(Operation) == 8, "Operation layout must be stable");
static_assert(sizeof(SpeciesRange) == 8, "SpeciesRange layout must be stable");
static_assert(sizeof(Partition) == 40, "Partition layout must be stable");

struct ResidentDataset {
    uint32_t taxa;
    uint64_t sites;
    std::vector<uint8_t> states;
    std::vector<Partition> partitions;
    std::vector<SpeciesRange> speciesRanges;
    std::vector<uint32_t> sitePartitions;
    std::vector<int8_t> initialColors;
    std::vector<uint16_t> initialCounts;
};

struct TapeBatch {
    std::vector<Operation> operations;
    size_t scoreCount;
};

struct ExecutionResult {
    std::vector<double> scores;
    double milliseconds;
};

CASTER_ACCELERATOR_HD inline int64_t xxyy(
    int64_t x0,
    int64_t x1,
    int64_t x2,
    int64_t y0,
    int64_t y1,
    int64_t y2
) {
    return x0 * (x0 - 1) * y1 * y2
        + x1 * (x1 - 1) * y2 * y0
        + x2 * (x2 - 1) * y0 * y1
        + y0 * (y0 - 1) * x1 * x2
        + y1 * (y1 - 1) * x2 * x0
        + y2 * (y2 - 1) * x0 * x1;
}

CASTER_ACCELERATOR_HD inline double scorePosition(
    const uint16_t* counts,
    const float* frequencies
) {
    const double a = frequencies[0];
    const double c = frequencies[1];
    const double g = frequencies[2];
    const double t = frequencies[3];
    const double r = a + g;
    const double y = c + t;
    const double r2 = a * a + g * g;
    const double y2 = c * c + t * t;

    const int64_t a0 = counts[0];
    const int64_t c0 = counts[1];
    const int64_t g0 = counts[2];
    const int64_t t0 = counts[3];
    const int64_t a1 = counts[4];
    const int64_t c1 = counts[5];
    const int64_t g1 = counts[6];
    const int64_t t1 = counts[7];
    const int64_t a2 = counts[8];
    const int64_t c2 = counts[9];
    const int64_t g2 = counts[10];
    const int64_t t2 = counts[11];
    const int64_t rr0 = a0 + g0;
    const int64_t rr1 = a1 + g1;
    const int64_t rr2 = a2 + g2;
    const int64_t yy0 = c0 + t0;
    const int64_t yy1 = c1 + t1;
    const int64_t yy2 = c2 + t2;

    const int64_t rryy = xxyy(rr0, rr1, rr2, yy0, yy1, yy2);
    const int64_t aayy = xxyy(a0, a1, a2, yy0, yy1, yy2);
    const int64_t ggyy = xxyy(g0, g1, g2, yy0, yy1, yy2);
    const int64_t rrcc = xxyy(rr0, rr1, rr2, c0, c1, c2);
    const int64_t rrtt = xxyy(rr0, rr1, rr2, t0, t1, t2);
    const int64_t aacc = xxyy(a0, a1, a2, c0, c1, c2);
    const int64_t aatt = xxyy(a0, a1, a2, t0, t1, t2);
    const int64_t ggcc = xxyy(g0, g1, g2, c0, c1, c2);
    const int64_t ggtt = xxyy(g0, g1, g2, t0, t1, t2);

    return rryy * r2 * y2
        - (aayy + ggyy) * (r * r) * y2
        - (rrcc + rrtt) * r2 * (y * y)
        + (aacc + aatt + ggcc + ggtt) * (r * r) * (y * y);
}

inline uint8_t encodeBase(char base) {
    switch (base) {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        default:
            return 4;
    }
}

inline ResidentDataset extractDataset(
    const TripartitionInitializer& initializer
) {
    if (initializer.nSpecies <= 0) {
        throw std::invalid_argument("CASTER dataset has no species");
    }
    if (
        static_cast<uint64_t>(initializer.nSpecies)
        > std::numeric_limits<uint32_t>::max()
    ) {
        throw std::invalid_argument("species count exceeds uint32_t range");
    }

    ResidentDataset dataset{
        static_cast<uint32_t>(initializer.nSpecies),
        0,
        {},
        {},
        {},
        {},
        {},
        {}
    };

    for (const TripartitionInitializer::Gene& gene : initializer.genes) {
        if (gene.nRep != 0 || gene.nKernal != gene.nSite) {
            throw std::invalid_argument(
                "Phase 1 supports non-replicated CASTER-site partitions"
            );
        }
        if (gene.nSite == 0) {
            continue;
        }
        if (
            gene.nSite < 0
            || static_cast<uint64_t>(gene.nSite)
                > std::numeric_limits<uint32_t>::max()
        ) {
            throw std::invalid_argument("partition site count is invalid");
        }
        if (
            dataset.speciesRanges.size()
                > std::numeric_limits<uint32_t>::max() - dataset.taxa
        ) {
            throw std::invalid_argument("species-range table exceeds uint32_t");
        }

        Partition partition{
            dataset.states.size(),
            dataset.sites,
            static_cast<uint32_t>(gene.nSite),
            static_cast<uint32_t>(dataset.speciesRanges.size()),
            {gene.pi[0], gene.pi[1], gene.pi[2], gene.pi[3]}
        };
        const uint32_t partitionIndex =
            static_cast<uint32_t>(dataset.partitions.size());
        uint32_t reorderedIndividual = 0;

        for (uint32_t species = 0; species < dataset.taxa; ++species) {
            const int rangeStart =
                species == 0 ? 0 : gene.species2indRange[species - 1];
            const int rangeEnd = gene.species2indRange[species];
            const uint32_t begin = reorderedIndividual;
            for (int index = rangeStart; index < rangeEnd; ++index) {
                const int individual = gene.indBin[index];
                for (int site = 0; site < gene.nSite; ++site) {
                    dataset.states.push_back(
                        encodeBase(
                            initializer.seq.get(
                                gene.ind2seq[individual]
                                    + static_cast<size_t>(site)
                            )
                        )
                    );
                }
                ++reorderedIndividual;
            }
            dataset.speciesRanges.push_back(
                {begin, reorderedIndividual}
            );
        }
        if (reorderedIndividual != static_cast<uint32_t>(gene.nInd)) {
            throw std::runtime_error("individual range extraction mismatch");
        }

        dataset.partitions.push_back(partition);
        dataset.sitePartitions.insert(
            dataset.sitePartitions.end(),
            static_cast<size_t>(gene.nSite),
            partitionIndex
        );
        dataset.sites += static_cast<uint64_t>(gene.nSite);
    }

    if (dataset.sites == 0) {
        throw std::invalid_argument("CASTER dataset has no informative sites");
    }
    if (dataset.sitePartitions.size() != dataset.sites) {
        throw std::runtime_error("site-partition extraction mismatch");
    }
    return dataset;
}

CASTER_ACCELERATOR_HD inline void applyObservation(
    uint16_t* counts,
    uint8_t base,
    int8_t from,
    int8_t to
) {
    if (base > 3) {
        return;
    }
    if (from >= 0) {
        --counts[static_cast<size_t>(from) * 4 + base];
    }
    if (to >= 0) {
        ++counts[static_cast<size_t>(to) * 4 + base];
    }
}

inline void applyDatasetUpdate(
    const ResidentDataset& dataset,
    std::vector<uint16_t>& counts,
    const Operation& operation
) {
    for (uint64_t site = 0; site < dataset.sites; ++site) {
        const Partition& partition =
            dataset.partitions[dataset.sitePartitions[site]];
        const uint64_t localSite = site - partition.siteOffset;
        const SpeciesRange range = dataset.speciesRanges[
            partition.rangeOffset + operation.taxon
        ];
        uint16_t* siteCounts = counts.data() + site * 12;
        for (uint32_t individual = range.begin; individual < range.end; ++individual) {
            const uint8_t base = dataset.states[
                partition.sequenceOffset
                    + static_cast<uint64_t>(individual) * partition.siteCount
                    + localSite
            ];
            applyObservation(siteCounts, base, operation.from, operation.to);
        }
    }
}

inline std::vector<uint16_t> buildInitialCounts(
    const ResidentDataset& dataset,
    const std::vector<int8_t>& colors
) {
    if (colors.size() != dataset.taxa) {
        throw std::invalid_argument("initial color count does not match taxa");
    }
    std::vector<uint16_t> counts(dataset.sites * 12, 0);
    for (uint32_t taxon = 0; taxon < dataset.taxa; ++taxon) {
        if (colors[taxon] < -1 || colors[taxon] > 2) {
            throw std::invalid_argument("initial color is out of range");
        }
        if (colors[taxon] >= 0) {
            applyDatasetUpdate(
                dataset,
                counts,
                {
                    taxon,
                    -1,
                    colors[taxon],
                    OperationKind::Update,
                    0
                }
            );
        }
    }
    return counts;
}

inline std::vector<TapeBatch> generateTapeBatches(
    const std::vector<int8_t>& initialColors,
    size_t moves,
    size_t movesPerBatch,
    size_t scoreEvery,
    uint64_t seed
) {
    if (initialColors.empty()) {
        throw std::invalid_argument("initial colors are empty");
    }
    if (movesPerBatch == 0 || scoreEvery == 0) {
        throw std::invalid_argument(
            "moves-per-batch and score-every must be positive"
        );
    }

    std::vector<int8_t> colors = initialColors;
    std::vector<TapeBatch> batches;
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<uint32_t> taxonDistribution(
        0,
        static_cast<uint32_t>(colors.size() - 1)
    );
    std::uniform_int_distribution<int> destinationDistribution(-1, 2);

    size_t completedMoves = 0;
    while (completedMoves < moves) {
        TapeBatch batch{{}, 0};
        const size_t batchMoves =
            std::min(movesPerBatch, moves - completedMoves);
        for (size_t move = 0; move < batchMoves; ++move) {
            const uint32_t taxon = taxonDistribution(random);
            const int8_t from = colors[taxon];
            int8_t to = from;
            while (to == from) {
                to = static_cast<int8_t>(destinationDistribution(random));
            }
            batch.operations.push_back(
                {taxon, from, to, OperationKind::Update, 0}
            );
            colors[taxon] = to;
            if ((completedMoves + move + 1) % scoreEvery == 0) {
                batch.operations.push_back(
                    {0, -1, -1, OperationKind::ScoreTripartition, 0}
                );
                ++batch.scoreCount;
            }
        }
        if (
            batch.operations.empty()
            || batch.operations.back().kind
                != OperationKind::ScoreTripartition
        ) {
            batch.operations.push_back(
                {0, -1, -1, OperationKind::ScoreTripartition, 0}
            );
            ++batch.scoreCount;
        }
        batches.push_back(std::move(batch));
        completedMoves += batchMoves;
    }
    return batches;
}

inline void validateTapeBatch(
    const ResidentDataset& dataset,
    const TapeBatch& batch,
    std::vector<int8_t>& colors
) {
    std::vector<int8_t> nextColors = colors;
    size_t scoreCount = 0;
    for (const Operation& operation : batch.operations) {
        if (operation.kind == OperationKind::ScoreTripartition) {
            ++scoreCount;
            continue;
        }
        if (operation.kind != OperationKind::Update) {
            throw std::invalid_argument("unknown operation kind");
        }
        if (operation.taxon >= dataset.taxa) {
            throw std::invalid_argument("operation taxon is out of range");
        }
        if (
            operation.from < -1
            || operation.from > 2
            || operation.to < -1
            || operation.to > 2
        ) {
            throw std::invalid_argument("operation group is out of range");
        }
        if (nextColors[operation.taxon] != operation.from) {
            throw std::invalid_argument(
                "operation source does not match current color"
            );
        }
        nextColors[operation.taxon] = operation.to;
    }
    if (scoreCount != batch.scoreCount) {
        throw std::invalid_argument("score-operation count mismatch");
    }
    colors.swap(nextColors);
}

class CpuResidentExecutor {
    const ResidentDataset& dataset;
    std::vector<uint16_t> counts;
    std::vector<int8_t> colors;

public:
    explicit CpuResidentExecutor(const ResidentDataset& dataset):
        dataset(dataset),
        counts(dataset.initialCounts),
        colors(dataset.initialColors) {}

    ExecutionResult execute(const TapeBatch& batch) {
        std::vector<int8_t> nextColors = colors;
        validateTapeBatch(dataset, batch, nextColors);
        const auto start = std::chrono::steady_clock::now();
        std::vector<double> scores(batch.scoreCount, 0.0);
        size_t scoreIndex = 0;

        for (const Operation& operation : batch.operations) {
            if (operation.kind == OperationKind::Update) {
                applyDatasetUpdate(dataset, counts, operation);
                continue;
            }
            for (uint64_t site = 0; site < dataset.sites; ++site) {
                const Partition& partition =
                    dataset.partitions[dataset.sitePartitions[site]];
                scores[scoreIndex] += scorePosition(
                    counts.data() + site * 12,
                    partition.frequencies
                );
            }
            ++scoreIndex;
        }

        const auto stop = std::chrono::steady_clock::now();
        colors.swap(nextColors);
        return {
            std::move(scores),
            std::chrono::duration<double, std::milli>(stop - start).count()
        };
    }

    const std::vector<uint16_t>& currentCounts() const {
        return counts;
    }
};

inline std::vector<uint16_t> productionCounts(
    const TripartitionInitializer& initializer
) {
    size_t sites = 0;
    for (const TripartitionInitializer::Gene& gene : initializer.genes) {
        if (gene.nRep == 0) {
            sites += gene.nKernal;
        }
    }
    std::vector<uint16_t> counts;
    counts.reserve(sites * 12);
    for (const TripartitionInitializer::Gene& gene : initializer.genes) {
        if (gene.nRep != 0) {
            throw std::invalid_argument(
                "production counter extraction requires nRep == 0"
            );
        }
        for (int site = 0; site < gene.nKernal; ++site) {
            for (size_t group = 0; group < 3; ++group) {
                for (size_t base = 0; base < 4; ++base) {
                    counts.push_back(gene.kernal[site].cnt[group][base]);
                }
            }
        }
    }
    return counts;
}

inline void validateScores(
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    double absoluteTolerance = 1e-6,
    double relativeTolerance = 1e-12
) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("score count mismatch");
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        const double absoluteDifference =
            std::abs(expected[index] - actual[index]);
        const double scale =
            std::max(std::abs(expected[index]), std::abs(actual[index]));
        if (
            absoluteDifference
            > absoluteTolerance + relativeTolerance * scale
        ) {
            throw std::runtime_error(
                "score mismatch at index " + std::to_string(index)
            );
        }
    }
}

}

#endif
