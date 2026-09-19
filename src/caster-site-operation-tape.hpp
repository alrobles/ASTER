#ifndef CASTER_SITE_OPERATION_TAPE_HPP
#define CASTER_SITE_OPERATION_TAPE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

struct TapeBatch {
    std::vector<Operation> operations;
    size_t scoreCount;
};

static_assert(sizeof(Operation) == 8, "Operation layout must be stable");

class OperationRecorder {
    TapeBatch tape{{}, 0};
    std::vector<int8_t> colors;
    std::vector<double> scoreResults;

public:
    explicit OperationRecorder(size_t taxa):
        colors(taxa, -1) {}

    explicit OperationRecorder(const std::vector<int8_t>& initialColors):
        colors(initialColors) {
        for (int8_t color : colors) {
            if (color < -1 || color > 2) {
                throw std::invalid_argument(
                    "initial operation group is out of range"
                );
            }
        }
    }

    void recordUpdate(uint32_t taxon, int8_t to) {
        if (taxon >= colors.size()) {
            throw std::invalid_argument("operation taxon is out of range");
        }
        if (to < -1 || to > 2) {
            throw std::invalid_argument("operation group is out of range");
        }
        tape.operations.push_back(
            {taxon, colors[taxon], to, OperationKind::Update, 0}
        );
        colors[taxon] = to;
    }

    void recordScore() {
        tape.operations.push_back(
            {0, -1, -1, OperationKind::ScoreTripartition, 0}
        );
        ++tape.scoreCount;
    }

    void recordScoreResult(double score) {
        if (scoreResults.size() >= tape.scoreCount) {
            throw std::logic_error("score result has no matching operation");
        }
        scoreResults.push_back(score);
    }

    const TapeBatch& batch() const {
        if (scoreResults.size() != tape.scoreCount) {
            throw std::logic_error("operation tape has pending scores");
        }
        return tape;
    }

    const std::vector<double>& scores() const {
        if (scoreResults.size() != tape.scoreCount) {
            throw std::logic_error("operation tape has pending scores");
        }
        return scoreResults;
    }
};

}

#endif
