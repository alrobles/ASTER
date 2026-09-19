#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "caster-site-workflow.hpp"
#include "caster-site-accelerator-data.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "caster-site-accelerator-hip.hpp"
#endif

namespace {

using caster_accelerator::ExecutionResult;
using caster_accelerator::Operation;
using caster_accelerator::OperationKind;
using caster_accelerator::ResidentDataset;
using caster_accelerator::TapeBatch;

std::vector<int8_t> generateInitialColors(uint32_t taxa, uint64_t seed) {
    std::vector<int8_t> colors(taxa);
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<int> groupDistribution(0, 2);
    for (int8_t& color : colors) {
        color = static_cast<int8_t>(groupDistribution(random));
    }
    return colors;
}

std::vector<double> initializeProduction(
    Tripartition& tripartition,
    const std::vector<int8_t>& colors,
    int threads
) {
    for (uint32_t taxon = 0; taxon < colors.size(); ++taxon) {
        for (int part = 0; part < threads; ++part) {
            tripartition.updatePart(part, colors[taxon], taxon);
        }
    }
    std::vector<double> scores(1, 0.0);
    for (int part = 0; part < threads; ++part) {
        scores[0] += tripartition.scorePart(part);
    }
    return scores;
}

ExecutionResult executeProduction(
    Tripartition& tripartition,
    const TapeBatch& batch,
    int threads
) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> scores(batch.scoreCount, 0.0);
    size_t scoreIndex = 0;
    for (const Operation& operation : batch.operations) {
        if (operation.kind == OperationKind::Update) {
            for (int part = 0; part < threads; ++part) {
                tripartition.updatePart(
                    part,
                    operation.to,
                    operation.taxon
                );
            }
            continue;
        }
        for (int part = 0; part < threads; ++part) {
            scores[scoreIndex] += tripartition.scorePart(part);
        }
        ++scoreIndex;
    }
    const auto stop = std::chrono::steady_clock::now();
    return {
        std::move(scores),
        std::chrono::duration<double, std::milli>(stop - start).count()
    };
}

void requireEqualCounts(
    const std::vector<uint16_t>& expected,
    const std::vector<uint16_t>& actual
) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("counter size mismatch");
    }
    const auto mismatch = std::mismatch(
        expected.begin(),
        expected.end(),
        actual.begin()
    );
    if (mismatch.first != expected.end()) {
        throw std::runtime_error(
            "counter mismatch at index "
                + std::to_string(mismatch.first - expected.begin())
        );
    }
}

void requireInvalidTapeChecks(const ResidentDataset& dataset) {
    {
        std::vector<int8_t> colors = dataset.initialColors;
        const TapeBatch batch{
            {{
                dataset.taxa,
                -1,
                0,
                OperationKind::Update,
                0
            }},
            0
        };
        try {
            caster_accelerator::validateTapeBatch(dataset, batch, colors);
            throw std::runtime_error("out-of-range taxon was accepted");
        }
        catch (const std::invalid_argument&) {
        }
    }
    {
        std::vector<int8_t> colors = dataset.initialColors;
        const int8_t incorrectFrom =
            static_cast<int8_t>((colors[0] + 1) % 3);
        const TapeBatch batch{
            {{0, incorrectFrom, -1, OperationKind::Update, 0}},
            0
        };
        try {
            caster_accelerator::validateTapeBatch(dataset, batch, colors);
            throw std::runtime_error("incorrect source color was accepted");
        }
        catch (const std::invalid_argument&) {
        }
    }
    {
        std::vector<int8_t> colors = dataset.initialColors;
        const TapeBatch batch{
            {{0, -1, -1, OperationKind::ScoreTripartition, 0}},
            0
        };
        try {
            caster_accelerator::validateTapeBatch(dataset, batch, colors);
            throw std::runtime_error("incorrect score count was accepted");
        }
        catch (const std::invalid_argument&) {
        }
    }
}

void requireConcurrentPrivateState(
    TripartitionInitializer& initializer,
    const std::vector<int8_t>& colors,
    int threads,
    const std::vector<uint16_t>& expectedCounts,
    const std::vector<double>& expectedScores
) {
    const auto run = [&initializer, &colors, threads]() {
        Tripartition tripartition(initializer);
        const std::vector<double> scores =
            initializeProduction(tripartition, colors, threads);
        return std::make_pair(
            caster_accelerator::productionCounts(tripartition),
            scores
        );
    };
    std::future<std::pair<std::vector<uint16_t>, std::vector<double>>> first =
        std::async(std::launch::async, run);
    std::future<std::pair<std::vector<uint16_t>, std::vector<double>>> second =
        std::async(std::launch::async, run);
    const auto firstResult = first.get();
    const auto secondResult = second.get();
    requireEqualCounts(expectedCounts, firstResult.first);
    requireEqualCounts(expectedCounts, secondResult.first);
    caster_accelerator::validateScores(expectedScores, firstResult.second);
    caster_accelerator::validateScores(expectedScores, secondResult.second);
}

}

int main(int argc, char** argv) {
    try {
        configureCasterSiteArguments(
            "caster-site-accelerator-bench",
            "Validate portable accelerator data against CASTER-site"
        );
        ARG.addIntArg(
            0,
            "accelerator-moves",
            120,
            "Number of deterministic replay moves"
        );
        ARG.addIntArg(
            0,
            "accelerator-batch-moves",
            30,
            "Number of replay moves per resident batch"
        );
        ARG.addIntArg(
            0,
            "accelerator-score-every",
            3,
            "Score after this many replay moves"
        );
        ARG.addIntArg(
            0,
            "accelerator-block-size",
            256,
            "HIP workgroup size"
        );
        Workflow workflow(argc, argv);

        const int moves = ARG.getIntArg("accelerator-moves");
        const int batchMoves = ARG.getIntArg("accelerator-batch-moves");
        const int scoreEvery = ARG.getIntArg("accelerator-score-every");
        const int blockSize = ARG.getIntArg("accelerator-block-size");
        if (
            moves <= 0
            || batchMoves <= 0
            || scoreEvery <= 0
            || blockSize <= 0
        ) {
            throw std::invalid_argument(
                "accelerator replay arguments must be positive"
            );
        }

        ResidentDataset dataset =
            caster_accelerator::extractDataset(workflow.tripInit);
        dataset.initialColors = generateInitialColors(
            dataset.taxa,
            static_cast<uint64_t>(ARG.getIntArg("seed"))
        );
        dataset.initialCounts = caster_accelerator::buildInitialCounts(
            dataset,
            dataset.initialColors
        );
        requireInvalidTapeChecks(dataset);
        const std::vector<TapeBatch> batches =
            caster_accelerator::generateTapeBatches(
                dataset.initialColors,
                static_cast<size_t>(moves),
                static_cast<size_t>(batchMoves),
                static_cast<size_t>(scoreEvery),
                static_cast<uint64_t>(ARG.getIntArg("seed")) + 1
            );

        Tripartition production(workflow.tripInit);
        const std::vector<double> productionInitialScore =
            initializeProduction(
                production,
                dataset.initialColors,
                workflow.tripInit.nThreads
            );
        const std::vector<uint16_t> productionInitialCounts =
            caster_accelerator::productionCounts(production);
        requireEqualCounts(
            productionInitialCounts,
            dataset.initialCounts
        );
        Tripartition independentProduction(workflow.tripInit);
        requireEqualCounts(
            productionInitialCounts,
            caster_accelerator::productionCounts(production)
        );
        requireConcurrentPrivateState(
            workflow.tripInit,
            dataset.initialColors,
            workflow.tripInit.nThreads,
            productionInitialCounts,
            productionInitialScore
        );
        initializeProduction(
            independentProduction,
            dataset.initialColors,
            workflow.tripInit.nThreads
        );
        requireEqualCounts(
            productionInitialCounts,
            caster_accelerator::productionCounts(independentProduction)
        );
        requireEqualCounts(
            productionInitialCounts,
            caster_accelerator::productionCounts(production)
        );

        caster_accelerator::CpuResidentExecutor executor(dataset);
        const TapeBatch initialScoreBatch{
            {{0, -1, -1, OperationKind::ScoreTripartition, 0}},
            1
        };
        const ExecutionResult acceleratorInitialScore =
            executor.execute(initialScoreBatch);
        caster_accelerator::validateScores(
            productionInitialScore,
            acceleratorInitialScore.scores
        );
        std::vector<int8_t> validationColors = dataset.initialColors;
        double productionMilliseconds = 0.0;
        double acceleratorCpuMilliseconds =
            acceleratorInitialScore.milliseconds;
        size_t totalScores = 0;

#ifdef __HIPCC__
        size_t maxOperations = initialScoreBatch.operations.size();
        size_t maxScores = initialScoreBatch.scoreCount;
        for (const TapeBatch& batch : batches) {
            maxOperations = std::max(maxOperations, batch.operations.size());
            maxScores = std::max(maxScores, batch.scoreCount);
        }
        caster_accelerator::PortableHipExecutor hipExecutor(
            dataset,
            maxOperations,
            maxScores,
            static_cast<uint32_t>(blockSize)
        );
        const caster_accelerator::HipBatchResult hipInitialScore =
            hipExecutor.execute(initialScoreBatch);
        caster_accelerator::validateScores(
            productionInitialScore,
            hipInitialScore.scores
        );
        double hipTapeUploadMilliseconds =
            hipInitialScore.tapeUploadMilliseconds;
        double hipKernelMilliseconds = hipInitialScore.kernelMilliseconds;
        double hipScoreDownloadMilliseconds =
            hipInitialScore.scoreDownloadMilliseconds;
        double hipHostReductionMilliseconds =
            hipInitialScore.hostReductionMilliseconds;
#endif

        for (const TapeBatch& batch : batches) {
            caster_accelerator::validateTapeBatch(
                dataset,
                batch,
                validationColors
            );
            const ExecutionResult expected = executeProduction(
                production,
                batch,
                workflow.tripInit.nThreads
            );
            const ExecutionResult actual = executor.execute(batch);
            caster_accelerator::validateScores(
                expected.scores,
                actual.scores
            );
#ifdef __HIPCC__
            const caster_accelerator::HipBatchResult hipResult =
                hipExecutor.execute(batch);
            caster_accelerator::validateScores(
                expected.scores,
                hipResult.scores
            );
            hipTapeUploadMilliseconds +=
                hipResult.tapeUploadMilliseconds;
            hipKernelMilliseconds += hipResult.kernelMilliseconds;
            hipScoreDownloadMilliseconds +=
                hipResult.scoreDownloadMilliseconds;
            hipHostReductionMilliseconds +=
                hipResult.hostReductionMilliseconds;
#endif
            requireEqualCounts(
                caster_accelerator::productionCounts(production),
                executor.currentCounts()
            );
            productionMilliseconds += expected.milliseconds;
            acceleratorCpuMilliseconds += actual.milliseconds;
            totalScores += actual.scores.size();
        }

#ifdef __HIPCC__
        requireEqualCounts(
            caster_accelerator::productionCounts(production),
            hipExecutor.downloadCounts()
        );
        hipDeviceProp_t properties{};
        int device = 0;
        caster_accelerator::checkHip(
            hipGetDevice(&device),
            "hipGetDevice"
        );
        caster_accelerator::checkHip(
            hipGetDeviceProperties(&properties, device),
            "hipGetDeviceProperties"
        );
#endif

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "taxa=" << dataset.taxa << '\n';
        std::cout << "partitions=" << dataset.partitions.size() << '\n';
        std::cout << "sites=" << dataset.sites << '\n';
        std::cout << "state_bytes=" << dataset.states.size() << '\n';
        std::cout << "counter_bytes="
                  << dataset.initialCounts.size() * sizeof(uint16_t) << '\n';
        std::cout << "batches=" << batches.size() << '\n';
        std::cout << "scores=" << totalScores + acceleratorInitialScore.scores.size()
                  << '\n';
        std::cout << "production_ms=" << productionMilliseconds << '\n';
        std::cout << "accelerator_cpu_ms="
                  << acceleratorCpuMilliseconds << '\n';
#ifdef __HIPCC__
        std::cout << "hip_device=" << properties.name << '\n';
        std::cout << "hip_blocks=" << hipExecutor.blocks() << '\n';
        std::cout << "hip_device_allocation_bytes="
                  << hipExecutor.deviceAllocationBytes() << '\n';
        std::cout << "hip_allocation_ms="
                  << hipExecutor.allocationMilliseconds() << '\n';
        std::cout << "hip_static_upload_ms="
                  << hipExecutor.staticUploadMilliseconds() << '\n';
        std::cout << "hip_tape_upload_ms="
                  << hipTapeUploadMilliseconds << '\n';
        std::cout << "hip_kernel_ms=" << hipKernelMilliseconds << '\n';
        std::cout << "hip_score_download_ms="
                  << hipScoreDownloadMilliseconds << '\n';
        std::cout << "hip_host_reduction_ms="
                  << hipHostReductionMilliseconds << '\n';
#endif
        std::cout << "invalid_input_validation=passed\n";
        std::cout << "private_state_validation=passed\n";
        std::cout << "concurrent_private_state_validation=passed\n";
        std::cout << "validation=passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 1;
    }
}
