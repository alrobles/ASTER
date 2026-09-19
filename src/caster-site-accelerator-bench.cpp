#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
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
using caster_accelerator::OperationRecorder;
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
        try {
            OperationRecorder recorder(std::vector<int8_t>{3});
            throw std::runtime_error(
                "invalid recorder colors were accepted"
            );
        }
        catch (const std::invalid_argument&) {
        }
    }
    {
        OperationRecorder recorder(dataset.taxa);
        recorder.recordScore();
        try {
            recorder.batch();
            throw std::runtime_error(
                "operation tape with a pending score was accepted"
            );
        }
        catch (const std::logic_error&) {
        }
    }
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

void expectScoreRejection(
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    const char* label
) {
    bool rejected = false;
    try {
        caster_accelerator::validateScores(expected, actual);
    }
    catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(
            std::string("score validation accepted ") + label
        );
    }
}

void expectBoundedScoreRejection(
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    const std::vector<double>& errorBounds,
    const char* label
) {
    bool rejected = false;
    try {
        caster_accelerator::validateScoreBounds(
            expected,
            actual,
            errorBounds
        );
    }
    catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(
            std::string("bounded score validation accepted ") + label
        );
    }
}

void requireInvalidScoreChecks() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const std::vector<double> reference{1.0, -2.0};
    const std::vector<double> bounds{0.5, 0.5};
    for (const double bad : {nan, infinity, -infinity}) {
        std::vector<double> poisoned = reference;
        poisoned[0] = bad;
        expectScoreRejection(reference, poisoned, "a non-finite score");
        // identical non-finite scores on both sides passed before the
        // finite check: NaN - NaN is NaN, and NaN > tolerance is false
        expectScoreRejection(poisoned, poisoned, "identical non-finite scores");
        expectBoundedScoreRejection(
            reference,
            poisoned,
            bounds,
            "a non-finite bounded score"
        );
    }
    // NaN bounds previously passed silently; infinite bounds stay legal
    // because conservativeScoreErrorBound uses +inf as an unbounded signal
    expectBoundedScoreRejection(
        reference,
        reference,
        {0.5, nan},
        "a NaN error bound"
    );
    expectBoundedScoreRejection(
        reference,
        reference,
        {0.5, -0.5},
        "a negative error bound"
    );
    caster_accelerator::validateScoreBounds(
        reference,
        reference,
        {0.5, infinity}
    );
    // a batch carrying only updates is not a score experiment
    expectScoreRejection({}, {}, "empty score vectors");
    expectBoundedScoreRejection({}, {}, {}, "empty bounded score vectors");
    expectScoreRejection(reference, {1.0}, "a score count mismatch");
    expectBoundedScoreRejection(
        reference,
        {1.0},
        bounds,
        "a bounded score count mismatch"
    );
    expectScoreRejection(
        reference,
        {1.0, -3.0},
        "an out-of-tolerance score"
    );
    expectBoundedScoreRejection(
        reference,
        {1.0, -4.0},
        bounds,
        "a score beyond its error bound"
    );
    // positive controls: in-tolerance and inside-bound pairs must pass
    caster_accelerator::validateScores(reference, {1.0 + 5e-7, -2.0});
    caster_accelerator::validateScoreBounds(reference, {1.2, -2.1}, bounds);
    std::cout << "invalid_score_validation=passed\n";
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

void requireProductionTapeReplay(
    Workflow& workflow,
    const ResidentDataset& sourceDataset,
    uint32_t blockSize
) {
#if !defined(__HIPCC__) && !defined(__CUDACC__)
    (void) blockSize;
#endif
    ResidentDataset dataset = sourceDataset;
    dataset.initialColors.assign(dataset.taxa, -1);
    dataset.initialCounts.assign(dataset.sites * 12, 0);

    ThreadPool threadPool(workflow.tripInit.nThreads);
    const int roundNN =
        20 + 2 * std::sqrt(dataset.taxa) * std::log2(dataset.taxa);
    ConstrainedOptimizationAlgorithm algorithm(
        dataset.taxa,
        workflow.tripInit,
        workflow.names,
        threadPool,
        roundNN
    );
    OperationRecorder recorder(dataset.taxa);
    PlacementAlgorithm placement(
        algorithm.taxonHash,
        workflow.tripInit,
        threadPool,
        roundNN,
        &recorder
    );
    algorithm.createPlacementAlgorithm(placement, 1.0);
    placement.run();

    caster_accelerator::CpuResidentExecutor executor(dataset);
    const ExecutionResult replay = executor.execute(recorder.batch());
    caster_accelerator::validateScores(recorder.scores(), replay.scores);
    requireEqualCounts(
        caster_accelerator::productionCounts(placement.trip),
        executor.currentCounts()
    );
    std::cout << "production_tape_operations="
              << recorder.batch().operations.size() << '\n';
    std::cout << "production_tape_scores="
              << recorder.batch().scoreCount << '\n';
#if defined(__HIPCC__) || defined(__CUDACC__)
    caster_accelerator::PortableHipExecutor hipExecutor(
        dataset,
        recorder.batch().operations.size(),
        recorder.batch().scoreCount,
        blockSize
    );
    const caster_accelerator::HipBatchResult hipReplay =
        hipExecutor.execute(recorder.batch());
    caster_accelerator::validateScoreBounds(
        recorder.scores(),
        hipReplay.scores,
        hipReplay.errorBounds
    );
    requireEqualCounts(
        caster_accelerator::productionCounts(placement.trip),
        hipExecutor.downloadCounts()
    );
    double maximumError = 0.0;
    double maximumBound = 0.0;
    for (size_t index = 0; index < hipReplay.scores.size(); ++index) {
        maximumError = std::max(
            maximumError,
            std::abs(recorder.scores()[index] - hipReplay.scores[index])
        );
        maximumBound = std::max(
            maximumBound,
            hipReplay.errorBounds[index]
        );
    }
    std::cout << "production_tape_max_score_error="
              << maximumError << '\n';
    std::cout << "production_tape_max_error_bound="
              << maximumBound << '\n';
    std::cout << "production_tape_hip_validation=passed\n";
#endif
    std::cout << "production_tape_validation=passed\n";
}

void requireNumericFallback() {
    const caster_accelerator::ScoreDecision separated =
        caster_accelerator::compareScoreEstimates(
            2.0,
            0.1,
            1.0,
            0.1,
            2.0,
            1.0,
            ERROR_TOLERANCE
        );
    if (!separated.candidateBetter || separated.usedCpuFallback) {
        throw std::runtime_error(
            "separated score intervals did not use the device result"
        );
    }
    const caster_accelerator::ScoreDecision nearTie =
        caster_accelerator::compareScoreEstimates(
            1.0000001,
            0.001,
            1.0,
            0.001,
            1.0,
            1.0,
            ERROR_TOLERANCE
        );
    if (nearTie.candidateBetter || !nearTie.usedCpuFallback) {
        throw std::runtime_error(
            "overlapping score intervals did not use CPU fallback"
        );
    }
    std::cout << "numeric_fallback_validation=passed\n";
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
        bool validateProductionTape = false;
        ARG.addFlag(
            0,
            "accelerator-production-tape",
            "Capture and replay one production placement search",
            [&validateProductionTape]() {
                validateProductionTape = true;
            }
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
        requireInvalidScoreChecks();
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
        requireNumericFallback();
        if (validateProductionTape) {
            requireProductionTapeReplay(
                workflow,
                dataset,
                static_cast<uint32_t>(blockSize)
            );
        }
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

#if defined(__HIPCC__) || defined(__CUDACC__)
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
        caster_accelerator::validateScoreBounds(
            productionInitialScore,
            hipInitialScore.scores,
            hipInitialScore.errorBounds
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
#if defined(__HIPCC__) || defined(__CUDACC__)
            const caster_accelerator::HipBatchResult hipResult =
                hipExecutor.execute(batch);
            caster_accelerator::validateScores(
                expected.scores,
                hipResult.scores
            );
            caster_accelerator::validateScoreBounds(
                expected.scores,
                hipResult.scores,
                hipResult.errorBounds
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

#if defined(__HIPCC__) || defined(__CUDACC__)
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
#if defined(__HIPCC__) || defined(__CUDACC__)
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
