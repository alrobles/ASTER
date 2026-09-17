#ifndef CASTER_SITE_ACCELERATOR_HIP_HPP
#define CASTER_SITE_ACCELERATOR_HIP_HPP

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#error "caster-site-accelerator-hip.hpp requires a HIP or CUDA compiler"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "caster-site-accelerator-data.hpp"

namespace caster_accelerator {

struct HipBatchResult {
    std::vector<double> scores;
    double tapeUploadMilliseconds;
    double kernelMilliseconds;
    double scoreDownloadMilliseconds;
    double hostReductionMilliseconds;
};

inline void checkHip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(status)
        );
    }
}

__global__ void executeTapeKernel(
    const uint8_t* states,
    const Partition* partitions,
    const SpeciesRange* speciesRanges,
    const uint32_t* sitePartitions,
    uint64_t sites,
    const Operation* operations,
    size_t operationCount,
    uint16_t* counts,
    double* partialScores
) {
    extern __shared__ double sharedScores[];
    const uint64_t site =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = site < sites;
    Partition partition{};
    uint64_t localSite = 0;
    uint16_t* siteCounts = nullptr;

    if (active) {
        partition = partitions[sitePartitions[site]];
        localSite = site - partition.siteOffset;
        siteCounts = counts + site * 12;
    }

    size_t scoreIndex = 0;
    for (size_t operationIndex = 0;
         operationIndex < operationCount;
         ++operationIndex) {
        const Operation operation = operations[operationIndex];
        if (operation.kind == OperationKind::Update) {
            if (active) {
                const SpeciesRange range = speciesRanges[
                    partition.rangeOffset + operation.taxon
                ];
                for (
                    uint32_t individual = range.begin;
                    individual < range.end;
                    ++individual
                ) {
                    const uint8_t base = states[
                        partition.sequenceOffset
                            + static_cast<uint64_t>(individual)
                                * partition.siteCount
                            + localSite
                    ];
                    applyObservation(
                        siteCounts,
                        base,
                        operation.from,
                        operation.to
                    );
                }
            }
            continue;
        }

        sharedScores[threadIdx.x] = active
            ? scorePosition(siteCounts, partition.frequencies)
            : 0.0;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                sharedScores[threadIdx.x] +=
                    sharedScores[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partialScores[
                scoreIndex * static_cast<size_t>(gridDim.x) + blockIdx.x
            ] = sharedScores[0];
        }
        __syncthreads();
        ++scoreIndex;
    }
}

class PortableHipExecutor {
    const ResidentDataset& dataset;
    uint32_t blockSize;
    uint32_t blockCount;
    size_t maxOperations;
    size_t maxScores;
    uint8_t* deviceStates = nullptr;
    Partition* devicePartitions = nullptr;
    SpeciesRange* deviceSpeciesRanges = nullptr;
    uint32_t* deviceSitePartitions = nullptr;
    Operation* deviceOperations = nullptr;
    uint16_t* deviceCounts = nullptr;
    double* devicePartials = nullptr;
    Operation* hostOperations = nullptr;
    double* hostPartials = nullptr;
    std::vector<int8_t> colors;
    bool usable = true;
    double allocationMillisecondsValue = 0.0;
    double staticUploadMillisecondsValue = 0.0;
    size_t deviceAllocationBytesValue = 0;

public:
    PortableHipExecutor(
        const ResidentDataset& dataset,
        size_t maxOperations,
        size_t maxScores,
        uint32_t blockSize
    ):
        dataset(dataset),
        blockSize(blockSize),
        blockCount(0),
        maxOperations(maxOperations),
        maxScores(maxScores),
        colors(dataset.initialColors) {
        if (
            blockSize == 0
            || (blockSize & (blockSize - 1)) != 0
            || maxOperations == 0
            || maxScores == 0
        ) {
            throw std::invalid_argument(
                "HIP capacities must be positive and block size a power of two"
            );
        }
        const uint64_t blockCount64 =
            (dataset.sites + blockSize - 1) / blockSize;
        if (blockCount64 > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("HIP grid exceeds uint32_t range");
        }
        blockCount = static_cast<uint32_t>(blockCount64);

        try {
            hipDeviceProp_t properties{};
            int device = 0;
            checkHip(hipGetDevice(&device), "hipGetDevice");
            checkHip(
                hipGetDeviceProperties(&properties, device),
                "hipGetDeviceProperties"
            );
            if (
                blockSize
                    > static_cast<uint32_t>(properties.maxThreadsPerBlock)
                || blockCount
                    > static_cast<uint32_t>(properties.maxGridSize[0])
            ) {
                throw std::invalid_argument(
                    "HIP launch dimensions exceed device limits"
                );
            }

            const auto allocationStart = std::chrono::steady_clock::now();
            allocate(
                deviceStates,
                dataset.states.size(),
                "hipMalloc states"
            );
            allocate(
                devicePartitions,
                dataset.partitions.size(),
                "hipMalloc partitions"
            );
            allocate(
                deviceSpeciesRanges,
                dataset.speciesRanges.size(),
                "hipMalloc species ranges"
            );
            allocate(
                deviceSitePartitions,
                dataset.sitePartitions.size(),
                "hipMalloc site partitions"
            );
            allocate(
                deviceOperations,
                maxOperations,
                "hipMalloc operations"
            );
            allocate(
                deviceCounts,
                dataset.initialCounts.size(),
                "hipMalloc counts"
            );
            allocate(
                devicePartials,
                maxScores * blockCount,
                "hipMalloc partial scores"
            );
            checkHip(
                hipHostMalloc(
                    reinterpret_cast<void**>(&hostOperations),
                    maxOperations * sizeof(Operation)
                ),
                "hipHostMalloc operations"
            );
            checkHip(
                hipHostMalloc(
                    reinterpret_cast<void**>(&hostPartials),
                    maxScores * blockCount * sizeof(double)
                ),
                "hipHostMalloc partial scores"
            );
            checkHip(hipDeviceSynchronize(), "allocation synchronize");
            const auto allocationStop = std::chrono::steady_clock::now();
            allocationMillisecondsValue =
                elapsedMilliseconds(allocationStart, allocationStop);

            deviceAllocationBytesValue =
                dataset.states.size() * sizeof(uint8_t)
                + dataset.partitions.size() * sizeof(Partition)
                + dataset.speciesRanges.size() * sizeof(SpeciesRange)
                + dataset.sitePartitions.size() * sizeof(uint32_t)
                + maxOperations * sizeof(Operation)
                + dataset.initialCounts.size() * sizeof(uint16_t)
                + maxScores * blockCount * sizeof(double);

            const auto uploadStart = std::chrono::steady_clock::now();
            copyToDevice(
                deviceStates,
                dataset.states,
                "hipMemcpy states"
            );
            copyToDevice(
                devicePartitions,
                dataset.partitions,
                "hipMemcpy partitions"
            );
            copyToDevice(
                deviceSpeciesRanges,
                dataset.speciesRanges,
                "hipMemcpy species ranges"
            );
            copyToDevice(
                deviceSitePartitions,
                dataset.sitePartitions,
                "hipMemcpy site partitions"
            );
            copyToDevice(
                deviceCounts,
                dataset.initialCounts,
                "hipMemcpy counts"
            );
            checkHip(hipDeviceSynchronize(), "static upload synchronize");
            const auto uploadStop = std::chrono::steady_clock::now();
            staticUploadMillisecondsValue =
                elapsedMilliseconds(uploadStart, uploadStop);
        }
        catch (...) {
            release();
            throw;
        }
    }

    PortableHipExecutor(const PortableHipExecutor&) = delete;
    PortableHipExecutor& operator=(const PortableHipExecutor&) = delete;

    ~PortableHipExecutor() {
        release();
    }

    HipBatchResult execute(const TapeBatch& batch) {
        if (!usable) {
            throw std::runtime_error("HIP executor is unusable after failure");
        }
        if (
            batch.operations.size() > maxOperations
            || batch.scoreCount > maxScores
        ) {
            throw std::invalid_argument("HIP batch exceeds executor capacity");
        }
        std::vector<int8_t> nextColors = colors;
        validateTapeBatch(dataset, batch, nextColors);
        std::copy(
            batch.operations.begin(),
            batch.operations.end(),
            hostOperations
        );

        std::chrono::steady_clock::time_point tapeUploadStart;
        std::chrono::steady_clock::time_point tapeUploadStop;
        std::chrono::steady_clock::time_point kernelStart;
        std::chrono::steady_clock::time_point kernelStop;
        std::chrono::steady_clock::time_point downloadStart;
        std::chrono::steady_clock::time_point downloadStop;
        try {
            tapeUploadStart = std::chrono::steady_clock::now();
            checkHip(
                hipMemcpy(
                    deviceOperations,
                    hostOperations,
                    batch.operations.size() * sizeof(Operation),
                    hipMemcpyHostToDevice
                ),
                "hipMemcpy operations"
            );
            tapeUploadStop = std::chrono::steady_clock::now();

            kernelStart = std::chrono::steady_clock::now();
            hipLaunchKernelGGL(
                executeTapeKernel,
                dim3(blockCount),
                dim3(blockSize),
                blockSize * sizeof(double),
                0,
                deviceStates,
                devicePartitions,
                deviceSpeciesRanges,
                deviceSitePartitions,
                dataset.sites,
                deviceOperations,
                batch.operations.size(),
                deviceCounts,
                devicePartials
            );
            checkHip(hipGetLastError(), "executeTapeKernel launch");
            checkHip(
                hipDeviceSynchronize(),
                "executeTapeKernel synchronize"
            );
            kernelStop = std::chrono::steady_clock::now();

            const size_t partialCount = batch.scoreCount * blockCount;
            downloadStart = std::chrono::steady_clock::now();
            checkHip(
                hipMemcpy(
                    hostPartials,
                    devicePartials,
                    partialCount * sizeof(double),
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy partial scores"
            );
            downloadStop = std::chrono::steady_clock::now();
        }
        catch (...) {
            usable = false;
            throw;
        }

        const auto reductionStart = std::chrono::steady_clock::now();
        std::vector<double> scores(batch.scoreCount, 0.0);
        for (size_t score = 0; score < batch.scoreCount; ++score) {
            for (uint32_t block = 0; block < blockCount; ++block) {
                scores[score] +=
                    hostPartials[score * blockCount + block];
            }
        }
        const auto reductionStop = std::chrono::steady_clock::now();
        colors.swap(nextColors);

        return {
            std::move(scores),
            elapsedMilliseconds(tapeUploadStart, tapeUploadStop),
            elapsedMilliseconds(kernelStart, kernelStop),
            elapsedMilliseconds(downloadStart, downloadStop),
            elapsedMilliseconds(reductionStart, reductionStop)
        };
    }

    std::vector<uint16_t> downloadCounts() const {
        std::vector<uint16_t> counts(dataset.initialCounts.size());
        checkHip(
            hipMemcpy(
                counts.data(),
                deviceCounts,
                counts.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost
            ),
            "hipMemcpy final counts"
        );
        return counts;
    }

    double allocationMilliseconds() const {
        return allocationMillisecondsValue;
    }

    double staticUploadMilliseconds() const {
        return staticUploadMillisecondsValue;
    }

    size_t deviceAllocationBytes() const {
        return deviceAllocationBytesValue;
    }

    uint32_t blocks() const {
        return blockCount;
    }

private:
    template<typename T>
    static void allocate(T*& pointer, size_t count, const char* operation) {
        checkHip(
            hipMalloc(
                reinterpret_cast<void**>(&pointer),
                count * sizeof(T)
            ),
            operation
        );
    }

    template<typename T>
    static void copyToDevice(
        T* destination,
        const std::vector<T>& source,
        const char* operation
    ) {
        checkHip(
            hipMemcpy(
                destination,
                source.data(),
                source.size() * sizeof(T),
                hipMemcpyHostToDevice
            ),
            operation
        );
    }

    static double elapsedMilliseconds(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& stop
    ) {
        return std::chrono::duration<double, std::milli>(
            stop - start
        ).count();
    }

    void release() {
        if (hostPartials != nullptr) {
            (void) hipHostFree(hostPartials);
            hostPartials = nullptr;
        }
        if (hostOperations != nullptr) {
            (void) hipHostFree(hostOperations);
            hostOperations = nullptr;
        }
        if (devicePartials != nullptr) {
            (void) hipFree(devicePartials);
            devicePartials = nullptr;
        }
        if (deviceCounts != nullptr) {
            (void) hipFree(deviceCounts);
            deviceCounts = nullptr;
        }
        if (deviceOperations != nullptr) {
            (void) hipFree(deviceOperations);
            deviceOperations = nullptr;
        }
        if (deviceSitePartitions != nullptr) {
            (void) hipFree(deviceSitePartitions);
            deviceSitePartitions = nullptr;
        }
        if (deviceSpeciesRanges != nullptr) {
            (void) hipFree(deviceSpeciesRanges);
            deviceSpeciesRanges = nullptr;
        }
        if (devicePartitions != nullptr) {
            (void) hipFree(devicePartitions);
            devicePartitions = nullptr;
        }
        if (deviceStates != nullptr) {
            (void) hipFree(deviceStates);
            deviceStates = nullptr;
        }
    }
};

}

#endif
