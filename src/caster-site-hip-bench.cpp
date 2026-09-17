#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define CASTER_HD __host__ __device__
#else
#define CASTER_HD
#endif

namespace caster_hip {

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

static_assert(sizeof(Operation) == 8, "Operation must have a stable device layout");

struct Config {
    size_t sites = 131072;
    uint32_t taxa = 128;
    size_t moves = 512;
    size_t scoreEvery = 32;
    uint64_t seed = 233;
    uint32_t blockSize = 256;
    uint32_t iterations = 5;
};

struct Dataset {
    Config config;
    std::vector<uint8_t> states;
    std::vector<int8_t> initialColors;
    std::vector<uint16_t> initialCounts;
    std::vector<Operation> tape;
    std::array<float, 4> frequencies;
    size_t scoreCount;
};

struct Result {
    std::vector<uint16_t> counts;
    std::vector<double> scores;
    double milliseconds;
};

CASTER_HD inline int64_t xxyy(
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

CASTER_HD inline double scorePosition(const uint16_t* counts, const float* frequencies) {
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

void applyUpdate(uint16_t* counts, uint8_t base, int8_t from, int8_t to) {
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

Dataset generateDataset(const Config& config) {
    if (config.sites == 0 || config.taxa == 0) {
        throw std::invalid_argument("sites and taxa must be positive");
    }
    if (config.taxa > std::numeric_limits<uint16_t>::max()) {
        throw std::invalid_argument("taxa exceed uint16_t counter capacity");
    }
    if (config.scoreEvery == 0) {
        throw std::invalid_argument("score-every must be positive");
    }

    Dataset dataset;
    dataset.config = config;
    dataset.frequencies = {0.29f, 0.21f, 0.21f, 0.29f};
    dataset.states.resize(static_cast<size_t>(config.taxa) * config.sites);

    std::mt19937_64 random(config.seed);
    std::uniform_int_distribution<int> baseDistribution(0, 99);
    for (uint8_t& state : dataset.states) {
        const int value = baseDistribution(random);
        state = value < 5 ? 4 : static_cast<uint8_t>((value - 5) % 4);
    }

    std::uniform_int_distribution<uint32_t> taxonDistribution(0, config.taxa - 1);
    std::uniform_int_distribution<int> groupDistribution(0, 2);
    dataset.initialColors.resize(config.taxa);
    dataset.initialCounts.resize(config.sites * 12, 0);

    for (uint32_t taxon = 0; taxon < config.taxa; ++taxon) {
        const int8_t group = static_cast<int8_t>(groupDistribution(random));
        dataset.initialColors[taxon] = group;
        for (size_t site = 0; site < config.sites; ++site) {
            const uint8_t base = dataset.states[
                static_cast<size_t>(taxon) * config.sites + site
            ];
            if (base <= 3) {
                ++dataset.initialCounts[
                    site * 12 + static_cast<size_t>(group) * 4 + base
                ];
            }
        }
    }
    dataset.tape.push_back({0, -1, -1, OperationKind::ScoreTripartition, 0});

    std::vector<int8_t> colors = dataset.initialColors;
    for (size_t move = 0; move < config.moves; ++move) {
        const uint32_t taxon = taxonDistribution(random);
        const int8_t from = colors[taxon];
        const int8_t to = static_cast<int8_t>(
            (from + 1 + groupDistribution(random) % 2) % 3
        );
        dataset.tape.push_back({taxon, from, to, OperationKind::Update, 0});
        colors[taxon] = to;
        if ((move + 1) % config.scoreEvery == 0) {
            dataset.tape.push_back({0, -1, -1, OperationKind::ScoreTripartition, 0});
        }
    }
    if (dataset.tape.back().kind != OperationKind::ScoreTripartition) {
        dataset.tape.push_back({0, -1, -1, OperationKind::ScoreTripartition, 0});
    }

    dataset.scoreCount = static_cast<size_t>(std::count_if(
        dataset.tape.begin(),
        dataset.tape.end(),
        [](const Operation& operation) {
            return operation.kind == OperationKind::ScoreTripartition;
        }
    ));
    return dataset;
}

void validateTape(const Dataset& dataset) {
    std::vector<int8_t> colors = dataset.initialColors;
    for (const Operation& operation : dataset.tape) {
        if (operation.kind == OperationKind::ScoreTripartition) {
            continue;
        }
        if (operation.kind != OperationKind::Update) {
            throw std::invalid_argument("unknown operation kind");
        }
        if (operation.taxon >= dataset.config.taxa) {
            throw std::invalid_argument("operation taxon is out of range");
        }
        if (operation.from < -1 || operation.from > 2 || operation.to < -1 || operation.to > 2) {
            throw std::invalid_argument("operation group is out of range");
        }
        if (colors[operation.taxon] != operation.from) {
            throw std::invalid_argument("operation source does not match current color");
        }
        colors[operation.taxon] = operation.to;
    }
}

std::vector<double> reduceScores(
    const std::vector<double>& siteScores,
    size_t scoreCount,
    size_t sites,
    uint32_t blockSize
) {
    const size_t blockCount = (sites + blockSize - 1) / blockSize;
    std::vector<double> scores(scoreCount, 0.0);
    std::vector<double> lanes(blockSize);

    for (size_t score = 0; score < scoreCount; ++score) {
        for (size_t block = 0; block < blockCount; ++block) {
            for (uint32_t lane = 0; lane < blockSize; ++lane) {
                const size_t site = block * blockSize + lane;
                lanes[lane] = site < sites ? siteScores[score * sites + site] : 0.0;
            }
            for (uint32_t stride = blockSize / 2; stride > 0; stride /= 2) {
                for (uint32_t lane = 0; lane < stride; ++lane) {
                    lanes[lane] += lanes[lane + stride];
                }
            }
            scores[score] += lanes[0];
        }
    }
    return scores;
}

Result runCpu(const Dataset& dataset) {
    const auto start = std::chrono::steady_clock::now();
    const size_t sites = dataset.config.sites;
    std::vector<uint16_t> counts(sites * 12, 0);
    std::vector<double> siteScores(dataset.scoreCount * sites, 0.0);

    for (size_t site = 0; site < sites; ++site) {
        uint16_t localCounts[12];
        std::copy(
            dataset.initialCounts.begin() + site * 12,
            dataset.initialCounts.begin() + (site + 1) * 12,
            localCounts
        );
        size_t scoreIndex = 0;
        for (const Operation& operation : dataset.tape) {
            if (operation.kind == OperationKind::Update) {
                const uint8_t base = dataset.states[
                    static_cast<size_t>(operation.taxon) * sites + site
                ];
                applyUpdate(localCounts, base, operation.from, operation.to);
            }
            else {
                siteScores[scoreIndex * sites + site] = scorePosition(
                    localCounts,
                    dataset.frequencies.data()
                );
                ++scoreIndex;
            }
        }
        std::copy(localCounts, localCounts + 12, counts.begin() + site * 12);
    }

    std::vector<double> scores = reduceScores(
        siteScores,
        dataset.scoreCount,
        sites,
        dataset.config.blockSize
    );
    const auto stop = std::chrono::steady_clock::now();
    return {
        std::move(counts),
        std::move(scores),
        std::chrono::duration<double, std::milli>(stop - start).count()
    };
}

uint64_t checksum(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t result = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        result ^= bytes[index];
        result *= 1099511628211ULL;
    }
    return result;
}

void printCpuResult(const Dataset& dataset, const Result& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "sites=" << dataset.config.sites << '\n'
              << "taxa=" << dataset.config.taxa << '\n'
              << "operations=" << dataset.tape.size() << '\n'
              << "score_operations=" << dataset.scoreCount << '\n'
              << "cpu_ms=" << result.milliseconds << '\n'
              << "counter_checksum=" << checksum(
                    result.counts.data(),
                    result.counts.size() * sizeof(uint16_t)
                 ) << '\n'
              << "score_checksum=" << checksum(
                    result.scores.data(),
                    result.scores.size() * sizeof(double)
                 ) << '\n';
}

#ifdef __HIPCC__

void checkHip(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error)
        );
    }
}

__global__ void executeTape(
    const uint8_t* states,
    size_t sites,
    const Operation* tape,
    size_t operationCount,
    const float* frequencies,
    size_t scoreCount,
    const uint16_t* initialCounts,
    uint16_t* finalCounts,
    double* partialScores
) {
    extern __shared__ double lanes[];
    const uint32_t lane = threadIdx.x;
    const size_t site = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
    const bool active = site < sites;
    uint16_t counts[12] = {};
    if (active) {
        for (size_t index = 0; index < 12; ++index) {
            counts[index] = initialCounts[site * 12 + index];
        }
    }
    size_t scoreIndex = 0;

    for (size_t operationIndex = 0; operationIndex < operationCount; ++operationIndex) {
        const Operation operation = tape[operationIndex];
        if (operation.kind == OperationKind::Update) {
            if (active) {
                const uint8_t base = states[
                    static_cast<size_t>(operation.taxon) * sites + site
                ];
                if (base <= 3) {
                    if (operation.from >= 0) {
                        --counts[static_cast<size_t>(operation.from) * 4 + base];
                    }
                    if (operation.to >= 0) {
                        ++counts[static_cast<size_t>(operation.to) * 4 + base];
                    }
                }
            }
            continue;
        }

        lanes[lane] = active ? scorePosition(counts, frequencies) : 0.0;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                lanes[lane] += lanes[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0 && scoreIndex < scoreCount) {
            partialScores[scoreIndex * gridDim.x + blockIdx.x] = lanes[0];
        }
        ++scoreIndex;
        __syncthreads();
    }

    if (active) {
        for (size_t index = 0; index < 12; ++index) {
            finalCounts[site * 12 + index] = counts[index];
        }
    }
}

struct HipResult {
    Result result;
    double setupMilliseconds;
    double allocationMilliseconds;
    double uploadMilliseconds;
    double kernelMilliseconds;
    double downloadMilliseconds;
    double reductionMilliseconds;
    double totalMilliseconds;
    double benchmarkWallMilliseconds;
    std::string deviceName;
};

HipResult runHip(const Dataset& dataset) {
    const auto totalStart = std::chrono::steady_clock::now();
    const size_t sites = dataset.config.sites;
    const size_t blockCount = (sites + dataset.config.blockSize - 1) / dataset.config.blockSize;
    const size_t statesBytes = dataset.states.size() * sizeof(uint8_t);
    const size_t tapeBytes = dataset.tape.size() * sizeof(Operation);
    const size_t frequenciesBytes = dataset.frequencies.size() * sizeof(float);
    const size_t countsBytes = sites * 12 * sizeof(uint16_t);
    const size_t partialBytes = dataset.scoreCount * blockCount * sizeof(double);

    hipDeviceProp_t properties;
    checkHip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    const auto setupStop = std::chrono::steady_clock::now();

    uint8_t* deviceStates = nullptr;
    Operation* deviceTape = nullptr;
    float* deviceFrequencies = nullptr;
    uint16_t* deviceInitialCounts = nullptr;
    uint16_t* deviceCounts = nullptr;
    double* devicePartials = nullptr;

    const auto allocationStart = std::chrono::steady_clock::now();
    const auto cleanup = [&]() {
        hipFree(deviceStates);
        hipFree(deviceTape);
        hipFree(deviceFrequencies);
        hipFree(deviceInitialCounts);
        hipFree(deviceCounts);
        hipFree(devicePartials);
    };
    try {
        checkHip(hipMalloc(&deviceStates, statesBytes), "hipMalloc states");
        checkHip(hipMalloc(&deviceTape, tapeBytes), "hipMalloc tape");
        checkHip(hipMalloc(&deviceFrequencies, frequenciesBytes), "hipMalloc frequencies");
        checkHip(
            hipMalloc(&deviceInitialCounts, countsBytes),
            "hipMalloc initial counts"
        );
        checkHip(hipMalloc(&deviceCounts, countsBytes), "hipMalloc counts");
        checkHip(hipMalloc(&devicePartials, partialBytes), "hipMalloc partial scores");
        const auto allocationStop = std::chrono::steady_clock::now();

        const auto uploadStart = std::chrono::steady_clock::now();
        checkHip(
            hipMemcpy(deviceStates, dataset.states.data(), statesBytes, hipMemcpyHostToDevice),
            "hipMemcpy states"
        );
        checkHip(
            hipMemcpy(deviceTape, dataset.tape.data(), tapeBytes, hipMemcpyHostToDevice),
            "hipMemcpy tape"
        );
        checkHip(
            hipMemcpy(
                deviceFrequencies,
                dataset.frequencies.data(),
                frequenciesBytes,
                hipMemcpyHostToDevice
            ),
            "hipMemcpy frequencies"
        );
        checkHip(
            hipMemcpy(
                deviceInitialCounts,
                dataset.initialCounts.data(),
                countsBytes,
                hipMemcpyHostToDevice
            ),
            "hipMemcpy initial counts"
        );
        const auto uploadStop = std::chrono::steady_clock::now();

        const dim3 blocks(static_cast<uint32_t>(blockCount));
        const dim3 threads(dataset.config.blockSize);
        const size_t sharedBytes = dataset.config.blockSize * sizeof(double);

        hipLaunchKernelGGL(
            executeTape,
            blocks,
            threads,
            sharedBytes,
            0,
            deviceStates,
            sites,
            deviceTape,
            dataset.tape.size(),
            deviceFrequencies,
            dataset.scoreCount,
            deviceInitialCounts,
            deviceCounts,
            devicePartials
        );
        checkHip(hipGetLastError(), "warmup launch");
        checkHip(hipDeviceSynchronize(), "warmup synchronization");

        hipEvent_t kernelStart;
        hipEvent_t kernelStop;
        checkHip(hipEventCreate(&kernelStart), "hipEventCreate start");
        checkHip(hipEventCreate(&kernelStop), "hipEventCreate stop");
        checkHip(hipEventRecord(kernelStart), "hipEventRecord start");
        for (uint32_t iteration = 0; iteration < dataset.config.iterations; ++iteration) {
            hipLaunchKernelGGL(
                executeTape,
                blocks,
                threads,
                sharedBytes,
                0,
                deviceStates,
                sites,
                deviceTape,
                dataset.tape.size(),
                deviceFrequencies,
                dataset.scoreCount,
                deviceInitialCounts,
                deviceCounts,
                devicePartials
            );
            checkHip(hipGetLastError(), "benchmark launch");
        }
        checkHip(hipEventRecord(kernelStop), "hipEventRecord stop");
        checkHip(hipEventSynchronize(kernelStop), "hipEventSynchronize");
        float kernelTotalMilliseconds = 0.0f;
        checkHip(
            hipEventElapsedTime(&kernelTotalMilliseconds, kernelStart, kernelStop),
            "hipEventElapsedTime"
        );
        checkHip(hipEventDestroy(kernelStart), "hipEventDestroy start");
        checkHip(hipEventDestroy(kernelStop), "hipEventDestroy stop");

        const auto downloadStart = std::chrono::steady_clock::now();
        std::vector<uint16_t> counts(sites * 12);
        std::vector<double> partials(dataset.scoreCount * blockCount);
        checkHip(
            hipMemcpy(counts.data(), deviceCounts, countsBytes, hipMemcpyDeviceToHost),
            "hipMemcpy counts"
        );
        checkHip(
            hipMemcpy(partials.data(), devicePartials, partialBytes, hipMemcpyDeviceToHost),
            "hipMemcpy partial scores"
        );
        const auto downloadStop = std::chrono::steady_clock::now();

        const auto reductionStart = std::chrono::steady_clock::now();
        std::vector<double> scores(dataset.scoreCount, 0.0);
        for (size_t score = 0; score < dataset.scoreCount; ++score) {
            for (size_t block = 0; block < blockCount; ++block) {
                scores[score] += partials[score * blockCount + block];
            }
        }
        const auto reductionStop = std::chrono::steady_clock::now();
        const double kernelMilliseconds =
            kernelTotalMilliseconds / dataset.config.iterations;
        const double setupMilliseconds =
            std::chrono::duration<double, std::milli>(setupStop - totalStart).count();
        const double allocationMilliseconds =
            std::chrono::duration<double, std::milli>(
                allocationStop - allocationStart
            ).count();
        const double uploadMilliseconds =
            std::chrono::duration<double, std::milli>(uploadStop - uploadStart).count();
        const double downloadMilliseconds =
            std::chrono::duration<double, std::milli>(
                downloadStop - downloadStart
            ).count();
        const double reductionMilliseconds =
            std::chrono::duration<double, std::milli>(
                reductionStop - reductionStart
            ).count();

        HipResult output{
            {
                std::move(counts),
                std::move(scores),
                kernelMilliseconds
            },
            setupMilliseconds,
            allocationMilliseconds,
            uploadMilliseconds,
            kernelMilliseconds,
            downloadMilliseconds,
            reductionMilliseconds,
            setupMilliseconds
                + allocationMilliseconds
                + uploadMilliseconds
                + kernelMilliseconds
                + downloadMilliseconds
                + reductionMilliseconds,
            std::chrono::duration<double, std::milli>(
                reductionStop - totalStart
            ).count(),
            properties.name
        };

        cleanup();
        return output;
    }
    catch (...) {
        cleanup();
        throw;
    }
}

void validateHipResult(const Result& cpu, const Result& gpu) {
    if (cpu.counts != gpu.counts) {
        for (size_t index = 0; index < cpu.counts.size(); ++index) {
            if (cpu.counts[index] != gpu.counts[index]) {
                throw std::runtime_error(
                    "counter mismatch at index " + std::to_string(index)
                );
            }
        }
        throw std::runtime_error("counter mismatch");
    }
    if (cpu.scores.size() != gpu.scores.size()) {
        throw std::runtime_error("score count mismatch");
    }

    constexpr double absoluteTolerance = 1e-6;
    constexpr double relativeTolerance = 1e-12;
    double maximumAbsoluteDifference = 0.0;
    double maximumRelativeDifference = 0.0;
    for (size_t index = 0; index < cpu.scores.size(); ++index) {
        const double absoluteDifference = std::abs(cpu.scores[index] - gpu.scores[index]);
        const double scale = std::max(std::abs(cpu.scores[index]), std::abs(gpu.scores[index]));
        const double relativeDifference = scale == 0.0 ? 0.0 : absoluteDifference / scale;
        maximumAbsoluteDifference = std::max(maximumAbsoluteDifference, absoluteDifference);
        maximumRelativeDifference = std::max(maximumRelativeDifference, relativeDifference);
        if (absoluteDifference > absoluteTolerance + relativeTolerance * scale) {
            throw std::runtime_error(
                "score mismatch at index " + std::to_string(index)
            );
        }
    }
    std::cout << std::scientific << std::setprecision(12)
              << "max_score_abs_diff=" << maximumAbsoluteDifference << '\n'
              << "max_score_rel_diff=" << maximumRelativeDifference << '\n';
}

#endif

size_t parseSize(const char* value, const char* option) {
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (value[consumed] != '\0') {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<size_t>(parsed);
    }
    catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
}

Config parseArguments(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            std::cout
                << "Usage: caster-site-hip-bench [options]\n"
                << "  --sites N\n"
                << "  --taxa N\n"
                << "  --moves N\n"
                << "  --score-every N\n"
                << "  --seed N\n"
                << "  --block-size N\n"
                << "  --iterations N\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        const size_t value = parseSize(argv[++index], option.c_str());
        if (option == "--sites") {
            config.sites = value;
        }
        else if (option == "--taxa") {
            if (value > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("taxa exceed uint32_t range");
            }
            config.taxa = static_cast<uint32_t>(value);
        }
        else if (option == "--moves") {
            config.moves = value;
        }
        else if (option == "--score-every") {
            config.scoreEvery = value;
        }
        else if (option == "--seed") {
            config.seed = value;
        }
        else if (option == "--block-size") {
            if (value > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("block-size exceeds uint32_t range");
            }
            config.blockSize = static_cast<uint32_t>(value);
        }
        else if (option == "--iterations") {
            if (value > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("iterations exceed uint32_t range");
            }
            config.iterations = static_cast<uint32_t>(value);
        }
        else {
            throw std::invalid_argument("unknown option " + option);
        }
    }

    if (config.blockSize == 0
        || config.blockSize > 1024
        || (config.blockSize & (config.blockSize - 1)) != 0) {
        throw std::invalid_argument("block-size must be a power of two up to 1024");
    }
    if (config.iterations == 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    return config;
}

}  // namespace caster_hip

int main(int argc, char** argv) {
    try {
        const caster_hip::Config config = caster_hip::parseArguments(argc, argv);
        const caster_hip::Dataset dataset = caster_hip::generateDataset(config);
        caster_hip::validateTape(dataset);

        const caster_hip::Result cpu = caster_hip::runCpu(dataset);
        const caster_hip::Result repeatedCpu = caster_hip::runCpu(dataset);
        if (cpu.counts != repeatedCpu.counts || cpu.scores != repeatedCpu.scores) {
            throw std::runtime_error("CPU reference is not deterministic");
        }
        caster_hip::printCpuResult(dataset, cpu);

#ifdef __HIPCC__
        const caster_hip::HipResult gpu = caster_hip::runHip(dataset);
        caster_hip::validateHipResult(cpu, gpu.result);
        std::cout << "device=" << gpu.deviceName << '\n'
                  << "setup_ms=" << gpu.setupMilliseconds << '\n'
                  << "allocation_ms=" << gpu.allocationMilliseconds << '\n'
                  << "upload_ms=" << gpu.uploadMilliseconds << '\n'
                  << "kernel_ms=" << gpu.kernelMilliseconds << '\n'
                  << "download_ms=" << gpu.downloadMilliseconds << '\n'
                  << "host_reduction_ms=" << gpu.reductionMilliseconds << '\n'
                  << "gpu_total_ms=" << gpu.totalMilliseconds << '\n'
                  << "benchmark_wall_ms=" << gpu.benchmarkWallMilliseconds << '\n'
                  << "resident_speedup=" << cpu.milliseconds / gpu.kernelMilliseconds << '\n'
                  << "end_to_end_speedup=" << cpu.milliseconds / gpu.totalMilliseconds << '\n'
                  << "validation=passed\n";
#else
        std::cout << "validation=cpu-only\n";
#endif
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 1;
    }
}
