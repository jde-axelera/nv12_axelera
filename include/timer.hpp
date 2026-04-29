#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct SectionTimer {
    std::string name;
    std::vector<double> samples;

    void record(double ms) { samples.push_back(ms); }

    double avg() const { return samples.empty() ? 0 : std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(); }
    double min() const { return samples.empty() ? 0 : *std::min_element(samples.begin(), samples.end()); }
    double max() const { return samples.empty() ? 0 : *std::max_element(samples.begin(), samples.end()); }
    double p95() const {
        if (samples.empty()) return 0;
        auto s = samples; std::sort(s.begin(), s.end());
        return s[static_cast<size_t>(s.size() * 0.95)];
    }

    void print() const {
        std::printf("  %-20s  avg=%7.3f ms  min=%7.3f ms  max=%7.3f ms  p95=%7.3f ms\n",
                    name.c_str(), avg(), min(), max(), p95());
    }
};

// RAII scope timer -- appends elapsed ms to SectionTimer on destruction
struct ScopeTimer {
    SectionTimer& st;
    std::chrono::time_point<Clock> t0;
    explicit ScopeTimer(SectionTimer& s) : st(s), t0(Clock::now()) {}
    ~ScopeTimer() { st.record(Ms(Clock::now() - t0).count()); }
};
