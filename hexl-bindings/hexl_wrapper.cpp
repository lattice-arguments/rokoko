#include <stdint.h>
#include <cstddef>
#include <hexl/hexl.hpp>
#include <unordered_map>
#include <memory>
#include <utility>
#include <immintrin.h>

class NTTCache { 
public:
    static intel::hexl::NTT& Get(size_t n, uint64_t modulus) {
        auto key = std::make_pair(n, modulus);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            // Construct and insert
            it = cache_.emplace(key, std::make_unique<intel::hexl::NTT>(n, modulus)).first;
        }
        return *(it->second);
    }

private:
    // Hash for std::pair
    struct pair_hash {
        std::size_t operator()(const std::pair<size_t, uint64_t>& p) const {
            return std::hash<size_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
        }
    };

    static inline std::unordered_map<std::pair<size_t, uint64_t>, std::unique_ptr<intel::hexl::NTT>, pair_hash> cache_;
};

extern "C" __attribute__((externally_visible)) uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return intel::hexl::MultiplyMod(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) 
    uint64_t add_mod(uint64_t a, uint64_t b, uint64_t modulus) {
        return intel::hexl::AddUIntMod(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return intel::hexl::SubUIntMod(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) const uint64_t* get_roots(size_t n, uint64_t modulus) {
    auto& ntt = NTTCache::Get(n, modulus);
    return ntt.GetRootOfUnityPowers().data(); 
}

extern "C" __attribute__((externally_visible)) const uint64_t inv_mod(uint64_t a, uint64_t modulus) {
    return intel::hexl::InverseMod(a, modulus);
}

extern "C" __attribute__((externally_visible)) const uint64_t* get_inv_roots(size_t n, uint64_t modulus) {
    auto& ntt = NTTCache::Get(n, modulus);
    return ntt.GetInvRootOfUnityPowers().data(); 
}

extern "C" __attribute__((externally_visible)) uint64_t power_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return intel::hexl::PowMod(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) void eltwise_mult_mod(uint64_t* result, const uint64_t* operand1, const uint64_t* operand2, uint64_t n, uint64_t modulus) {
    intel::hexl::EltwiseMultMod(result, operand1, operand2, n, modulus, 1);
}

extern "C" __attribute__((externally_visible)) void eltwise_fma_mod(uint64_t* result, const uint64_t* operand1, const uint64_t operand2, const uint64_t* operand3, uint64_t n, uint64_t modulus) {
    intel::hexl::EltwiseFMAMod(result, operand1, operand2, operand3, n, modulus, 1);
}

extern "C" __attribute__((externally_visible)) void eltwise_reduce_mod(
    uint64_t* result,
    const uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    intel::hexl::EltwiseReduceMod(result, operand, n, modulus, modulus, 1);
}


extern "C" __attribute__((externally_visible)) void eltwise_add_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t* operand2,
    size_t n,
    uint64_t modulus
) {
    intel::hexl::EltwiseAddMod(result, operand1, operand2, n, modulus);
}


extern "C" __attribute__((externally_visible)) void eltwise_sub_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t* operand2,
    size_t n,
    uint64_t modulus
) {
    intel::hexl::EltwiseSubMod(result, operand1, operand2, n, modulus);
}


#include <vector>
#include <hexl/ntt/ntt.hpp>

extern "C" __attribute__((externally_visible)) void ntt_forward_in_place(
    uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    auto& ntt = NTTCache::Get(n, modulus);
    ntt.ComputeForward(operand, operand, 1, 1);
}

extern "C" __attribute__((externally_visible)) void ntt_inverse_in_place(
    uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    auto& ntt = NTTCache::Get(n, modulus);
    ntt.ComputeInverse(operand, operand, 1, 1);
}
