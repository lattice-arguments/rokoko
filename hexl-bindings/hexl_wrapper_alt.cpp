#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using U128 = __uint128_t;

uint64_t mod_add_internal(uint64_t a, uint64_t b, uint64_t modulus) {
    const uint64_t res = a + b;
    if (res >= modulus || res < a) {
        return res - modulus;
    }
    return res;
}

uint64_t mod_sub_internal(uint64_t a, uint64_t b, uint64_t modulus) {
    return (a >= b) ? (a - b) : (modulus - (b - a));
}

uint64_t multiply_mod_internal(uint64_t a, uint64_t b, uint64_t modulus) {
    return static_cast<uint64_t>((static_cast<U128>(a) * b) % modulus);
}

uint64_t power_mod_internal(uint64_t base, uint64_t exp, uint64_t modulus) {
    uint64_t result = 1;
    uint64_t cur = base % modulus;
    while (exp > 0) {
        if (exp & 1) {
            result = multiply_mod_internal(result, cur, modulus);
        }
        cur = multiply_mod_internal(cur, cur, modulus);
        exp >>= 1;
    }
    return result;
}

uint64_t inv_mod_internal(uint64_t value, uint64_t modulus) {
    // Modulus is prime for our use, so Fermat's little theorem works.
    return power_mod_internal(value, modulus - 2, modulus);
}

std::vector<uint64_t> unique_prime_factors(uint64_t value) {
    std::vector<uint64_t> factors;
    if (value % 2 == 0) {
        factors.push_back(2);
        while (value % 2 == 0) {
            value /= 2;
        }
    }
    for (uint64_t p = 3; p * p <= value; p += 2) {
        if (value % p == 0) {
            factors.push_back(p);
            while (value % p == 0) {
                value /= p;
            }
        }
    }
    if (value > 1) {
        factors.push_back(value);
    }
    return factors;
}

uint64_t find_primitive_generator(uint64_t modulus) {
    const uint64_t phi = modulus - 1;
    const auto factors = unique_prime_factors(phi);
    for (uint64_t cand = 2; cand < modulus; ++cand) {
        bool ok = true;
        for (uint64_t factor : factors) {
            if (power_mod_internal(cand, phi / factor, modulus) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return cand;
        }
    }
    return 0;
}

unsigned int log2_size_t(size_t n) {
    unsigned int bits = 0;
    while ((1ULL << bits) < n) {
        ++bits;
    }
    return bits;
}

size_t reverse_bits(size_t value, unsigned int width) {
    size_t result = 0;
    for (unsigned int i = 0; i < width; ++i) {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    return result;
}

uint64_t minimal_primitive_root(uint64_t degree, uint64_t modulus) {
    // degree is 2 * n
    const uint64_t generator = find_primitive_generator(modulus);
    uint64_t root = power_mod_internal(generator, (modulus - 1) / degree, modulus);

    uint64_t min_root = root;
    uint64_t generator_sq = multiply_mod_internal(root, root, modulus);
    uint64_t current = root;
    for (uint64_t i = 0; i < degree; ++i) {
        if (current < min_root) {
            min_root = current;
        }
        current = multiply_mod_internal(current, generator_sq, modulus);
    }
    return min_root;
}

struct NTTParams {
    size_t n = 0;
    uint64_t modulus = 0;
    uint64_t primitive_root = 0;
    std::vector<uint64_t> root_powers;     // bit-reversed order
    std::vector<uint64_t> inv_root_powers; // reordered for inverse transform
};

struct PairHash {
    std::size_t operator()(const std::pair<size_t, uint64_t>& p) const {
        return std::hash<size_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
    }
};

class NTTCache {
public:
    static NTTParams& Get(size_t n, uint64_t modulus) {
        const auto key = std::make_pair(n, modulus);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            it = cache_.emplace(key, BuildParams(n, modulus)).first;
        }
        return it->second;
    }

private:
    static NTTParams BuildParams(size_t n, uint64_t modulus) {
        NTTParams params;
        params.n = n;
        params.modulus = modulus;
        params.primitive_root = minimal_primitive_root(2 * n, modulus);

        const unsigned int log_n = log2_size_t(n);

        params.root_powers.assign(n, 0);
        std::vector<uint64_t> inv_raw(n, 0);
        params.root_powers[0] = 1;
        inv_raw[0] = 1;
        size_t prev_idx = 0;
        for (size_t i = 1; i < n; ++i) {
            const size_t idx = reverse_bits(i, log_n);
            params.root_powers[idx] =
                multiply_mod_internal(params.root_powers[prev_idx], params.primitive_root, modulus);
            inv_raw[idx] = inv_mod_internal(params.root_powers[idx], modulus);
            prev_idx = idx;
        }

        params.inv_root_powers.assign(n, 0);
        params.inv_root_powers[0] = inv_raw[0];
        size_t fill_idx = 1;
        for (size_t m = (n >> 1); m > 0; m >>= 1) {
            for (size_t i = 0; i < m; ++i) {
                params.inv_root_powers[fill_idx++] = inv_raw[m + i];
            }
        }

        return params;
    }

    static inline std::unordered_map<std::pair<size_t, uint64_t>, NTTParams, PairHash> cache_;
};

void ntt_forward(uint64_t* data, size_t n, uint64_t modulus) {
    auto& params = NTTCache::Get(n, modulus);
    const auto& roots = params.root_powers;

    size_t t = n >> 1;
    uint64_t W = roots[1];
    for (size_t j = 0; j < t; ++j) {
        uint64_t y = multiply_mod_internal(data[j + t], W, modulus);
        uint64_t x = data[j];
        data[j] = mod_add_internal(x, y, modulus);
        data[j + t] = mod_sub_internal(x, y, modulus);
    }
    t >>= 1;

    for (size_t m = 2; m < n; m <<= 1) {
        for (size_t i = 0; i < m; ++i) {
            uint64_t W_m = roots[m + i];
            size_t offset = i * (t << 1);
            for (size_t j = 0; j < t; ++j) {
                size_t pos = offset + j;
                uint64_t y = multiply_mod_internal(data[pos + t], W_m, modulus);
                uint64_t x = data[pos];
                data[pos] = mod_add_internal(x, y, modulus);
                data[pos + t] = mod_sub_internal(x, y, modulus);
            }
        }
        t >>= 1;
    }
    // make sure that the output is reduced mod modulus
    for (size_t i = 0; i < n; ++i) {
        data[i] %= modulus;
    }
}

void ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) {
    auto& params = NTTCache::Get(n, modulus);
    const auto& inv_roots = params.inv_root_powers;

    size_t t = 1;
    size_t root_index = 1;
    size_t m = n >> 1;
    while (m > 1) {
        size_t offset = 0;
        for (size_t i = 0; i < m; ++i, ++root_index) {
            if (i != 0) {
                offset += (t << 1);
            }
            uint64_t W = inv_roots[root_index];
            for (size_t j = 0; j < t; ++j) {
                size_t pos = offset + j;
                uint64_t x = data[pos];
                uint64_t y = data[pos + t];
                uint64_t sum = mod_add_internal(x, y, modulus);
                uint64_t diff = mod_sub_internal(x, y, modulus);
                data[pos] = sum;
                data[pos + t] = multiply_mod_internal(diff, W, modulus);
            }
        }
        t <<= 1;
        m >>= 1;
    }

    uint64_t W = inv_roots[n - 1];
    uint64_t inv_n = inv_mod_internal(static_cast<uint64_t>(n % modulus), modulus);
    uint64_t inv_n_w = multiply_mod_internal(inv_n, W, modulus);
    size_t n_div_2 = n >> 1;
    for (size_t j = 0; j < n_div_2; ++j) {
        uint64_t x = data[j];
        uint64_t y = data[j + n_div_2];
        uint64_t sum = mod_add_internal(x, y, modulus);
        uint64_t diff = mod_sub_internal(x, y, modulus);
        data[j] = multiply_mod_internal(sum, inv_n, modulus);
        data[j + n_div_2] = multiply_mod_internal(diff, inv_n_w, modulus);
    }
}

}  // namespace

extern "C" __attribute__((externally_visible)) uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return multiply_mod_internal(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) uint64_t add_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return mod_add_internal(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return mod_sub_internal(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) const uint64_t* get_roots(size_t n, uint64_t modulus) {
    return NTTCache::Get(n, modulus).root_powers.data();
}

extern "C" __attribute__((externally_visible)) const uint64_t inv_mod(uint64_t a, uint64_t modulus) {
    return inv_mod_internal(a, modulus);
}

extern "C" __attribute__((externally_visible)) const uint64_t* get_inv_roots(size_t n, uint64_t modulus) {
    return NTTCache::Get(n, modulus).inv_root_powers.data();
}

extern "C" __attribute__((externally_visible)) uint64_t power_mod(uint64_t a, uint64_t b, uint64_t modulus) {
    return power_mod_internal(a, b, modulus);
}

extern "C" __attribute__((externally_visible)) void eltwise_mult_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t* operand2,
    uint64_t n,
    uint64_t modulus
) {
    for (uint64_t i = 0; i < n; ++i) {
        result[i] = multiply_mod_internal(operand1[i], operand2[i], modulus);
    }
}

extern "C" __attribute__((externally_visible)) void eltwise_fma_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t operand2,
    const uint64_t* operand3,
    uint64_t n,
    uint64_t modulus
) {
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t prod = multiply_mod_internal(operand1[i], operand2, modulus);
        result[i] = mod_add_internal(prod, operand3[i], modulus);
    }
}

extern "C" __attribute__((externally_visible)) void eltwise_reduce_mod(
    uint64_t* result,
    const uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = operand[i] % modulus;
    }
}

extern "C" __attribute__((externally_visible)) void eltwise_add_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t* operand2,
    size_t n,
    uint64_t modulus
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = mod_add_internal(operand1[i], operand2[i], modulus);
    }
}

extern "C" __attribute__((externally_visible)) void eltwise_sub_mod(
    uint64_t* result,
    const uint64_t* operand1,
    const uint64_t* operand2,
    size_t n,
    uint64_t modulus
) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = mod_sub_internal(operand1[i], operand2[i], modulus);
    }
}

extern "C" __attribute__((externally_visible)) void ntt_forward_in_place(
    uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    ntt_forward(operand, n, modulus);
}

extern "C" __attribute__((externally_visible)) void ntt_inverse_in_place(
    uint64_t* operand,
    size_t n,
    uint64_t modulus
) {
    ntt_inverse(operand, n, modulus);
}
