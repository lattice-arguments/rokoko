use std::sync::LazyLock;

fn env_disabled(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub static HAS_AVX512DQ: LazyLock<bool> = LazyLock::new(|| {
    let disable_avx512dq = env_disabled("HEXL_DISABLE_AVX512DQ");
    if disable_avx512dq {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
});

pub static HAS_AVX512IFMA: LazyLock<bool> = LazyLock::new(|| {
    let disable_avx512dq = env_disabled("HEXL_DISABLE_AVX512DQ");
    let disable_avx512ifma = env_disabled("HEXL_DISABLE_AVX512IFMA");
    if disable_avx512dq || disable_avx512ifma {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512ifma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
});

pub static HAS_AVX512VBMI2: LazyLock<bool> = LazyLock::new(|| {
    let disable_avx512dq = env_disabled("HEXL_DISABLE_AVX512DQ");
    let disable_avx512vbmi2 = env_disabled("HEXL_DISABLE_AVX512VBMI2");
    if disable_avx512dq || disable_avx512vbmi2 {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512vbmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
});
