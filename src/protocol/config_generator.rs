use crate::protocol::commitment::{Prefix, RecursionConfig};
use crate::protocol::config::{
    Config, Projection, SimpleConfig, SumcheckConfig, Type1ProjectionConfig,
};

#[derive(Clone)]
pub struct AuxRecursionConfig {
    pub decomposition_base_log: usize,
    pub decomposition_chunks: usize,
    pub rank: usize,
    pub next: Option<Box<AuxRecursionConfig>>,
}

#[derive(Clone)]
pub enum AuxProjection {
    Type0(AuxRecursionConfig),
    Type1 {
        nof_batches: usize,
        recursion_constant_term: AuxRecursionConfig,
        recursion_batched_projection: AuxRecursionConfig,
    },
    Skip
}

#[derive(Clone)]
pub enum AuxConfig {
    Sumcheck(AuxSumcheckConfig),
    Simple(SimpleConfig), // no prefix assignment needed
}

#[derive(Clone)]
pub struct AuxSumcheckConfig {
    pub witness_height: usize,
    pub witness_width: usize,
    pub projection_ratio: usize,
    pub projection_height: usize,
    pub basic_commitment_rank: usize,
    pub nof_openings: usize,
    pub commitment_recursion: AuxRecursionConfig,
    pub opening_recursion: AuxRecursionConfig,
    pub projection_recursion: AuxProjection,
    pub witness_decomposition_base_log: usize,
    pub witness_decomposition_chunks: usize,
    pub next: Option<Box<AuxConfig>>,
}

#[derive(Clone)]
struct ComponentInfo {
    name: String,
    size: usize,
    path: Vec<String>,
}

impl AuxSumcheckConfig {
    pub fn generate_config(&self) -> Config {
        self.generate_config_inner(0)
    }
    pub fn generate_config_inner(&self, depth: usize) -> Config {
        let mut components = Vec::new();

        // Collect all components that need prefixes
        self.collect_components(&mut components);

        // Calculate required composed_witness_length (round up to nearest power of 2)
        let total_size: usize = components.iter().map(|c| c.size).sum();
        let composed_witness_length = total_size.next_power_of_two();

        // Sort by size (largest to smallest)
        components.sort_by(|a, b| b.size.cmp(&a.size));

        println!("\n=== Prefix Assignment level {} ===", depth);
        println!(
            "Total size needed: {} -> Composed witness length: {} (compresion ratio {:.2}%)",
            total_size,
            composed_witness_length,
            (composed_witness_length as f64 / (self.witness_width * self.witness_height) as f64)
                * 100.0
        );
        println!("\nComponents sorted by size:");

        let total_bits = composed_witness_length.ilog2() as usize;
        let mut assigned_prefixes = Vec::new();
        let mut used_prefixes = std::collections::HashSet::new();

        for comp in &components {
            let required_bits = comp.size.ilog2() as usize;
            let prefix_length = total_bits - required_bits;

            // Find an unused prefix
            let mut prefix_value = 0;
            let max_prefix = 1 << prefix_length;

            while prefix_value < max_prefix {
                // Check if this prefix conflicts with any already assigned
                let start = prefix_value << required_bits;
                let end = start + comp.size;

                let mut conflict = false;
                for i in start..end {
                    if used_prefixes.contains(&i) {
                        conflict = true;
                        break;
                    }
                }

                if !conflict {
                    // Mark this range as used
                    for i in start..end {
                        used_prefixes.insert(i);
                    }
                    break;
                }

                prefix_value += 1;
            }

            if prefix_value >= max_prefix {
                panic!("Could not find a valid prefix for component: {}", comp.name);
            }

            let prefix = Prefix {
                prefix: prefix_value,
                length: prefix_length,
            };

            assigned_prefixes.push((comp.clone(), prefix));

            let prefix_binary = format!("{:0width$b}", prefix_value, width = prefix_length);
            let start = prefix_value << required_bits;
            let end = start + comp.size;

            // Calculate indentation based on path depth
            let indent_level = comp.path.iter().filter(|s| *s == "next").count();
            let indent = "  ".repeat(indent_level + 1);

            println!(
                "{}{} (size={}): prefix=0b{} (len={}) -> indices [{}, {}]",
                indent,
                comp.name,
                comp.size,
                prefix_binary,
                prefix_length,
                start,
                end - 1
            );
        }

        // Calculate usage ratio
        let used_memory = used_prefixes.len();
        let usage_ratio = used_memory as f64 / composed_witness_length as f64;
        println!("\n=== Memory Usage level {} ===", depth);
        println!("Used: {} / {}", used_memory, composed_witness_length);
        println!("Usage ratio: {:.2}%\n", usage_ratio * 100.0);

        // Build the actual config with assigned prefixes
        self.build_config_with_prefixes(
            &assigned_prefixes,
            composed_witness_length,
            usage_ratio,
            depth,
        )
    }

    fn collect_components(&self, components: &mut Vec<ComponentInfo>) {
        // Folded witness
        let folded_size = self.witness_height * self.witness_decomposition_chunks;
        components.push(ComponentInfo {
            name: "folded_witness".to_string(),
            size: folded_size,
            path: vec!["folded_witness".to_string()],
        });

        // Commitment recursion chain
        self.collect_recursion_components(
            &self.commitment_recursion,
            "commitment",
            self.basic_commitment_rank.next_power_of_two() * self.witness_width, // It seems that we cannot fit into 4 rank, but 8 seems too large, so we use next power of two for simplicity
            components,
            vec!["commitment_recursion".to_string()],
        );

        // Opening recursion chain
        self.collect_recursion_components(
            &self.opening_recursion,
            "opening",
            self.nof_openings * self.witness_width,
            components,
            vec!["opening_recursion".to_string()],
        );

        // Projection recursion
        match &self.projection_recursion {
            AuxProjection::Type0(config) => {
                let base_size = self.witness_height * self.witness_width / self.projection_ratio;
                self.collect_recursion_components(
                    config,
                    "projection",
                    base_size,
                    components,
                    vec!["projection_recursion".to_string()],
                );
            }
            AuxProjection::Type1 {
                nof_batches,
                recursion_constant_term,
                recursion_batched_projection,
            } => {
                let constant_term_size =
                    self.witness_height * self.witness_width / self.projection_ratio;
                self.collect_recursion_components(
                    recursion_constant_term,
                    "projection_constant_term",
                    constant_term_size,
                    components,
                    vec![
                        "projection_recursion".to_string(),
                        "constant_term".to_string(),
                    ],
                );

                let batched_size = self.witness_width * nof_batches;
                self.collect_recursion_components(
                    recursion_batched_projection,
                    "projection_batched",
                    batched_size,
                    components,
                    vec![
                        "projection_recursion".to_string(),
                        "batched_projection".to_string(),
                    ],
                );
            }
            AuxProjection::Skip => {
                // No components to collect
            }
        }
    }

    fn collect_recursion_components(
        &self,
        config: &AuxRecursionConfig,
        name_prefix: &str,
        base_size: usize,
        components: &mut Vec<ComponentInfo>,
        mut path: Vec<String>,
    ) {
        let size = base_size * config.decomposition_chunks;
        let depth = path.iter().filter(|s| *s == "next").count();

        let name = if depth == 0 {
            format!("{}", name_prefix)
        } else {
            format!("{}_level_{}", name_prefix, depth)
        };

        components.push(ComponentInfo {
            name,
            size,
            path: path.clone(),
        });

        if let Some(next) = &config.next {
            path.push("next".to_string());
            self.collect_recursion_components(next, name_prefix, config.rank, components, path);
        }
    }

    fn build_config_with_prefixes(
        &self,
        assigned_prefixes: &[(ComponentInfo, Prefix)],
        composed_witness_length: usize,
        usage_ratio: f64,
        depth: usize,
    ) -> Config {
        // Helper to find prefix by path
        let find_prefix = |path: &[String]| -> Prefix {
            assigned_prefixes
                .iter()
                .find(|(comp, _)| comp.path == path)
                .map(|(_, prefix)| prefix.clone())
                .expect(&format!("Prefix not found for path: {:?}", path))
        };

        // Build commitment recursion
        let commitment_recursion = self.build_recursion_config(
            &self.commitment_recursion,
            assigned_prefixes,
            &["commitment_recursion".to_string()],
        );

        // Build opening recursion
        let opening_recursion = self.build_recursion_config(
            &self.opening_recursion,
            assigned_prefixes,
            &["opening_recursion".to_string()],
        );

        // Build projection recursion
        let projection_recursion = match &self.projection_recursion {
            AuxProjection::Type0(config) => Projection::Type0(self.build_recursion_config(
                config,
                assigned_prefixes,
                &["projection_recursion".to_string()],
            )),
            AuxProjection::Type1 {
                nof_batches,
                recursion_constant_term,
                recursion_batched_projection,
            } => {
                let constant_term = self.build_recursion_config(
                    recursion_constant_term,
                    assigned_prefixes,
                    &[
                        "projection_recursion".to_string(),
                        "constant_term".to_string(),
                    ],
                );

                let batched_projection = self.build_recursion_config(
                    recursion_batched_projection,
                    assigned_prefixes,
                    &[
                        "projection_recursion".to_string(),
                        "batched_projection".to_string(),
                    ],
                );

                Projection::Type1(Type1ProjectionConfig {
                    nof_batches: *nof_batches,
                    recursion_constant_term: constant_term,
                    recursion_batched_projection: batched_projection,
                })
            }
            AuxProjection::Skip => Projection::Skip,
        };

        // Get folded witness prefix
        let folded_witness_prefix = find_prefix(&["folded_witness".to_string()]);

        Config::Sumcheck(SumcheckConfig {
            witness_height: self.witness_height,
            witness_width: self.witness_width,
            projection_ratio: self.projection_ratio,
            projection_height: self.projection_height,
            basic_commitment_rank: self.basic_commitment_rank,
            nof_openings: self.nof_openings,
            commitment_recursion,
            next_level_usage_ratio: usage_ratio,
            opening_recursion,
            projection_recursion,
            witness_decomposition_base_log: self.witness_decomposition_base_log,
            witness_decomposition_chunks: self.witness_decomposition_chunks,
            folded_witness_prefix,
            composed_witness_length,
            next: self.next.as_ref().map(|next| {
                Box::new(match next.as_ref() {
                    AuxConfig::Sumcheck(cfg) => cfg.generate_config_inner(depth + 1),
                    AuxConfig::Simple(cfg) => Config::Simple(cfg.clone()),
                })
            }),
        })
    }

    fn build_recursion_config(
        &self,
        aux_config: &AuxRecursionConfig,
        assigned_prefixes: &[(ComponentInfo, Prefix)],
        base_path: &[String],
    ) -> RecursionConfig {
        let prefix = assigned_prefixes
            .iter()
            .find(|(comp, _)| comp.path == base_path)
            .map(|(_, prefix)| prefix.clone())
            .expect(&format!("Prefix not found for path: {:?}", base_path));

        let next = if let Some(aux_next) = &aux_config.next {
            let mut next_path = base_path.to_vec();
            next_path.push("next".to_string());
            Some(Box::new(self.build_recursion_config(
                aux_next,
                assigned_prefixes,
                &next_path,
            )))
        } else {
            None
        };

        RecursionConfig {
            decomposition_base_log: aux_config.decomposition_base_log,
            decomposition_chunks: aux_config.decomposition_chunks,
            rank: aux_config.rank,
            prefix,
            next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_config_generation() {
        let aux_config = AuxSumcheckConfig {
            witness_height: 512,
            witness_width: 16,
            projection_ratio: 32,
            projection_height: 8,
            basic_commitment_rank: 2,
            nof_openings: 1,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8,
                    rank: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Type0(AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 2,
                rank: 1,
                next: None,
            }),
            witness_decomposition_base_log: 15,
            witness_decomposition_chunks: 2,
            next: None,
        };

        let _config = aux_config.generate_config();
    }

    #[test]
    fn test_toy_config_ii_generation() {
        let aux_config = AuxSumcheckConfig {
            witness_height: 512,
            witness_width: 16,
            projection_ratio: 64,
            projection_height: 8,
            basic_commitment_rank: 2,
            nof_openings: 1,
            commitment_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: Some(Box::new(AuxRecursionConfig {
                    decomposition_base_log: 7,
                    decomposition_chunks: 8,
                    rank: 1,
                    next: None,
                })),
            },
            opening_recursion: AuxRecursionConfig {
                decomposition_base_log: 15,
                decomposition_chunks: 4,
                rank: 1,
                next: None,
            },
            projection_recursion: AuxProjection::Type1 {
                nof_batches: 2,
                recursion_constant_term: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 2,
                    rank: 1,
                    next: None,
                },
                recursion_batched_projection: AuxRecursionConfig {
                    decomposition_base_log: 15,
                    decomposition_chunks: 4,
                    rank: 1,
                    next: None,
                },
            },
            witness_decomposition_base_log: 15,
            witness_decomposition_chunks: 2,
            next: None,
        };

        let _config = aux_config.generate_config();
    }
}
