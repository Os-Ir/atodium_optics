use rspirv_reflect::{DescriptorInfo, PushConstantInfo, Reflection};
use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use log::warn;

pub type DescriptorTemplate = BTreeMap<u32, BTreeMap<u32, DescriptorInfo>>;
pub type BindingMap = HashMap<String, ShaderBinding>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderBinding {
    pub set: u32,
    pub binding: u32,
    pub info: DescriptorInfo,
}

impl ShaderBinding {
    pub fn new(set: u32, binding: u32, info: DescriptorInfo) -> Self {
        Self { set, binding, info }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ShaderReflection {
    pub descriptor_template: DescriptorTemplate,
    pub push_constant_infos: Vec<PushConstantInfo>,
    pub binding_map: BindingMap,
}

impl ShaderReflection {
    pub fn new(shader_stages: &[&[u8]]) -> Result<Self> {
        let mut descriptor_template: DescriptorTemplate = BTreeMap::new();
        let mut push_constant_infos: Vec<PushConstantInfo> = vec![];

        for &shader_stage in shader_stages {
            let stage_reflection = Reflection::new_from_spirv(shader_stage)?;

            let descriptor_sets = stage_reflection.get_descriptor_sets()?;

            for (set, descriptor_set) in descriptor_sets {
                if let Some(existing_descriptor_set) = descriptor_template.get_mut(&set) {
                    for (binding, descriptor) in descriptor_set {
                        if let Some(existing_descriptor) = existing_descriptor_set.get(&binding) {
                            if descriptor.ty != existing_descriptor.ty || descriptor.name != existing_descriptor.name {
                                warn!("Descriptor inconsistent between shader stages at position [ set: {} | binding: {} ]", set, binding);
                            }
                        } else {
                            existing_descriptor_set.insert(binding, descriptor);
                        }
                    }
                } else {
                    descriptor_template.insert(set, descriptor_set);
                }
            }

            if let Some(push_constant_info) = stage_reflection.get_push_constant_range()? {
                push_constant_infos.push(push_constant_info);
            }
        }

        let binding_map: HashMap<String, ShaderBinding> = descriptor_template
            .iter()
            .flat_map(|(&set, descriptor_bindings)| {
                let bindings: HashMap<String, ShaderBinding> = descriptor_bindings
                    .iter()
                    .map(|(&binding, descriptor_info)| {
                        let name = if descriptor_info.name.is_empty() {
                            format!("internal_name::<{}, {}>", set, binding)
                        } else {
                            descriptor_info.name.clone()
                        };

                        (name, ShaderBinding::new(set, binding, descriptor_info.clone()))
                    })
                    .collect();

                bindings
            })
            .collect();

        Ok(Self {
            descriptor_template,
            push_constant_infos,
            binding_map,
        })
    }

    pub fn sub_binding_map(&self, set: u32) -> BindingMap {
        self.binding_map
            .iter()
            .filter_map(|(name, binding)| if binding.set == set { Some((name.clone(), binding.clone())) } else { None })
            .collect()
    }

    pub fn get_binding(&self, name: &str) -> Option<ShaderBinding> {
        self.binding_map.get(name).cloned()
    }
}
