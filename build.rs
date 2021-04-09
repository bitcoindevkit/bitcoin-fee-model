use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ModelData {
    pub norm: FieldsDescribe,
    pub weights: Weights,
    pub fields: Vec<String>,
    pub alpha: f32,
}

impl ModelData {
    fn into_src(self, model_name: &str) -> (HashSet<usize>, String) {
        let fields = self
            .fields
            .iter()
            .fold(String::new(), |acc, f| acc + "\"" + f + "\".to_string(), ");

        let i_size = self.fields.len();
        let l0_size = self.weights.l0_bias.len();
        let l1_size = self.weights.l1_bias.len();
        let o_size = self.weights.l2_bias.len();

        if l0_size != l1_size {
            panic!(
                "Layer 0 and Layer 1 must have the same number of neurons. Found: {}, {}",
                l0_size, l1_size
            );
        }
        if o_size != 1 {
            panic!("Layer 2 should only have one output. Found: {}", o_size);
        }

        let req_sizes = vec![i_size, l0_size, o_size].into_iter().collect();
        let src = format!(
            r#"
        pub fn get_model_{name}() -> ModelData<Size{i_size}, Size{n_size}, Size{o_size}> {{
            ModelData {{
                norm: {norm},
                weights: {weights},
                fields: vec![{fields}],
                alpha: {alpha},
            }}
        }}
        "#,
            i_size = i_size,
            n_size = l0_size,
            o_size = o_size,
            name = model_name,
            norm = self.norm.into_src(),
            weights = self.weights.into_src(),
            fields = fields,
            alpha = self.alpha
        );

        (req_sizes, src)
    }
}

#[derive(Deserialize, Debug)]
pub struct FieldsDescribe {
    mean: HashMap<String, f32>,
    std: HashMap<String, f32>,
}

impl FieldsDescribe {
    fn into_src(&self) -> String {
        let mean = self
            .mean
            .iter()
            .map(|(k, v)| format!("(\"{}\".to_string(), {})", k, v))
            .fold(String::new(), |acc, f| acc + &f + ", ");
        let std = self
            .std
            .iter()
            .map(|(k, v)| format!("(\"{}\".to_string(), {})", k, v))
            .fold(String::new(), |acc, f| acc + &f + ", ");

        format!(
            r#"
        FieldsDescribe {{
            mean: vec![{mean}].into_iter().collect(),
            std: vec![{std}].into_iter().collect(),
        }}
        "#,
            mean = mean,
            std = std
        )
    }
}

#[derive(Deserialize, Debug)]
pub struct Weights {
    #[serde(rename = "dense/bias:0")]
    pub l0_bias: Vec<f32>,
    #[serde(rename = "dense/kernel:0")]
    pub l0_kernel: Vec<Vec<f32>>,

    #[serde(rename = "dense_1/bias:0")]
    pub l1_bias: Vec<f32>,
    #[serde(rename = "dense_1/kernel:0")]
    pub l1_kernel: Vec<Vec<f32>>,

    #[serde(rename = "dense_2/bias:0")]
    pub l2_bias: Vec<f32>,
    #[serde(rename = "dense_2/kernel:0")]
    pub l2_kernel: Vec<Vec<f32>>,
}

fn compress_buffer(v: Vec<f32>) -> String {
    let v_bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let s = v_bytes
        .into_iter()
        .map(|c| std::ascii::escape_default(*c))
        .flatten()
        .map(|c| char::from(c))
        .collect::<String>();

    format!("b\"{}\"", s)
}

fn decompress_buffer(data: String) -> String {
    format!(
        r#"{{
        let data: Vec<u8> = {data}.to_vec();
        let v_floats = unsafe {{ Vec::from_raw_parts(data.as_ptr() as *mut f32, data.len() / 4, data.len() / 4) }};

        std::mem::forget(data);

        v_floats
    }}"#,
        data = data
    )
}

impl Weights {
    fn into_src(self) -> String {
        fn serialize_vec(field_name: &str, v: Vec<f32>) -> String {
            format!(
                r#"
                {field}: Matrix::from_buffer({data}.into_boxed_slice())
            "#,
                field = field_name,
                data = decompress_buffer(compress_buffer(v))
            )
        }
        fn serialize_matrix(field_name: &str, v: Vec<Vec<f32>>) -> String {
            format!(
                r#"
                {field}: Matrix::from_buffer({data}.into_boxed_slice())
            "#,
                field = field_name,
                data =
                    decompress_buffer(compress_buffer(v.into_iter().flatten().collect::<Vec<_>>()))
            )
        }

        format!(
            r#"
            Weights {{
                {l0_bias},
                {l0_kernel},
                {l1_bias},
                {l1_kernel},
                {l2_bias},
                {l2_kernel},
            }}
            "#,
            l0_bias = serialize_vec("l0_bias", self.l0_bias),
            l0_kernel = serialize_matrix("l0_kernel", self.l0_kernel),
            l1_bias = serialize_vec("l1_bias", self.l1_bias),
            l1_kernel = serialize_matrix("l1_kernel", self.l1_kernel),
            l2_bias = serialize_vec("l2_bias", self.l2_bias),
            l2_kernel = serialize_matrix("l2_kernel", self.l2_kernel),
        )
    }
}

fn emit_sizes_src(sizes: &HashSet<usize>) -> String {
    sizes
        .iter()
        .map(|s| {
            format!(
                r#"
                #[derive(Debug)]
                pub enum Size{s} {{}}
                impl crate::matrix::SizeMarker for Size{s} {{
                    fn size() -> usize {{
                        {s}
                    }}
                }}
                "#,
                s = s
            )
        })
        .fold(String::new(), |acc, x| acc + "\n" + &x)
}

fn add_model(
    path: &str,
    model_name: &str,
) -> Result<(HashSet<usize>, String), Box<dyn std::error::Error>> {
    let model: ModelData = serde_cbor::from_reader(File::open(path)?)?;
    println!("cargo:rerun-if-changed={}", path);

    Ok(model.into_src(model_name))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const CUSTOM_FEE_ERR: &str = "Custom models must be specified with a comma separated list of `<name>:<path>`, with no space in between. Trailing commas are not supported";

    let default_models = vec![
        ("test_model", "./models/test_model.cbor"),
        ("low", "./models/20210408-202241/model.cbor"),
        ("high", "./models/20210408-202237/model.cbor"),
    ];
    let extra_models = option_env!("CUSTOM_FEE_MODELS")
        .map(|s| s.split(','))
        .map(|ps| {
            ps.map(|p| p.split(':')).map(|mut p| {
                (
                    p.next().expect(CUSTOM_FEE_ERR),
                    p.next().expect(CUSTOM_FEE_ERR),
                )
            })
        })
        .map(|s| s.collect::<Vec<_>>());

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let models_path = Path::new(&out_dir).join("models.rs");
    let sizes_path = Path::new(&out_dir).join("sizes.rs");

    let mut sizes = vec![1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512]
        .into_iter()
        .collect::<HashSet<_>>();
    let mut models_file = File::create(models_path)?;

    for (name, path) in default_models
        .into_iter()
        .chain(extra_models.unwrap_or_default())
    {
        let (req_sizes, content) = add_model(path, name)?;

        models_file.write_all(content.as_bytes())?;
        sizes.extend(req_sizes);
    }

    fs::write(&sizes_path, emit_sizes_src(&sizes)).unwrap();

    Ok(())
}
