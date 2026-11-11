//! Lie la bibliothèque sherpa-onnx (vendor/sherpa-onnx) et copie ses DLL à côté
//! des exécutables pour qu'ils se lancent sans réglage du PATH.

use std::{env, fs, path::PathBuf};

const DLLS: [&str; 3] = [
    "sherpa-onnx-c-api.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest.join("vendor").join("sherpa-onnx").join("lib");
    assert!(
        lib_dir.join("sherpa-onnx-c-api.lib").exists(),
        "sherpa-onnx introuvable dans {} (voir README, section Installation)",
        lib_dir.display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    println!("cargo:rerun-if-changed=build.rs");

    // OUT_DIR = target/<profil>/build/<paquet>-<hash>/out → target/<profil>
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir.ancestors().nth(3).unwrap().to_path_buf();
    for target in [profile_dir.clone(), profile_dir.join("deps")] {
        let _ = fs::create_dir_all(&target);
        for dll in DLLS {
            let _ = fs::copy(lib_dir.join(dll), target.join(dll));
        }
    }
}
