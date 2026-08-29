# NGX-DL-Benchmarks

Benchmarking Low-Precision Arithmetic for DNN Training in Julia

![Julia](https://img.shields.io/badge/Julia-9558B2?logo=julia)
![Open Source](https://img.shields.io/badge/Open-Source-green)
![Research Code](https://img.shields.io/badge/Code-Research-blue)


![IEEE 754](https://img.shields.io/badge/IEEE754-FP16%20FP32%20FP64-blue)
![Posits](https://img.shields.io/badge/Posits-supported-success)
![Takum](https://img.shields.io/badge/Takum-supported-success)
![bfloat16](https://img.shields.io/badge/bfloat16-supported-success)
![OFP8](https://img.shields.io/badge/OFP8-supported-success)

## Overview

This repository provides reproducible benchmarks for training deep neural networks using **next-generation low-precision arithmetic formats**, such as:

- IEEE `FP16`  
- `bfloat16`  
- `posit16`
- `takum16`
- `FP8` with varying mantissa lengths

The goal is to systematically evaluate **stability, convergence, and performance tradeoffs** when training DNNs in low-precision, providing insights for both numerical analysis and machine learning research.



## Features

- Benchmark multiple arithmetic formats on standard neural network architectures  
- Measure **accuracy degradation**, **training stability**, and **performance metrics**  
- Easy to extend to new arithmetic formats or architectures  
- Fully reproducible with Julia scripts and configuration files  



## Installation

Requires Julia ≥ 1.9 and standard scientific packages. First, clone the repository:

```bash
git clone https://github.com/yourusername/ngx-dl-benchmarks.git
cd ngx-dl-benchmarks
````

Open Julia and activate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

**Key Julia packages used:**

* [Lux.jl](https://github.com/FluxML/Lux.jl) – Neural networks
* [UniversalNumbers.jl](https://github.com/jamesquinlan/UniversalNumbers.jl) – Includes: Posits, Takums, BFloats, etc.
* [Plots.jl](https://github.com/JuliaPlots/Plots.jl) – Plotting results
* [CSV.jl / DataFrames.jl](https://github.com/JuliaData) – Export results
* [Distributed.jl](https://github.com/JuliaLang/Distributed.jl) – Distributes training of multiple formats to different processes



## Usage

Run a benchmark with default settings:

```julia
julia run_benchmark.jl --arithmetic=fp16 --model=resnet18 --dataset=cifar10
```

### Testing examples

These are the commands run for the associated paper:

julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=lenet5 --dataset=mnist
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=smalldropoutnin --dataset=emnistbalanced
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=smallbatchnormnin --dataset=emnistbalanced
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=tinyresnet --dataset=cifar10
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=tinysqueezenet --dataset=cifar10
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=microscopicvit --dataset=svhn2
julia src/run_benchmark.jl --arithmetic=bf16,bf16+fp32,fp16+fp32,fp32,posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2,takum8,takum8+takum16,takum16 --model=chimera --dataset=fashionmnist

### Command-Line Options

* `--arithmetic`: `fp16`, `bfloat16`, `posit16`, `fp8`, `posit8`
* `--model`: neural network architecture (e.g., `resnet18`, `mlp`)
* `--dataset`: dataset (e.g., `cifar10`, `mnist`)
* `--epochs`: number of training epochs
* `--batch-size`: batch size
* `--seed`: random seed
* `--patience`: Used for early stopping; number of epochs the model will wait for an improvement
* `--improvement-epsilon`: Used for early stopping; minimum improvement for the model to reset patience
* `--workers`: Number of processes training will distribute over

Example benchmarking multiple arithmetic types:

```julia
julia run_benchmark.jl --arithmetic=fp16,fp16+fp32,posit8_2 --model=lenet5 --dataset=mnist
```



## Results

Benchmark outputs include:

* Per-epoch accuracy metrics for training and testing data

Results can be exported in CSV or JSON for further analysis.



## Contributing

Contributions are welcome! You can help by:

* Adding support for additional arithmetic formats
* Extending benchmarks to new architectures or datasets
* Improving reproducibility and logging

Please fork the repository and submit pull requests.



## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.



## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{ngx-dl-benchmarks2026,
  author       = {Nelson, David, Arciero, Michael and Quinlan, James},
  title        = {LP-NGX-DL-Benchmarks: Benchmarking Neural Network Training in Next-Generation Low-Precision Arithmetic},
  year         = {2026},
  howpublished = {\url{https://github.com/jamesquinlan/lp-ngx-dl-benchmarks}}
}
```



## Roadmap / Future Work

* Expand FP8 variants and custom posit configurations
* Integrate energy / hardware-level benchmarking
* Include advanced architectures (e.g. RNNs)
* Automated plots and LaTeX-ready summary tables for publications