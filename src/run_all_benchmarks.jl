using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()


RUN = "all"
# RUN = "mnist"
# RUN = "emnist_dropout"
# RUN = "emnist_batchnorm"
# RUN = "cifar10_resnet"
# RUN = "cifar10_squeezenet"
# RUN = "svhn_vit"
# RUN = "fashion_chimera"


const ARITHMETIC =
    "bf16,bf16+fp32,fp16+fp32,fp32," *
    "posit8_2,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2," *
    "takum8,takum8+takum16,takum16"


const BENCHMARKS = Dict(
    "mnist" => (
        model = "lenet5",
        dataset = "mnist"
    ),

    "emnist_dropout" => (
        model = "smalldropoutnin",
        dataset = "emnistbalanced"
    ),

    "emnist_batchnorm" => (
        model = "smallbatchnormnin",
        dataset = "emnistbalanced"
    ),

    "cifar10_resnet" => (
        model = "tinyresnet",
        dataset = "cifar10"
    ),

    "cifar10_squeezenet" => (
        model = "tinysqueezenet",
        dataset = "cifar10"
    ),

    "svhn_vit" => (
        model = "microscopicvit",
        dataset = "svhn2"
    ),

    "fashion_chimera" => (
        model = "chimera",
        dataset = "fashionmnist"
    )
)


function run_benchmark(name)

    benchmark = BENCHMARKS[name]

    println()
    println("="^70)
    println("Running benchmark: $name")
    println("Model:   $(benchmark.model)")
    println("Dataset: $(benchmark.dataset)")
    println("="^70)
    println()

    script = joinpath(@__DIR__, "run_benchmark.jl")

    cmd = `$(Base.julia_cmd())
        $script
        --arithmetic=$ARITHMETIC
        --model=$(benchmark.model)
        --dataset=$(benchmark.dataset)`

    run(cmd)

    println()
    println("Completed: $name")
    println()
end


if RUN == "all"

    benchmarks = [
        "mnist",
        "emnist_dropout",
        "emnist_batchnorm",
        "cifar10_resnet",
        "cifar10_squeezenet",
        "svhn_vit",
        "fashion_chimera"
    ]

    for name in benchmarks
        run_benchmark(name)
    end

elseif haskey(BENCHMARKS, RUN)

    run_benchmark(RUN)

else

    error("""
    Unknown benchmark: $RUN

    Valid choices are:

        all
        mnist
        emnist_dropout
        emnist_batchnorm
        cifar10_resnet
        cifar10_squeezenet
        svhn_vit
        fashion_chimera
    """)

end

println()
println("="^70)
println("Finished.")
println("="^70)
