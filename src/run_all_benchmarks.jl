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


const SEEDS = "0"
# const SEEDS = "0,1,2,3,4"

# Worker processes; -1 = one per job, capped at CPU threads minus one
const WORKERS = -1

# "-sr" = same format with stochastic rounding of the weight update
const ARITHMETIC =
    "fp16,bf16,bf16+fp32,fp16+fp32,fp32," *
    "e5m2,e5m2-sr," *
    "posit8_2,posit8_2-sr,posit8_2+posit12_1,posit8_2+posit16_2,posit16_2," *
    "takum8,takum8-sr,takum8+takum16,takum16"


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


function run_benchmarks(names)

    models   = join((BENCHMARKS[n].model for n in names), ",")
    datasets = join((BENCHMARKS[n].dataset for n in names), ",")

    println()
    println("="^70)
    println("Running benchmarks: $(join(names, ", "))")
    println("Seeds: $SEEDS   Workers: $WORKERS")
    println("="^70)
    println()

    script = joinpath(@__DIR__, "run_benchmark.jl")

    cmd = `$(Base.julia_cmd())
        $script
        --arithmetic=$ARITHMETIC
        --model=$models
        --dataset=$datasets
        --seed=$SEEDS
        --workers=$WORKERS`

    run(cmd)
end


if RUN == "all"

    benchmarks = [
        "mnist",
        "emnist_dropout",
        "emnist_batchnorm",
        "fashion_chimera"
    ]

    run_benchmarks(benchmarks)

elseif haskey(BENCHMARKS, RUN)

    run_benchmarks([RUN])

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