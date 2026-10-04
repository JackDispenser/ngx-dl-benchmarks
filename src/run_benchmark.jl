import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Distributed
using ArgParse
#=
  Command-line argument parser (read arguments)

  Example:
  > julia run_benchmark.jl --arithmetic=fp16,fp16+fp32,posit8_2 --model=lenet5 --dataset=cifar10
=#
function parse_cmdline_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--arithmetic"
        default = "fp16" # Comma-separated list of arithmetic types (fp16, bf16, posit16, posit8, takum16, ...)
        help = "Comma-separated list of arithmetic types (e.g. fp16,bf16). For a mixed-precision typing, separate " *
               "the two types with a plus (e.g. fp16+fp32), with the 'larger' type second."

        "--model"
        default = "lenet5" # Model architecture (e.g., resnet18, mlp)
        help = "Model, or a comma-separated list paired one-to-one with --dataset (e.g. lenet5,chimera)"

        "--dataset"
        default = "mnist" # Dataset to train on (e.g., cifar10, mnist)
        help = "Dataset, or a comma-separated list paired one-to-one with --model (e.g. mnist,fashionmnist)"

        "--epochs"
        help = "Number of epochs"
        arg_type = Int
        default = 10

        "--batch-size"
        help = "Batch size"
        arg_type = Int
        default = 64

        "--seed"
        help = "Random seed, or a comma-separated list (e.g. 0,1,2,3,4); each seed is a separate job. " *
               "Seed 0 keeps the original results path; other seeds go in a seed<k> subfolder."
        default = "0"

        "--patience"
        help = "Epochs the model will run without requiring improvement"
        arg_type = Int
        default = 3

        "--improvement-epsilon"
        help = "Minimum improvement the model requires to keep running"
        arg_type = Float64
        default = 0.001

        "--workers"
        help = "Number of worker processes (one job each at a time). -1 = one per job, capped at the " *
               "number of CPU threads minus one."
        arg_type = Int
        default = -1
    end
    return parse_args(s)
end


warmup_args = parse_cmdline_args()

arith_list = String.(split(warmup_args["arithmetic"], ","))
model_list = String.(split(warmup_args["model"], ","))
dataset_list = lowercase.(String.(split(warmup_args["dataset"], ",")))
seed_list = parse.(Int, split(warmup_args["seed"], ","))
length(model_list) == length(dataset_list) ||
    error("--model and --dataset must list the same number of entries (they are paired one-to-one)")


function job_cost_rank(arith)
    compute = first(split(replace(arith, "-sr" => ""), "+"))
    compute in ("fp32", "fp16") && return 2
    occursin(r"^(posit8|takum8|cfloat8|e4m3|e5m2)", compute) && return 1
    return 0
end
const DATASET_SIZE = Dict("emnistbalanced" => 112_800, "svhn2" => 73_257, "mnist" => 60_000,
                          "fashionmnist" => 60_000, "cifar10" => 50_000)
jobs = [(model = m, dataset = d, arith = a, seed = s)
        for (m, d) in zip(model_list, dataset_list) for a in arith_list for s in seed_list]
sort!(jobs; by = j -> (job_cost_rank(j.arith), -get(DATASET_SIZE, j.dataset, 0)), alg = MergeSort)

worker_count = warmup_args["workers"]
if (worker_count == -1) worker_count = min(length(jobs), max(1, Sys.CPU_THREADS - 1)) end
println("$(length(jobs)) jobs on $worker_count workers ($(Sys.CPU_THREADS) CPU threads)")

addprocs(worker_count; env = ["OPENBLAS_NUM_THREADS" => "1"])

@everywhere const RESULTS_DIR = joinpath(dirname(@__DIR__), "results")

# Results subfolder (and CSV "Number_Format" label) for one run
@everywhere run_label(arith_label::AbstractString, seed::Integer) =
    seed == 0 ? String(arith_label) : "$(arith_label)/seed$(seed)"

@everywhere begin
    using Lux
    using UniversalNumbers
    using DataFrames
    using CSV
    using MLDatasets
    using OneHotArrays
    using Statistics
    using Random, WeightInitializers, Optimisers, Zygote, MLUtils
    using Printf, JLD2
    using Accessors

    include("model_definitions.jl")
    include("utilities.jl")
    include("typeminmaxsupport.jl")
    include("csv_handling.jl")
end


@everywhere const SR_SUFFIX = "-sr"

@everywhere const ARITH_TYPES = Dict(
    "fp16"        => Float16,
    "fp32"        => Float32,
    "bf16"        => BF16,
    "posit8_0"    => Posit{8, 0, UInt8},
    "posit8_1"    => Posit{8, 1, UInt8},
    "posit8_2"    => Posit{8, 2, UInt8},
    "posit12_1"   => Posit{12, 1, UInt16},
    "posit16_1"   => Posit{16, 1, UInt16},
    "posit16_2"   => Posit{16, 2, UInt16},
    "posit19_2"   => Posit{19, 2, UInt32},
    "posit19_3"   => Posit{19, 3, UInt32},
    "posit32_2"   => Posit{32, 2, UInt32},
    "posit64_2"   => Posit{64, 2, UInt64},
    "posit64_3"   => Posit{64, 3, UInt64},
    "cfloat8_2"   => CFloat{8, 2, UInt8},
    "cfloat8_3"   => CFloat{8, 3, UInt8},
    "cfloat8_4"   => CFloat{8, 4, UInt8},
    "cfloat8_5"   => CFloat{8, 5, UInt8},
    "e5m2"        => E5M2{UInt8},             # OCP FP8 E5M2 (same type as cfloat8_5)
    "takum8"      => Takum{8, UInt8},
    "takum16"     => Takum{16, UInt16},
    "takum32"     => Takum{32, UInt32},
)


@everywhere const MODEL_TYPES = Dict(
    "lenet5"            => (LeNet5,            Lux.CrossEntropyLoss()),
    "smalldropoutnin"   => (SmallDropoutNIN,   Lux.CrossEntropyLoss()),
    "smallbatchnormnin" => (SmallBatchNormNIN, Lux.CrossEntropyLoss()),
    "resnet18"          => (ResNet18,          Lux.CrossEntropyLoss(logits = true)),
    "tinyresnet"        => (TinyResNet,        Lux.CrossEntropyLoss(logits = true)),
    "squeezenet1"       => (SqueezeNet1,       Lux.CrossEntropyLoss(logits = true)),
    "tinysqueezenet"    => (TinySqueezeNet,    Lux.CrossEntropyLoss(logits = true)),
    "vitbase"           => (VitBase,           Lux.CrossEntropyLoss(logits = true)),
    "microscopicvit"    => (MicroscopicVit,    Lux.CrossEntropyLoss(logits = true)),
    "chimera"           => (Chimera,           Lux.CrossEntropyLoss(logits = true)),
)

@everywhere const DATASETS = Dict(
    "cifar10"        => (CIFAR10,      ),
    "cifar100fine"   => (CIFAR100,     :fine),
    "cifar100coarse" => (CIFAR100,     :coarse),
    "emnistbalanced" => (EMNIST,       :balanced),
    "emnistletters"  => (EMNIST,       :letters),
    "emnistdigits"   => (EMNIST,       :digits),
    "fashionmnist"   => (FashionMNIST, ),
    "mnist"          => (MNIST,        ),
    "svhn2"          => (SVHN2,        )
)


for dataset_name in unique(dataset_list)
    haskey(DATASETS, dataset_name) ||
        error("Unknown dataset: $dataset_name. Valid options: $(join(keys(DATASETS), ", "))")
    dataset_info = DATASETS[dataset_name]
    println("Initializing $dataset_name...")
    remotecall_fetch(first(workers()), dataset_info) do info
        dataset = info[1]
        if dataset == EMNIST
            dataset(info[2], split=:train)
            dataset(info[2], split=:test)
        else
            dataset(split=:train)
            dataset(split=:test)
        end
        nothing
    end
end

@everywhere function get_results(d, t, ps, st, model)
    test_data_loader = DataLoader((d, t), batchsize = 64)
    total = 0.
    matched = 0.
    for (x, y) in test_data_loader
        outputs, _ = model(x, ps, Lux.testmode(st))
        predictions = onecold(outputs)
        targets = onecold(y)
        total += length(targets)
        matched += sum(predictions .== targets)
    end
    return matched / total
end

@everywhere function step_training(loss::Lux.AbstractLossFunction, data::Tuple, train_state)
    _, _, _, train_state = Training.single_train_step!(
                           AutoZygote(),
                           loss,
                           data,
                           train_state
                        )
    return train_state
end


@everywhere function save_checkpoint(path::String; kwargs...)
    tmp = path * ".tmp"
    jldsave(tmp; kwargs...)
    mv(tmp, path; force = true)
end


@everywhere function load_dataset(dataset_info)
    dataset = dataset_info[1]

    if dataset == EMNIST
        train_data = dataset(dataset_info[2], split = :train)
        test_data = dataset(dataset_info[2], split = :test)
    else
        train_data = dataset(split = :train)
        test_data = dataset(split = :test)
    end

    train_data_f, train_data_t = train_data[:]
    test_data_f, test_data_t = test_data[:]

    if dataset == CIFAR100
        train_data_t, test_data_t = getproperty(train_data_t, dataset_info[2]),
                                    getproperty(test_data_t, dataset_info[2])
    end

 
    if ndims(train_data_f) == 3
        train_data_f = reshape(train_data_f, size(train_data_f, 1), size(train_data_f, 2), 1, :)
        test_data_f = reshape(test_data_f, size(test_data_f, 1), size(test_data_f, 2), 1, :)
    end

    labels = unique(vcat(train_data_t, test_data_t))
    train_data_t = onehotbatch(train_data_t, labels);
    test_data_t = onehotbatch(test_data_t, labels);

    for channel in (1:size(train_data_f, 3))
        data_mean = mean(train_data_f[:, :, channel, :])
        data_std = std(train_data_f[:, :, channel, :])
        if (data_std == 0.0) data_std = one(data_std) end
        @. train_data_f[:, :, channel, :] = (train_data_f[:, :, channel, :] - data_mean) / (data_std)
        @. test_data_f[:, :, channel, :] = (test_data_f[:, :, channel, :] - data_mean) / (data_std)
    end

    return train_data_f, train_data_t, test_data_f, test_data_t
end

#=
  Training function
=#
@everywhere function benchmark_model(arith_label::String, opt, model::String, dataset::String;
                          epochs::Int=10, batch_size::Int=64, seed::Int=0,
                          patience::Int=3, improvement_epsilon::AbstractFloat=0.001)
                          
    base_label = String(first(split(chopsuffix(arith_label, SR_SUFFIX), "+")))
    T = get(ARITH_TYPES, base_label) do
        error("Unknown arithmetic type: $base_label. Valid options: $(join(keys(ARITH_TYPES), ", "))")
    end
    m = get(MODEL_TYPES, model) do
        error("Unknown model type: $model. Valid options: $(join(keys(MODEL_TYPES), ", "))")
    end
    d = get(DATASETS, dataset) do
        error("Unknown dataset: $dataset. Valid options: $(join(keys(DATASETS), ", "))")
    end

    report_path = joinpath(RESULTS_DIR, model, dataset, run_label(arith_label, seed)) * "/"

    tag = "[$model/$dataset/$(run_label(arith_label, seed))]"

    mkpath(report_path)
    adaptor(m) = (Lux.LuxEltypeAdaptor{T}())(m)

    println("Running benchmark with:")
    opt isa MixedPrecision ?
        println("Arithmetic: $arith_label -> Mixed ($T, $(typeof(opt).parameters[1]))") :
        println("Arithmetic: $arith_label -> $T")
    println("Model: $model")
    println("Dataset: $dataset")
    println("Epochs: $epochs, Batch size: $batch_size")
    completion_message = ""

    train_data_f, train_data_t, test_data_f, test_data_t = load_dataset(d)
    train_data_f, test_data_f = adaptor(train_data_f), adaptor(test_data_f)

    model_built = m[1](size(train_data_t, 1), size(train_data_f, 3))
    loss = m[2]

    local reload_epoch, ps, st, train_state, opt_state, epochs_accuracy, bestAccModel, done, rng
    checkpoint_path = report_path * "progress_controller.jld2"
    if isfile(checkpoint_path)
        reload_epoch, ps, st, opt_state, epochs_accuracy, bestAccModel, done, rng = JLD2.load(
                                        checkpoint_path,
                                        "reload_epoch", "ps", "st", "opt_state", "epochs_accuracy", "bestAccModel", "done", "rng")
        train_state = Training.TrainState(model_built, ps, st, opt)
        @reset train_state.optimizer_state = opt_state
    else
        println("Generating fresh model data")
        reload_epoch = 1
        rng = Xoshiro(seed)
        ps, st = LuxCore.setup(rng, model_built)
        ps, st = adaptor(ps), adaptor(st)
        train_state = Training.TrainState(model_built, ps, st, opt)
        opt_state = train_state.optimizer_state
        # epochs_accuracy[1][:] => training set accuracy
        # epochs_accuracy[2][:] => testing set accuracy
        epochs_accuracy = (fill(Float64(0.0), epochs + 1),
                           fill(Float64(0.0), epochs + 1))
        bestAccModel = (0, 0)
        done = false
        save_checkpoint(checkpoint_path;
                reload_epoch, ps, st, opt_state, epochs_accuracy, bestAccModel, done, rng)
    end

    if done
        completion_message = "$tag is already complete! \n"
        print(completion_message)
        return completion_message
    end

    train_data_loader = DataLoader((train_data_f, train_data_t), batchsize = batch_size, shuffle = true, rng = rng)

    if reload_epoch > 1
        println("Resuming from epoch $(reload_epoch)")
    end
    
    halted = :no
    for epoch in reload_epoch:epochs
        stime = time()
        iter = 0
        reload_epoch = epoch
        unchanged_count, param_count = 0, 0
        ever_changed = nothing   # per weight: did it change at any step this epoch?
        for (x, y) in train_data_loader
            iter += 1
            before = copy.(param_arrays(train_state.parameters))
            try
                train_state = step_training(loss, (x, y), train_state)
            catch e
                completion_message = "Model break detected in $tag by epoch $epoch: $e \n"
                print(completion_message)
                halted = :error
            end
            # Count weights the step left exactly unchanged (update rounded away)
            after = param_arrays(train_state.parameters)
            unchanged_count += sum(sum(a .== b) for (a, b) in zip(after, before); init = 0)
            param_count += sum(length, after; init = 0)
   
            ever_changed === nothing && (ever_changed = [falses(size(a)) for a in after])
            for (m, a, b) in zip(ever_changed, after, before)
                m .|= (a .!= b)
            end
            for layer in train_state.parameters
                if (!finite_test(layer))
                    completion_message = "Model break detected in $tag by epoch $epoch, iter $iter: Non-finite weights. \n"
                    print(completion_message)
                    halted = :error
                    break
                end
                if (!type_test(layer, T))
                    completion_message = "Model break detected in $tag by epoch $epoch, iter $iter: Type promotion. \n"
                    print(completion_message)
                    halted = :error
                    break
                end
            end
            if halted != :no
                break
            end
        end
        # results
        train_accuracy = get_results(train_data_f, train_data_t, train_state.parameters, train_state.states, model_built)
        epochs_accuracy[1][epoch + 1] = train_accuracy
        test_accuracy = get_results(test_data_f, test_data_t, train_state.parameters, train_state.states, model_built)
        epochs_accuracy[2][epoch + 1] = test_accuracy
        if (test_accuracy > bestAccModel[1] + improvement_epsilon)
            bestAccModel = (test_accuracy, epoch)
        elseif (epoch > bestAccModel[2] + patience)
            completion_message = "Early stop for $tag at epoch $(bestAccModel[2]). \n"
            print(completion_message)
            halted = :earlystop
        end
        unchanged_frac = param_count == 0 ? 0.0 : unchanged_count / param_count
        never_frac = ever_changed === nothing ? 0.0 :
                     1 - sum(sum, ever_changed) / sum(length, ever_changed)
        open(report_path * "update_stats.csv", "a") do io
            epoch == 1 && println(io, "epoch,unchanged_per_step,never_changed_in_epoch")
            println(io, epoch, ",", unchanged_frac, ",", never_frac)
        end
        if halted != :error
            @printf("%s Epoch %i, Type %s, time: %.2f seconds, train accuracy: %.2f, test accuracy: %.2f, unchanged per step: %.1f%%, never changed: %.1f%%\n",
                    tag, epoch, opt isa MixedPrecision ? "Mixed: ($T, $(typeof(opt).parameters[1]))" : "$T",
                    (time() - stime), (train_accuracy * 100), (test_accuracy * 100),
                    (unchanged_frac * 100), (never_frac * 100))
        else
            println("Saving failed model for inspection...")
            ps = train_state.parameters
            st = train_state.states
            opt_state = train_state.optimizer_state
            jldsave(report_path * "failed_model.jld2";
                    reload_epoch, ps, st, opt_state, epochs_accuracy, bestAccModel, done, rng)
        end
        ps = train_state.parameters
        st = train_state.states
        opt_state = train_state.optimizer_state
        reload_epoch = epoch + 1
        save_checkpoint(checkpoint_path;
                        reload_epoch, ps, st, opt_state, epochs_accuracy, bestAccModel, done, rng)
        if halted != :no
            break
        end
    end
    done = true
    if halted != :error
        ps = train_state.parameters
        st = train_state.states
        opt_state = train_state.optimizer_state
    end
    save_checkpoint(checkpoint_path;
                reload_epoch, ps, st, opt_state, epochs_accuracy, bestAccModel, done, rng)
    return completion_message
end

@everywhere function obtain_optimizer(arith::String, seed::Int = 0)
    # "-sr" suffix: same optimiser, with the final weight update rounded stochastically
    sr = endswith(arith, SR_SUFFIX)
    arith = String(chopsuffix(arith, SR_SUFFIX))
    base_rule = OptimiserChain(WeightDecay(0.001), YunAdam(; eta = 0.001, epsilon = 1e-4))
    rule = sr ? StochasticRounding(base_rule; seed = seed) : base_rule   # own rounding stream per seed

    parts = split(arith, "+")
    if length(parts) == 1
        return rule
    end
    if length(parts) != 2
        error("Mixed precision requires exactly two formats (received $arith)")
    end
    if parts[1] == parts[2]
        println("Warning: attempted mixed-precision with $(parts[1]) as both types. Running in uniform-precision.")
        return rule
    end
    T = get(ARITH_TYPES, parts[2]) do
        error("Unknown arithmetic type: $(parts[2]). Valid options: $(join(keys(ARITH_TYPES), ", "))")
    end
    return MixedPrecision(T, rule)
end


function main()
    args = warmup_args

    results = pmap(jobs; on_error = e -> "Job failed with error: $e \n") do job
        benchmark_model(job.arith, obtain_optimizer(job.arith, job.seed), job.model, job.dataset;
                        epochs=args["epochs"], batch_size=args["batch-size"], seed=job.seed,
                        patience=args["patience"], improvement_epsilon=args["improvement-epsilon"])
    end


    for (model, dataset) in unique(zip(model_list, dataset_list))
        report_path = joinpath(RESULTS_DIR, model, dataset) * "/"
        idx = [i for (i, j) in enumerate(jobs) if j.model == model && j.dataset == dataset]
        labels = [run_label(jobs[i].arith, jobs[i].seed) for i in idx]
        logs_for = String.(results[idx])
        csv_log = CSVPrinter(report_path, labels)
        if !(csv_log == "") push!(logs_for, "\n~~~~\n" * csv_log) end
        open(report_path * "log.txt", "a") do logs
            foreach(l -> write(logs, l), logs_for)
        end
    end

    return results
end

main()