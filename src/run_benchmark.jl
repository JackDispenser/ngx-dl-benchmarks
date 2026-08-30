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

        "--dataset"
        default = "mnist" # Dataset to train on (e.g., cifar10, mnist)

        "--epochs"
        help = "Number of epochs"
        arg_type = Int
        default = 10

        "--batch-size"
        help = "Batch size"
        arg_type = Int
        default = 64

        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 0

        "--patience"
        help = "Epochs the model will run without requiring improvement"
        arg_type = Int
        default = 3

        "--improvement-epsilon"
        help = "Minimum improvement the model requires to keep running"
        arg_type = Float64
        default = 0.001

        "--workers"
        help = "Number of workers the model will use to parallelize different-format training; -1 assigns 1 worker per format"
        arg_type = Int
        default = -1
    end
    return parse_args(s)
end

# Warmup section, so the program can automatically assign the correct worker count
warmup_args = parse_cmdline_args()
arith_list = String.(split(warmup_args["arithmetic"], ","))

arith_list = [length(split(arith, "+")) == 1 ? arith : String.(split(arith, "+"))
                for arith in arith_list]
worker_count = warmup_args["workers"]
if (worker_count == -1) worker_count = length(arith_list) end
addprocs(worker_count)

@everywhere const RESULTS_DIR = joinpath(dirname(@__DIR__), "results")

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


# Map CLI arithmetic labels to the actual Julia numeric types
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
    "takum8"      => Takum{8, UInt8},
    "takum16"     => Takum{16, UInt16},
    "takum32"     => Takum{32, UInt32},
)

@everywhere arith_format(arith::AbstractString) = ARITH_TYPES[arith]
@everywhere arith_format(arith::AbstractVector) = (ARITH_TYPES[arith[1]], ARITH_TYPES[arith[2]])

# Maps CLI model labels to the actual model definition and matching loss function
# Note that some models are only present as reference, and are too large for practical
# emulated-arithmetic training; see model_definitions.jl
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


# Map CLI dataset labels to the actual dataset definition from MLDatasets.jl
# First entry in each tuple is the dataset call; remainder are special fields for handling of certain datasets
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

# Modification proposed by Claude from original system to minimize risk of failed saves
@everywhere function save_checkpoint(path::String; kwargs...)
    tmp = path * ".tmp"
    jldsave(tmp; kwargs...)
    mv(tmp, path; force = true)
end


# Loads and standardizes a dataset
# Some special handling exists for EMNIST and CIFAR100
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

    # Grayscale sets (MNIST, FashionMNIST, EMNIST) arrive as W×H×N with no channel
    # dimension; add it so normalization and the model both see a single channel
    # This fix added by Claude
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
                          
    T = get(ARITH_TYPES, arith_label) do
        error("Unknown arithmetic type: $arith_label. Valid options: $(join(keys(ARITH_TYPES), ", "))")
    end
    m = get(MODEL_TYPES, model) do
        error("Unknown model type: $model. Valid options: $(join(keys(MODEL_TYPES), ", "))")
    end
    d = get(DATASETS, dataset) do
        error("Unknown dataset: $dataset. Valid options: $(join(keys(DATASETS), ", "))")
    end
    opt isa MixedPrecision ?
        report_path = joinpath(RESULTS_DIR, model, dataset, "($T, $(typeof(opt).parameters[1]))") * "/" :
        report_path = joinpath(RESULTS_DIR, model, dataset, "$T") * "/"

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
    # First dim of the onehot encoding is the full label set (train ∪ test), so a class
    # missing from the train split still gets an output unit
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
        completion_message = "$T is already complete! \n"
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
        for (x, y) in train_data_loader
            iter += 1
            try
                train_state = step_training(loss, (x, y), train_state)
            catch e
                completion_message = "Model break detected in $T by epoch $epoch: $e \n"
                print(completion_message)
                halted = :error
            end 
            for layer in train_state.parameters
                if (!finite_test(layer))
                    completion_message = "Model break detected in $T by epoch $epoch, iter $iter: Non-finite weights. \n"
                    print(completion_message)
                    halted = :error
                    break
                end
                if (!type_test(layer, T))
                    completion_message = "Model break detected in $T by epoch $epoch, iter $iter: Type promotion. \n"
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
            completion_message = "Early stop for $T at epoch $(bestAccModel[2]). \n"
            print(completion_message)
            halted = :earlystop
        end
        if halted != :error
            @printf("Epoch %i, Type %s, time: %.2f seconds, train accuracy: %.2f, test accuracy: %.2f\n",
                    epoch, opt isa MixedPrecision ? "Mixed: ($T, $(typeof(opt).parameters[1]))" : "$T",
                    (time() - stime), (train_accuracy * 100), (test_accuracy * 100))
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

@everywhere function obtain_optimizer(arith::String)
    return OptimiserChain(WeightDecay(0.001), YunAdam(; eta = 0.001, epsilon = 1e-4))
end

@everywhere function obtain_optimizer(arith::Vector)
    if length(arith) != 2
        error("Mixed precision requires exactly two formats (received $arith)")
    end
    if arith[1] == arith[2]
        println("Warning: attempted mixed-precision with $(arith[1]) as both types. Running in uniform-precision.")
        return obtain_optimizer(arith[1])
    end
    T = get(ARITH_TYPES, arith[2]) do
        error("Unknown arithmetic type: $(arith[2]). Valid options: $(join(keys(ARITH_TYPES), ", "))")
    end
    return MixedPrecision(T, OptimiserChain(WeightDecay(0.001), YunAdam(; eta = 0.001, epsilon = 1e-4)))
end


function main()
    args = warmup_args

    # Claude used to convert to pmap
    results = pmap(arith_list; #=on_error = identity=#) do arith
        benchmark_model(arith isa Vector ? arith[1] : arith, obtain_optimizer(arith), args["model"], args["dataset"];
                        epochs=args["epochs"], batch_size=args["batch-size"], seed=args["seed"],
                        patience=args["patience"], improvement_epsilon=args["improvement-epsilon"])
    end

    report_path = joinpath(RESULTS_DIR, args["model"], args["dataset"]) * "/"

    # Only runs after all training is done; should be in the clear to assume done = true in all progress controllers
    csv_log = "\n~~~~\n" * CSVPrinter(report_path, arith_format.(arith_list))
    if !(csv_log == "") results = [results; csv_log] end
    logs = open(report_path * "log.txt", "a")
    for log in results
        write(logs, log)
    end
    close(logs)

    return results
end

main()