using ConcreteStructs
using Lux
using NNlib
using ArgCheck
using Random

#region LeNet5
function LeNet5(targets::Integer, in_chs::Integer = 1)
    return Chain(
        Conv((5,5), in_chs => 6, tanh; stride = 1),
        AdaptiveMeanPool((14, 14)),
        Conv((5,5), 6 => 16, tanh; stride = 1),
        AdaptiveMeanPool((5, 5)),
        Conv((5,5), 16 => 120, tanh; stride = 1),
        FlattenLayer(),
        Dense(120 => 84, tanh),
        Dense(84 => targets),
        softmax
    )
end
#endregion

#region NetworkInNetwork
function NINBlock(k::Integer, in_chs::Integer, out_chs::Integer, activation::Function, stride::Integer, pad::Integer)
    return Chain(
        Conv((k, k), in_chs => out_chs, activation; stride = stride, pad = pad),
        Conv((1, 1), out_chs => out_chs, activation; stride = 1),
        Conv((1, 1), out_chs => out_chs, activation; stride = 1)
    )
end

function SmallDropoutNIN(targets::Integer, in_chs::Integer = 1)
    return Chain(
        Chain(
            NINBlock(3, in_chs, 16, relu, 2, 1),
            Dropout(0.5),
            NINBlock(3, 16, 32, relu, 2, 1),
            Dropout(0.5),
            NINBlock(3, 32, 64, relu, 2, 1),
            Dropout(0.5),
        ),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(64 => targets),
        softmax
    )
end


function SmallBatchNormNIN(targets::Integer, in_chs::Integer = 1)
    return Chain(
        Chain(
            NINBlock(3, in_chs, 16, relu, 2, 1),
            BatchNorm(16),
            NINBlock(3, 16, 32, relu, 2, 1),
            BatchNorm(32),
            NINBlock(3, 32, 64, relu, 2, 1),
            BatchNorm(64),
        ),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(64 => targets),
        softmax
    )
end
#endregion

#region ResNet18
function ResNetBlock(in_chs::Integer, out_chs::Integer; stride::Integer = 1)
    return Chain(
        Parallel(+,
            Chain(
                Conv((3,3), in_chs => out_chs, stride = stride, pad = 1, use_bias = false, init_weight = kaiming_normal),
                BatchNorm(out_chs),
                relu,
                Conv((3,3), out_chs => out_chs, stride = 1, pad = 1, use_bias = false, init_weight = kaiming_normal),
                BatchNorm(out_chs),
            ),
            (stride == 1 && in_chs == out_chs) ? NoOpLayer() : Chain(
                Conv((1,1), in_chs => out_chs, stride = stride, use_bias = false, init_weight = kaiming_normal),
                BatchNorm(out_chs),
            )
        ),
        relu
    )
end

function ResNetBlockLayer(in_chs::Integer, out_chs::Integer, stride::Integer)
    return Chain(
        ResNetBlock(in_chs, out_chs, stride = stride),
        ResNetBlock(out_chs, out_chs, stride = 1)
    )
end

# As with other similar models, size makes this untenable for posit testing
function ResNet18(targets::Integer, in_chs::Integer = 3)
    return Chain(
        Conv((3, 3), in_chs => 64, stride = 1, pad = 1, use_bias = false, init_weight = kaiming_normal),
        BatchNorm(64),
        relu,
        ResNetBlockLayer(64, 64, 1),
        ResNetBlockLayer(64, 128, 2),
        ResNetBlockLayer(128, 256, 2),
        ResNetBlockLayer(256, 512, 2),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(512, targets)
    )
end

function TinyResNet(targets::Integer, in_chs::Integer = 3)
    return Chain(
        Conv((3,3), in_chs => 8, stride = 1, pad = 1, use_bias = false, init_weight = kaiming_normal),
        BatchNorm(8),
        relu,
        ResNetBlockLayer(8, 16, 1),
        ResNetBlockLayer(16, 32, 2),
        ResNetBlockLayer(32, 32, 2),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(32, targets)
    )
end
#endregion

#region Squeezenet
function FireModule(in_chs::Integer, squeeze::Integer, expand::Integer)
    if (expand % 2 == 1)
        error("Fire module needs an even output count!")
    end
    expand = expand ÷ 2 # Distribution between the two conv paths
    return Chain(
        Conv((1,1), in_chs => squeeze, relu),
        Parallel(
            (a, b) -> cat(a, b; dims = 3),
            Conv((1,1), squeeze => expand, relu),
            Conv((3,3), squeeze => expand, relu; pad = SamePad()),
        )
    )
end

# Reference only
function SqueezeNet1(targets::Integer, in_chs::Integer = 3)
    return Chain(
        Conv((7,7), in_chs => 96, relu; stride = 2),
        MaxPool((3,3); stride = 2),
        FireModule(96, 16, 128),
        FireModule(128, 16, 128),
        FireModule(128, 32, 256),
        MaxPool((3,3); stride = 2),
        FireModule(256, 32, 256),
        FireModule(256, 48, 384),
        FireModule(384, 48, 384),
        FireModule(384, 64, 512),
        MaxPool((3,3); stride = 2),
        FireModule(512, 64, 512),
        Dropout(0.5),
        Conv((1,1), 512 => targets; init_weight = truncated_normal(; std = 0.01f0), init_bias = zeros32),
        relu,
        GlobalMeanPool(),
        FlattenLayer()
    )
end

# Simple model for CIFAR10
function TinySqueezeNet(targets::Integer, in_chs::Integer = 3)
    return Chain(
        Conv((5,5), in_chs => 16, relu; stride = 1, pad = 1),
        FireModule(16, 8, 32),
        FireModule(32, 16, 64),
        MaxPool((3,3); stride = 2),
        Conv((3,3), 64 => 128, relu; stride = 2, pad = 1),
        FireModule(128,20,64),
        MaxPool((3,3); stride = 2),
        Dropout(0.2),
        Conv((1,1), 64 => targets; init_weight = truncated_normal(; std = 0.01f0), init_bias = zeros32),
        GlobalMeanPool(),
        FlattenLayer()
    )
end
#endregion

#region ViT_support
# Largely mirrors Boltz.jl's implementation
# However, compatibility issues mean we can't just use Boltz.jl
function flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

# returns a 1x1x1...xsize(x, N) array of ones matching the type of the input array
# produces new array, doesn't mutate, because Zygote doesn't like mutation
function ones_batch_like(x::AbstractArray{T, N}) where {T, N}
    return fill(one(T), ntuple(_ -> 1, N - 1)..., size(x, N))
end

second_dim_mean(x) = dropdims(mean(x; dims=2); dims=2)

#region PatchEmbedding
# These functions are collectively used to divide an image into patches, flatten the patches, and normalize.

@concrete struct PatchEmbedding <: AbstractLuxContainerLayer{(:patch, :flatten, :norm)}
    patch <: AbstractLuxLayer
    flatten <: AbstractLuxLayer
    norm <: AbstractLuxLayer
end

function (pe::PatchEmbedding)(x, ps, st)
    y₁, st₁ = pe.patch(x, ps.patch, st.patch)
    y₂, st₂ = pe.flatten(y₁, ps.flatten, st.flatten)
    y₃, st₃ = pe.norm(y₂, ps.norm, st.norm)
    return y₃, (; patch = st₁, flatten = st₂, norm = st₃)
end

function PatchEmbedding(
    image_size::Dims{N},
    patch_size::Dims{N},
    in_channels::Integer,
    embed_planes::Integer,
    norm_layer = Returns(Lux.NoOpLayer()),
    flatten::Bool = true
) where {N}
    foreach(zip(image_size, patch_size)) do (i, p)
        @argcheck i % p == 0 "Image size ($i) must be divisible by patch size ($p)"
    end

    return PatchEmbedding(
        Lux.Conv(patch_size, (in_channels => embed_planes); stride = patch_size),
        ifelse(flatten, Lux.WrappedFunction(flatten_spatial), Lux.NoOpLayer()),
        norm_layer(embed_planes)
    )
end

#endregion

#region ClassTokens
# These are used to concatenate the special class tokens with the flattened patch data

@concrete struct ClassTokens <: AbstractLuxLayer
    dim::Int
    init
end

ClassTokens(dim::Int; init = zeros32) = ClassTokens(dim, init)

LuxCore.initialparameters(rng::AbstractRNG, c::ClassTokens) = (; token = c.init(rng, c.dim))

function (m::ClassTokens)(x::AbstractArray{T,N}, ps, st) where {T,N}
    tokens = reshape(ps.token, :, ntuple(_ -> 1, N - 1)...) .* ones_batch_like(x)
    return cat(x, tokens; dims = Val(N - 1)), st
end

#endregion

#region ViPosEmbedding
# These serve to add positional data to the flattened patch data

@concrete struct ViPosEmbedding <: AbstractLuxLayer
    embedding_size::Int
    number_patches::Int
    init
end

function ViPosEmbedding(embedding_size::Int, number_patches::Int; init=randn32)
    return ViPosEmbedding(embedding_size, number_patches, init)
end

function LuxCore.initialparameters(rng::AbstractRNG, v::ViPosEmbedding)
    return (; vectors = v.init(rng, v.embedding_size, v.number_patches))
end

(v::ViPosEmbedding)(x, ps, st) = x .+ ps.vectors, st
#endregion

#region VisionTransformerEncoder

function VisionTransformerEncoder(
    in_planes, depth, number_heads; mlp_ratio = 4.0f0, dropout_rate = 0.0f0
)
    hidden_planes = floor(Int, mlp_ratio * in_planes)
    layers = [
        Lux.Chain(
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm((in_planes, 1); dims = 1, affine = true),
                    Lux.MultiHeadAttention(
                        in_planes; nheads = number_heads,
                        attention_dropout_probability = dropout_rate
                    ),
                    Lux.WrappedFunction(first),
                    Lux.Dropout(dropout_rate)
                ),
                +
            ),
            Lux.SkipConnection(
                Lux.Chain(
                    Lux.LayerNorm((in_planes, 1); dims = 1, affine = true),
                    Lux.Dense(in_planes => hidden_planes, NNlib.gelu),
                    Lux.Dropout(dropout_rate),
                    Lux.Dense(hidden_planes => in_planes),
                    Lux.Dropout(dropout_rate)
                ),
                +
            )
        ) for _ in 1:depth
    ]
end

#endregion

#region VisionTransformer

function VisionTransformer(;
    imsize::Dims{2}=(256,256),
    in_channels::Integer=3,
    patch_size::Dims{2}=(16,16),
    embed_planes::Integer=768,
    depth::Integer=6,
    number_heads=16,
    mlp_ratio=4.0f0,
    dropout_rate=0.1f0,
    embedding_dropout_rate=0.1f0,
    num_classes::Integer=1000,
    pool::Symbol = :class
    )
    @argcheck pool in (:class, :mean)
    return Chain(
        PatchEmbedding(imsize, patch_size, in_channels, embed_planes),
        ClassTokens(embed_planes),
        ViPosEmbedding(embed_planes, prod(imsize .÷ patch_size) + 1),
        Dropout(embedding_dropout_rate),
        VisionTransformerEncoder(embed_planes, depth, number_heads; mlp_ratio = mlp_ratio, dropout_rate = dropout_rate),
        WrappedFunction(ifelse(pool === :class, x -> x[:, 1, :], second_dim_mean)),
        LayerNorm((embed_planes); affine = true),
        Dense(embed_planes, num_classes)
    )
end
#endregion

#endregion

#region ViT_models
# Too large to test practically
function VitBase(targets::Integer, in_chs::Integer = 3)
    return VisionTransformer(in_channels = in_chs, num_classes = targets)
end

# Therefore, a smaller version (not 'tiny', because that's spoken for and still much larger)
function MicroscopicVit(targets::Integer, in_chs::Integer = 3)
    return VisionTransformer(imsize = (32, 32), in_channels = in_chs, patch_size = (4,4), embed_planes = 72, depth = 5,
                             number_heads = 6, mlp_ratio = 2.0, num_classes = targets)
end
#endregion

#region Chimera models
function Chimera(targets::Integer, in_chs::Integer)
    return Chain(
        Conv((5,5), in_chs => 6, tanh; stride = 1),
        AdaptiveMeanPool((14, 14)),
        NINBlock(3, 6, 16, relu, 2, 1),
        ResNetBlockLayer(16, 32, 1),
        AdaptiveMeanPool((5, 5)),
        FireModule(32, 16, 32),
        BatchNorm(32),
        Conv((1, 1), 32 => 1),
        FlattenLayer(),
        Dense(25 => targets)
    )
end
#endregion