using HNSW
using Distances
using NPZ

const K_DEFAULT         = 1
const M_DEFAULT         = 16
const EF_CONSTR_DEFAULT = 200
const EF_SEARCH_DEFAULT = 50

# continua existindo se ainda quiser carregar de .npy via NPZ
function load_npy_as_vectors(path::String)
    println("Lendo arquivo: $path")
    arr = npzread(path)

    if ndims(arr) != 2
        error("Esperado array 2D em $path, mas veio com dimensão $(ndims(arr))")
    end

    n, d = size(arr)
    println("Shape de $path: ($n, $d)")

    data = [vec(arr[i, :]) for i in 1:n]
    return data, n, d
end

# helper para converter matriz em Vector{Vector}
function matrix_to_vecs(arr::AbstractMatrix)
    if ndims(arr) != 2
        error("Esperado array 2D, mas veio com dimensão $(ndims(arr))")
    end
    n, d = size(arr)
    data = [vec(arr[i, :]) for i in 1:n]
    return data, n, d
end

"Constrói o índice HNSW e o retorna."
function build_hnsw(
    base_arr::AbstractMatrix;
    M::Int = M_DEFAULT,
    ef_constr::Int = EF_CONSTR_DEFAULT,
    ef_search::Int = EF_SEARCH_DEFAULT,
)
    println("=== Construindo índice HNSW (arrays em memória) ===")
    base_data, nb, dim = matrix_to_vecs(base_arr)
    println("Base: $nb vetores, dimensão = $dim")

    t_build = @elapsed begin
        hnsw = HierarchicalNSW(
            base_data;
            metric = Euclidean(),
            M = M,
            efConstruction = ef_constr,
            ef = ef_search,
        )
        add_to_graph!(hnsw)
    end
    println("Índice HNSW construído. Tempo de construção = $(t_build) s\n")

    return hnsw
end

"Faz busca k-NN usando um índice HNSW já construído."
function search_hnsw(
    hnsw::HierarchicalNSW,
    query_arr::AbstractMatrix;
    K::Int = K_DEFAULT,
)
    println("=== Buscando em índice HNSW ===")
    queries_data, nq, dim_q = matrix_to_vecs(query_arr)
    println("Queries: $nq vetores, dimensão = $dim_q")

    t_search = @elapsed begin
        idxs, dists = knn_search(hnsw, queries_data, K)
        # Converte Vector{Vector{T}} em Matrix (nq × K)
        global idxs_mat  = reduce(hcat, idxs)'
        global dists_mat = reduce(hcat, dists)'
    end
    println("Buscas k-NN concluídas. Tempo de busca = $(t_search) s\n")

    return idxs_mat, dists_mat, t_search
end

"""
Convenience: mantém a interface antiga
run_hnsw(base_arr, query_arr; ...)

– constrói o índice e já faz a busca.
"""
function run_hnsw(
    base_arr::AbstractMatrix,
    query_arr::Union{Nothing,AbstractMatrix}=nothing;
    K::Int = K_DEFAULT,
    M::Int = M_DEFAULT,
    ef_constr::Int = EF_CONSTR_DEFAULT,
    ef_search::Int = EF_SEARCH_DEFAULT,
)
    if query_arr === nothing
        println("Query não fornecida; usando 10 vetores da própria base como queries.")
        nb = size(base_arr, 1)
        nq = min(10, nb)
        query_arr = base_arr[1:nq, :]
    end

    hnsw = build_hnsw(base_arr; M=M, ef_constr=ef_constr, ef_search=ef_search)
    return search_hnsw(hnsw, query_arr; K=K)
end
