using HNSW
using Distances
using NPZ

const K_DEFAULT         = 1
const M_DEFAULT         = 16
const EF_CONSTR_DEFAULT = 200
const EF_SEARCH_DEFAULT = 50

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

"""
run_hnsw(base_path, query_path; K, M, ef_constr, ef_search)

Retorna:
  idxs  :: Array{Int64,2}   (n_queries × K)
  dists :: Array{Float64,2} (n_queries × K)
"""
function run_hnsw(
    base_path::String,
    query_path::Union{Nothing,String}=nothing;
    K::Int = K_DEFAULT,
    M::Int = M_DEFAULT,
    ef_constr::Int = EF_CONSTR_DEFAULT,
    ef_search::Int = EF_SEARCH_DEFAULT,
)

    println("=== HNSW.jl para embeddings ===")

    # 1. Base
    base_data, nb, dim = load_npy_as_vectors(base_path)
    println("Base carregada: $nb vetores, dimensão = $dim")

    # 2. Queries
    queries_data = nothing
    if query_path === nothing
        println("Usando 10 vetores da própria base como queries.")
        nq = min(10, nb)
        queries_data = base_data[1:nq]
    else
        queries_data, nq, dim_q = load_npy_as_vectors(query_path)
        println("Queries carregadas: $nq vetores, dimensão = $dim_q")
        if dim_q != dim
            error("Dimensão das queries ($dim_q) difere da base ($dim).")
        end
    end

    # 3. Índice HNSW
    println("\nCriando índice HNSW...")
    hnsw = HierarchicalNSW(
        base_data;
        metric = Euclidean(),
        M = M,
        efConstruction = ef_constr,
        ef = ef_search,
    )
    println("Construindo o grafo (add_to_graph!)...")
    add_to_graph!(hnsw)
    println("Grafo HNSW construído.\n")

    # 4. Busca k-NN
    println("Fazendo buscas k-NN para $(length(queries_data)) queries, k = $K ...")
    idxs, dists = knn_search(hnsw, queries_data, K)

    # Converte Vector{Vector{T}} em Matrix (nq × K)
    idxs_mat  = reduce(hcat, idxs)'    # cada linha = vizinhos de 1 query
    dists_mat = reduce(hcat, dists)'

    return idxs_mat, dists_mat
end
