write_cytofruv_fixture_provenance <- function(
  root,
  input,
  markers,
  replicate_map,
  clustering_seed,
  number_of_clusters
) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  scale_path <- file.path(root, "preprocessing_manifest.txt")
  design_path <- file.path(root, "experimental_design.txt")
  map_path <- file.path(root, "replicate_map.csv")
  clustering_path <- file.path(root, "clustering_provenance.json")
  writeLines("Synthetic test fixture already transformed as arcsinh(intensity/5).", scale_path)
  writeLines("Synthetic test fixture with a deliberately remeasured reference specimen.", design_path)
  write.csv(replicate_map, map_path, row.names = FALSE, quote = TRUE)
  cluster_provenance <- list(
    schema_version = "final_xu.cytofruv.clustering_provenance.v1",
    method = "CytofRUV::cluster_data",
    upstream_commit = "7088af8794ede0fc7e3c6ef545638e279e8101eb",
    cluster_source_sha256 = "2154ae39693cc3036e59c824b14a6ddc7d76f0c776d19e997b7743eadbb6f937",
    cell_cluster_sha256 = cytofruv_cell_cluster_sha256(input),
    type_markers = as.list(markers),
    number_of_clusters = as.integer(number_of_clusters),
    seed = as.integer(clustering_seed)
  )
  jsonlite::write_json(
    cluster_provenance,
    clustering_path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null"
  )
  list(
    scale_path = normalizePath(scale_path, mustWork = TRUE),
    scale_sha256 = cytofruv_sha256(scale_path),
    design_path = normalizePath(design_path, mustWork = TRUE),
    design_sha256 = cytofruv_sha256(design_path),
    map_path = normalizePath(map_path, mustWork = TRUE),
    map_sha256 = cytofruv_sha256(map_path),
    clustering_path = normalizePath(clustering_path, mustWork = TRUE),
    clustering_sha256 = cytofruv_sha256(clustering_path)
  )
}
