# Scott-Knott ESD ranking of GPT2SP, Deep-SE, Mean and Median (Tantithamthavorn et al.), as in the paper.
#
# Input : results/abs_errors_within.csv  (project, issuekey, method, abs_error), from build_tables.py
# Output: results/tables/scottknott_within.csv  (project, method, rank, mae, n)
#
# Per project, the methods are ranked on their per-issue absolute errors (same issues for every method).
# The "ALL (per-project MAE)" rows rank the methods on their per-project MAEs over the projects
# where every method has results (Deep-SE has no Talend projects in Tawosi's package).
# sk_esd ranks larger values as better, so errors are negated; rank 1 = lowest error.
#
# Usage: Rscript replication/scottknott.R
suppressPackageStartupMessages(library(ScottKnottESD))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
root <- dirname(here)
ae <- read.csv(file.path(root, "results", "abs_errors_within.csv"), stringsAsFactors = FALSE)

rank_methods <- function(wide) {
  sk <- sk_esd(-wide, version = "np")
  data.frame(method = names(sk$groups), rank = as.integer(sk$groups),
             mae = colMeans(wide)[names(sk$groups)], n = nrow(wide), stringsAsFactors = FALSE)
}

out <- list()
per_project <- list()
for (p in unique(ae$project)) {
  d <- ae[ae$project == p, ]
  wide <- reshape(d[, c("issuekey", "method", "abs_error")], idvar = "issuekey",
                  timevar = "method", direction = "wide")
  rownames(wide) <- wide$issuekey
  wide$issuekey <- NULL
  names(wide) <- sub("^abs_error\\.", "", names(wide))
  wide <- wide[complete.cases(wide), , drop = FALSE]
  r <- rank_methods(wide)
  out[[p]] <- cbind(project = p, r)
  per_project[[p]] <- colMeans(wide)
}

all_methods <- unique(ae$method)
m <- do.call(rbind, lapply(per_project, function(v) v[all_methods]))
m <- as.data.frame(m[complete.cases(m), , drop = FALSE])
out[["ALL"]] <- cbind(project = "ALL (per-project MAE)", rank_methods(m))

res <- do.call(rbind, out)
rownames(res) <- NULL
dir.create(file.path(root, "results", "tables"), showWarnings = FALSE)
write.csv(res, file.path(root, "results", "tables", "scottknott_within.csv"), row.names = FALSE)
print(res[res$project == "ALL (per-project MAE)", ])
cat("\nRank-1 methods per project:\n")
print(table(res$method[res$rank == 1 & res$project != "ALL (per-project MAE)"]))
