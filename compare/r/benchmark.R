# Times shrinkGPR() training for the configurations written by compare/benchmark.py.
# Usage: Rscript benchmark.R <configs.json> <results.json>

args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "common.R"))

input <- read_input(args[1])
D <- function(x) as_array(dec(x, torch_float64()))

fit <- function(df, flow, device, n_epochs) {
  shrinkGPR(y ~ ., data = df, flow_func = flows[[flow]], n_epochs = n_epochs,
            auto_stop = FALSE, display_progress = FALSE, device = torch_device(device))
}

warmed <- character(0)
results <- lapply(input$configs, function(cfg) {
  df <- data.frame(y = D(cfg$y), D(cfg$x))
  key <- paste(cfg$flow, cfg$device)
  if (!key %in% warmed) {
    invisible(fit(df, cfg$flow, cfg$device, 5))
    warmed <<- c(warmed, key)
  }
  torch_manual_seed(1)
  elapsed <- system.time(res <- fit(df, cfg$flow, cfg$device, cfg$n_epochs))[["elapsed"]]
  list(seconds = elapsed, epochs = sum(!is.na(res$loss_stor)), loss = res$loss)
})

write_output(list(results = results, threads = torch_get_num_threads(),
                  torch_version = as.character(packageVersion("torch"))), args[2])
